import copy,unittest
import torch
from scripts.prepare_direct_v9 import direct
from tide_lab.capacity_balance import CapacityBalancedLM,opportunity_counts,bias_step
from tide_lab.direct_specialized import SpecializedSemanticLM
from tide_lab.campaign_runtime import make_model,optimizer_for,logical_update,exact_tree

class BalanceChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.c=direct('check','hb',2,64,'temperature1',region_width=4,k=2)
        self.c['kind']='direct_specialized';self.c.pop('planned_inventory',None)
        for s in self.c['direct']['plan']['regions']:s.pop('activity_schedule',None)
        self.c['implementation']={'routing_balance':dict(kind='candidate-load-bias-v1',rate=.01,start_tokens=0,max_abs_bias=16.)}
    def test_absence_padding_bias_direction(self):
        rec=torch.tensor([[[1,1,1]],[[1,1,1]],[[0,0,1]]],dtype=torch.bool)
        full=rec.clone();full[1]=False
        q,f,p=opportunity_counts(rec,full,torch.tensor([[True,True,False]]))
        self.assertEqual(q.tolist(),[1,1,0]);self.assertEqual(f.tolist(),[2,0,0])
        b=bias_step(torch.zeros(3),q,f,torch.tensor(1.),.01,16.)
        self.assertLess(float(b[0]),0);self.assertGreater(float(b[1]),0);self.assertEqual(float(b[2]),0.)
    def test_zero_bias_function_and_vjp(self):
        base=SpecializedSemanticLM(self.c['direct']['plan']);other=make_model(self.c,torch.device('cpu'))
        x=torch.tensor([[1,2,3],[4,5,6]])
        y=base(x);z=other(x);torch.testing.assert_close(y,z,atol=0,rtol=0)
        y.square().sum().backward();z.square().sum().backward()
        for p,q in zip(base.parameters(),other.parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
    def test_global_batch_and_restore_bias(self):
        a=make_model(self.c,torch.device('cpu')).double();b=make_model(self.c,torch.device('cpu')).double()
        opt=optimizer_for(a,self.c);other=optimizer_for(b,self.c)
        x=torch.tensor([[1,2,3],[4,5,6]]);batch=dict(input_ids=x,labels=x)
        row=logical_update(a,opt,[batch],self.c,0,torch.device('cpu'))
        micros=[{k:v[i:i+1] for k,v in batch.items()} for i in range(2)]
        logical_update(b,other,micros,self.c,0,torch.device('cpu'))
        for j in range(len(a.members)):
            torch.testing.assert_close(getattr(a,f'balance_bias_{j}'),getattr(b,f'balance_bias_{j}'),rtol=0,atol=1e-7)
        restored=make_model(self.c,torch.device('cpu')).double();restored.load_state_dict(a.state_dict());exact_tree(restored.state_dict(),a.state_dict())
        before=copy.deepcopy(restored.state_dict());restored.eval();restored(x);exact_tree(before,restored.state_dict())
        self.assertEqual(int(a.balance_updates),1)
