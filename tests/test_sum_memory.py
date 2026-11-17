import copy,unittest
import torch
from tide_lab.sum_memory import sum_read
from tide_lab.direct_semantic_ops import time_weights
from tide_lab.direct_semantic import DirectSemanticLM
from tide_lab.direct_specialized import SpecializedSemanticLM
from tide_lab.hb_mechanisms import HBMechanismLM
from scripts.prepare_direct_v9 import direct


class SumMemoryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_independent_history_forward_vjp_and_empty(self):
        torch.manual_seed(44)
        for length in (3,35,67):
            w=torch.rand(2,2,length,dtype=torch.float64);w[0]=0;w.requires_grad_()
            p=torch.randn(2,2,length,5,dtype=torch.float64,requires_grad=True)
            a,_,_=time_weights(w,7,.099);decay=time_weights(torch.ones(length,dtype=w.dtype),7,.099)[0]
            masses=[a.sum(-1),w.cumsum(-1)]
            expected=[]
            for beta in (.099,0.):
                history=[]
                for t in range(length):
                    weights=w[...,:t+1]*torch.exp(-beta*7*torch.arange(t,-1,-1,dtype=w.dtype))
                    history.append((weights.unsqueeze(-1)*p[...,:t+1,:]).sum(-2)/weights.sum(-1).clamp_min(1).unsqueeze(-1))
                expected.append(torch.stack(history,-2))
            g=[torch.randn_like(x) for x in expected]
            eg=torch.autograd.grad(sum((v*z).sum() for v,z in zip(expected,g)),(w,p),retain_graph=True)
            for mode in ('matrix','blocked'):
                actual=sum_read(w,p,masses,decay,rate=.099,stride=7,implementation=mode)
                for x,y in zip(actual,expected):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)
                ag=torch.autograd.grad(sum((v*z).sum() for v,z in zip(actual,g)),(w,p),retain_graph=True)
                for x,y in zip(ag,eg):torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-11)
                self.assertEqual(actual[0][0].count_nonzero().item(),0)

    def test_whole_model_vjp_and_common_initialization(self):
        plan=direct('check','hb',2,64,'temperature1',region_width=4,k=2)['direct']['plan']
        for s in plan['regions']:s.pop('activity_schedule',None)
        qry=SpecializedSemanticLM(plan).double()
        sp=copy.deepcopy(plan)
        for m in sp['memories']:m['memory_read']='sum'
        a=DirectSemanticLM(sp).double();x=torch.tensor([[1,2,3],[4,5,6]])
        for mode in ('matrix','blocked'):
            b=SpecializedSemanticLM(sp,sum_implementation=mode).double()
            for n,p in b.named_parameters():torch.testing.assert_close(p,dict(qry.named_parameters())[n],atol=0,rtol=0)
            a.zero_grad(set_to_none=True);ya=a(x);yb=b(x)
            torch.testing.assert_close(ya,yb,atol=1e-10,rtol=1e-10)
            ya.square().mean().backward();yb.square().mean().backward()
            for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
                self.assertEqual(n,m);self.assertEqual(p.grad is None,q.grad is None,n)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-10,msg=n)
            other=x.clone();other[:,-1]+=1
            torch.testing.assert_close(b(x)[:,:-1],b(other)[:,:-1],atol=0,rtol=0)
            self.assertEqual(b.last_metrics['work/logical_read_attention_pairs'],0)
            self.assertGreater(b.last_metrics['work/sum_read_candidate_rows'],0)
        c=HBMechanismLM(sp,mechanism={'read_content':True}).double()
        c(x).square().mean().backward()
        self.assertGreater(c.gradient_metrics()['gradient/summary_l2'],0)


if __name__=='__main__':unittest.main()
