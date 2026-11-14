import copy
import unittest
import torch
from scripts.prepare_direct_v9 import direct
from tide_lab.direct_semantic import DirectSemanticLM
from tide_lab.direct_specialized import SpecializedSemanticLM
from tide_lab.special_attention import weighted_attention
from tide_lab.direct_semantic_ops import weighted_softmax,time_weights


class SpecializedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_weighted_attention_gradient_and_zero_boundary(self):
        torch.manual_seed(1)
        q,k,v=[torch.randn(2,9,4,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        w=torch.rand(2,9,dtype=torch.float64);w[:,0]=0;w.requires_grad_()
        for rate in (0.,.099):
            a,mask,_=time_weights(w,7,rate)
            expected=weighted_softmax(q@k.transpose(-1,-2)/2,a,mask)@v
            actual=weighted_attention(q,k,v,w,rate=rate,stride=7)
            torch.testing.assert_close(expected,actual,atol=1e-12,rtol=1e-12)
            g=torch.randn_like(expected)
            eg=torch.autograd.grad((expected*g).sum(),(q,k,v,w),retain_graph=True)
            ag=torch.autograd.grad((actual*g).sum(),(q,k,v,w),retain_graph=True)
            for x,y in zip(eg,ag):torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-11)
            self.assertTrue(torch.equal(ag[-1][:,0],torch.zeros_like(ag[-1][:,0])))

    def test_whole_model_parameter_roles_vjp_and_activity(self):
        for candidate,k in [('send-pooled_attention',None),('temperature1',2)]:
            c=direct('check','hb',2,64,candidate,region_width=4,k=k)
            plan=c['direct']['plan']
            for s in plan['regions']:s.pop('activity_schedule',None)
            a=DirectSemanticLM(plan).double();b=SpecializedSemanticLM(plan).double()
            self.assertEqual(list(a.state_dict()),[k for k in b.state_dict() if k!='_extra_state'])
            x=torch.tensor([[1,2,3],[4,5,6]])
            ya=a(x);yb=b(x)
            torch.testing.assert_close(ya,yb,atol=1e-10,rtol=1e-10)
            ya.square().mean().backward();yb.square().mean().backward()
            for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
                self.assertEqual(n,m);self.assertEqual(p.grad is None,q.grad is None,n)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-10,msg=n)
            self.assertEqual(a.last_metrics['work/full'],b.last_metrics['work/full'])
            self.assertEqual(a.last_metrics['work/candidates'],b.last_metrics['work/candidates'])
            if k is None:self.assertAlmostEqual(b.last_metrics['activity/active_parameters_per_position'],b.total_parameter_count)
            with torch.no_grad():
                other=x.clone();other[:,-1]+=1
                self.assertTrue(torch.equal(b(x)[:,:-1],b(other)[:,:-1]))

    def test_ffn_recompute_gradients(self):
        c=direct('check','hb',2,64,'send-pooled_attention')
        a=SpecializedSemanticLM(c['direct']['plan']);b=SpecializedSemanticLM(c['direct']['plan'],ffn_checkpoint=True)
        x=torch.tensor([[1,2,3]])
        a(x).square().mean().backward();b(x).square().mean().backward()
        for p,q in zip(a.parameters(),b.parameters()):
            self.assertEqual(p.grad is None,q.grad is None)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)

    def test_explicit_npu_request_rejected_on_cpu(self):
        x=torch.ones(1,2,4)
        with self.assertRaises(ValueError):weighted_attention(x,x,x,torch.ones(1,2),implementation='npu_fusion_augmented')

    def test_activity_controller_state_and_eval_isolation(self):
        c=direct('check','hb',2,64,'temperature1',region_width=4,k=2)
        a=SpecializedSemanticLM(c['direct']['plan'],activity_controller={'target':.1,'start_tokens':0})
        for _ in range(8):a.observe_update({'work/full':20,'work/node_slots':100},100)
        self.assertLess(a.activity_state['scale'],1.)
        b=SpecializedSemanticLM(c['direct']['plan'],activity_controller={'target':.1,'start_tokens':0})
        b.load_state_dict(a.state_dict());self.assertEqual(a.activity_state,b.activity_state)
        before=copy.deepcopy(b.activity_state)
        b.eval();b(torch.tensor([[1,2,3]]))
        self.assertEqual(before,b.activity_state)


if __name__=='__main__':unittest.main()
