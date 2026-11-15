import copy,unittest
import torch
from tide_lab.hb_mechanisms import SharedNodeLinear,smooth_top2,HBMechanismLM
from tide_lab.direct_semantic_ops import NodeLinear,allocation
from tide_lab.direct_specialized import SpecializedSemanticLM
from tide_lab.campaign_runtime import exact_tree
from scripts.prepare_direct_v9 import direct


class HBMechanisms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def plan(self):
        p=direct('check','hb',2,64,'temperature1',region_width=4,k=2)['direct']['plan']
        for s in p['regions']:s.pop('activity_schedule',None)
        return p

    def test_shared_vjp_sums_nodes_and_private_adapters(self):
        torch.manual_seed(19);m=SharedNodeLinear(NodeLinear(3,4,5),2,11,'check').double()
        x=torch.randn(3,2,4,dtype=torch.float64,requires_grad=True)
        y=m(x);expected=torch.stack([torch.nn.functional.linear(x[i],m.weight+m.up.weight[i]@m.down.weight[i]) for i in range(3)])
        torch.testing.assert_close(y,expected,atol=1e-13,rtol=1e-13)
        g=torch.randn_like(y);inputs=(x,*m.parameters())
        a=torch.autograd.grad((y*g).sum(),inputs,retain_graph=True)
        b=torch.autograd.grad((expected*g).sum(),inputs)
        for u,v in zip(a,b):torch.testing.assert_close(u,v,atol=1e-12,rtol=1e-12)
        self.assertEqual(len(list(m.parameters())),3)

    def test_fixed_budget_single_interior_score_gradient_zero(self):
        t=torch.tensor([[2.,.5,-3.]],dtype=torch.float64,requires_grad=True)
        c,_=allocation(t,torch.ones_like(t),torch.tensor([[1.5]]),torch.tensor([[1.5]]),1.)
        torch.testing.assert_close(c,torch.tensor([[1.,.5,0.]],dtype=torch.float64))
        (c*torch.tensor([[2.,7.,-4.]])).sum().backward()
        self.assertTrue(torch.equal(t.grad,torch.zeros_like(t)))

    def test_centered_quad_same_real_function_and_vjp(self):
        torch.manual_seed(17);x=torch.randn(9,8,dtype=torch.float64,requires_grad=True)
        mu=torch.rand_like(x,requires_grad=True);b=mu.sum(-1,keepdim=True)*.4
        y,_=allocation(x,mu,b,b,.7);z,_=allocation(x-x.amax(-1,keepdim=True).detach(),mu,b,b,.7)
        torch.testing.assert_close(y,z,atol=1e-12,rtol=1e-12)
        g=torch.randn_like(y)
        a=torch.autograd.grad((y*g).sum(),(x,mu),retain_graph=True);c=torch.autograd.grad((z*g).sum(),(x,mu))
        for u,v in zip(a,c):torch.testing.assert_close(u,v,atol=1e-12,rtol=1e-12)

    def test_top2_caps_absence_and_independent_formula(self):
        s=torch.tensor([[2.,1.,-1.,4.],[1.,2.,3.,4.]],dtype=torch.float64,requires_grad=True)
        cap=torch.tensor([[1.,1.,.3,0.],[0.,0.,0.,0.]],dtype=torch.float64)
        c,e=smooth_top2(s,cap,torch.tensor([[1.5],[0.]],dtype=torch.float64),1.)
        p=torch.sigmoid(8*torch.tanh((s[0,0]-s[0,1])/8))
        torch.testing.assert_close(c[0,0],.5+.5*p);torch.testing.assert_close(c[0,1],1.5-c[0,0])
        self.assertTrue(bool(((c>=0)&(c<=cap)).all()));self.assertEqual(int((c[0]>0).sum()),2)
        self.assertTrue(torch.equal(c[1],torch.zeros_like(c[1])));self.assertLess(float(e.max()),1e-12)
        self.assertTrue(torch.autograd.gradcheck(lambda x:smooth_top2(x,cap,torch.tensor([[1.5],[0.]],dtype=torch.float64),1.)[0],(s,)))

    def test_empty_mechanism_preserves_old_model(self):
        p=self.plan();a=SpecializedSemanticLM(p).double();b=HBMechanismLM(p,mechanism={'center_quad':False}).double()
        x=torch.tensor([[1,2,3],[4,5,6]]);ya=a(x);yb=b(x)
        torch.testing.assert_close(ya,yb,atol=0,rtol=0)
        ya.sum().backward();yb.sum().backward()
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(na,nb)
            if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,atol=0,rtol=0)
        self.assertEqual(a.last_metrics['work/active_parameter_uses'],b.last_metrics['work/active_parameter_uses'])

    def test_readout_ports_initial_equivalence_and_all_variant_causality(self):
        p=self.plan();x=torch.tensor([[1,2,3]])
        a=HBMechanismLM(p,mechanism={'readout':'untied_mean'}).double()
        b=HBMechanismLM(p,mechanism={'readout':'ports4'}).double()
        torch.testing.assert_close(a(x),b(x),atol=1e-12,rtol=1e-12)
        for spec in ({'share':'ffn','rank':0},{'share':'ffn','rank':8},{'share':'full','rank':8},
                     {'route':'top2'},{'read_content':True},{'readout':'ports4'},{'position':True},{'residual':True}):
            m=HBMechanismLM(p,mechanism=spec).double();m.set_progress(4*2**20)
            other=x.clone();other[:,-1]+=1
            torch.testing.assert_close(m(x)[:,:-1],m(other)[:,:-1],atol=0,rtol=0)
            m(x).square().mean().backward()
            self.assertTrue(all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None))
            restored=HBMechanismLM(p,mechanism=spec).double();restored.load_state_dict(m.state_dict());exact_tree(m.state_dict(),restored.state_dict())
            self.assertGreater(m.last_metrics['work/active_parameter_uses'],0)
            self.assertLessEqual(m.last_metrics['activity/active_parameters_per_position'],m.total_parameter_count)

if __name__=='__main__':unittest.main()
