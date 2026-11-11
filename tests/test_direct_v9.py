import copy
import json
from pathlib import Path
import unittest
import torch
from tide_lab.direct_semantic import Aggregation, DirectSemanticLM
from scripts.prepare_direct_v9 import inlet_tree

ROOT=Path(__file__).resolve().parents[1]


class DirectV9Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_aggregation_value_input_parameter_vjp_and_empty_target(self):
        torch.manual_seed(171)
        for mode in ('region','source'):
            a=Aggregation(3,4,6,mode,None,[(0,0),(1,0),(2,1)],[0,1]).double()
            b=copy.deepcopy(a);b.reduction='matmul'
            z=torch.randn(3,2,3,4,dtype=torch.float64,requires_grad=True)
            mass=torch.rand(3,2,3,dtype=torch.float64,requires_grad=True)
            incidence=torch.tensor([[1.,1.,0.],[0.,1.,2.],[0.,0.,0.]],dtype=torch.float64)
            ya,ra=a(z,mass,incidence,.7);yb,rb=b(z,mass,incidence,.7)
            torch.testing.assert_close(ya,yb,atol=1e-12,rtol=1e-12)
            self.assertEqual(int(torch.count_nonzero(yb[2])),0)
            torch.testing.assert_close(ra,rb,atol=0,rtol=0)
            u=torch.randn_like(ya)
            ga=torch.autograd.grad((ya*u).sum(),(z,mass,*a.parameters()),retain_graph=True)
            gb=torch.autograd.grad((yb*u).sum(),(z,mass,*b.parameters()))
            for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-10)

    def test_tree_preserves_body_initialization_reachability_and_token_time(self):
        c=json.loads((ROOT/'configs/studies/language/direct-v9/table/main/r32-h2-q10-s11.json').read_text())
        p=c['direct']['plan'];q=inlet_tree(p);a=DirectSemanticLM(p);b=DirectSemanticLM(q)
        ap=dict(a.named_parameters());bp=dict(b.named_parameters())
        for key,value in ap.items():self.assertTrue(torch.equal(value,bp[key]),key)
        g=q['graph'];reachable=set(g['inputs'])
        while True:
            nxt=reachable|{v for u,v,_ in g['edges'] if u in reachable}
            if nxt==reachable:break
            reachable=nxt
        self.assertEqual(len(reachable),g['nodes'])
        old=a.memories[0].double();new=b.memories[0].double()
        u=torch.randn(32,1,4,64,dtype=torch.float64);r=torch.rand(32,1,4,dtype=torch.float64)
        x=old(u,r);y=new(u,r)
        torch.testing.assert_close(x['content'],y['content'],atol=1e-11,rtol=1e-10)
        for xx,yy in zip(x['memory'],y['memory']):torch.testing.assert_close(xx,yy,atol=1e-11,rtol=1e-10)

    def test_tree_causal_and_received_unsent_gradient(self):
        c=json.loads((ROOT/'configs/studies/language/direct-v9/table/main/tree-r32-h2-q10-s11.json').read_text())
        m=DirectSemanticLM(c['direct']['plan']);m.set_progress(9*2**20)
        x=torch.tensor([[1,7,13]]);y=m(x);changed=x.clone();changed[:,-1]=19
        future=m(changed)
        torch.testing.assert_close(y[:,:2],future[:,:2],atol=0,rtol=0)
        y.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.selectors[-1].parameters()))
        self.assertLessEqual(m.last_metrics['work/full'],m.last_metrics['work/candidates'])
        self.assertGreater(m.last_metrics['work/boundary_full'],0)


if __name__=='__main__':unittest.main()
