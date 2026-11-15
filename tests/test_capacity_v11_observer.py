import unittest
import torch
from scripts.prepare_direct_v9 import direct
from tide_lab.direct_specialized import SpecializedSemanticLM
from scripts.diagnose_capacity_v11 import Observe
from tide_lab.moe import MoETransformer

class ObserverTests(unittest.TestCase):
    def test_hb_hook_preserves_output_and_gradients(self):
        torch.set_num_threads(2)
        c=direct('check','hb',2,64,'temperature1',region_width=4,k=2)
        a=SpecializedSemanticLM(c['direct']['plan']).double()
        b=SpecializedSemanticLM(c['direct']['plan']).double()
        x=torch.tensor([[1,2,3],[4,5,6]])
        obs=Observe(b);obs.valid=torch.tensor([[True,True,True],[True,False,False]]);obs.backward=True
        ya=a(x);yb=b(x);torch.testing.assert_close(ya,yb,atol=0,rtol=0)
        ya.sum().backward();yb.sum().backward()
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
        self.assertTrue(obs.grads);self.assertTrue(obs.export())
        for r in obs.export().values():self.assertLessEqual(sum(r['receipt_valid']),sum(r['receipt_all']))
        obs.remove()
    def test_moe_valid_denominator(self):
        torch.set_num_threads(2)
        m=MoETransformer(width=16,layers=2,hidden=24,experts=8,vocab_size=32)
        x=torch.tensor([[1,2,3],[4,5,6]]);valid=torch.tensor([[True,True,True],[True,False,False]])
        obs=Observe(m)
        m(x,valid_mask=valid)
        for r in obs.export().values():self.assertEqual(sum(r['assignments_valid']),8)
        obs.remove()
