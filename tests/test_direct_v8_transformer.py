import unittest
import torch
from tide_lab.language import SmallTransformer
from tide_lab.campaign_runtime import optimizer_for


class TransformerRecipes(unittest.TestCase):
    def test_tying_causality_and_optimizer(self):
        torch.set_num_threads(1)
        for init in ('legacy','gpt2','fan_in_depth'):
            m=SmallTransformer(width=16,layers=2,heads=2,ffn=32,initialization=init)
            self.assertIs(m.embedding.weight,m.head.weight)
            x=torch.tensor([[1,2,3]]);y=m(x)
            changed=x.clone();changed[0,2]=4
            self.assertTrue(torch.equal(y[:,:2],m(changed)[:,:2]))
            c=dict(training=dict(lr=.001,eps=1e-8,weight_decay=.1,decay_policy='matrix_only',betas=[.9,.95]))
            opt=optimizer_for(m,c)
            owned=[p for g in opt.param_groups for p in g['params']]
            self.assertEqual(len(owned),len({id(p) for p in owned}))
            before=m.blocks[0].qkv.weight.detach().clone()
            y.square().mean().backward();opt.step()
            self.assertFalse(torch.equal(before,m.blocks[0].qkv.weight))

    def test_depth_scaling_and_zero_bias(self):
        m=SmallTransformer(width=64,layers=4,heads=4,ffn=128,initialization='gpt2')
        self.assertAlmostEqual(float(m.blocks[0].qkv.weight.std()),.02,delta=.001)
        self.assertAlmostEqual(float(m.blocks[0].out.weight.std()),.02/(8**.5),delta=.0005)
        self.assertTrue(torch.equal(m.blocks[0].qkv.bias,torch.zeros_like(m.blocks[0].qkv.bias)))

if __name__=='__main__':unittest.main()
