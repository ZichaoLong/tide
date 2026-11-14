import copy
import unittest
import torch
from tide_lab.ordered_divide import ordered_divide


class OrderedDivideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_config_migration_rejects_unrelated_changes(self):
        from tide_lab.v10_repair import validate_config_change
        original=dict(kind='direct_specialized',implementation=dict(aggregation='neighbors'),
                      seed=11,training=dict(lr=.001,batch=8))
        changed=copy.deepcopy(original);changed['implementation']['aggregate_vjp']='ordered-v1'
        validate_config_change(original,changed)
        for field,value in [('seed',22),('training',dict(lr=.002,batch=8)),('kind','direct_semantic')]:
            bad=copy.deepcopy(changed);bad[field]=value
            with self.assertRaises(ValueError):validate_config_change(original,bad)

    def test_production_model_dispatch_and_parameter_ownership(self):
        from tide_lab.v10_config import hb
        from tide_lab.campaign_runtime import make_model
        c=hb('ordered-model-check',32,8,None,micro=2)
        a=make_model(c,torch.device('cpu'))
        c['implementation']['aggregate_vjp']='ordered-v1'
        b=make_model(c,torch.device('cpu'));b.load_state_dict(a.state_dict())
        self.assertEqual(b.aggregate_vjp,'ordered-v1')
        self.assertIs(b.embedding.weight,b.head.weight)
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        x=torch.tensor([[1,2,3],[4,5,6]])
        ya=a(x);yb=b(x);self.assertTrue(torch.equal(ya,yb))
        ya.square().mean().backward();yb.square().mean().backward()
        for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(n,m);self.assertEqual(p.grad is None,q.grad is None,n)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5,msg=n)

    def test_actual_tiny_receipt_witnesses_against_float64(self):
        # Two actual R64 coordinates and an R28 subnormal upstream VJP.
        for a,b,g in [(8.686186863960191e-37,4.967183787530951e-38,0.),
                      (-2.724980329536433e-36,8.636574620505972e-38,0.),
                      (2.773051279204547e-37,2.612801141005781e-38,-2.802596928649634e-45),
                      (5e-38,1.2e-38,1e-5)]:
            x,y,seed=[torch.tensor(v,dtype=torch.float32) for v in (a,b,g)]
            x.requires_grad_();y.requires_grad_()
            out=ordered_divide(x,y);grads=torch.autograd.grad(out,(x,y),seed)
            xx=x.detach().double().requires_grad_();yy=y.detach().double().requires_grad_()
            ref=torch.autograd.grad(xx/yy,(xx,yy),seed.double())
            self.assertTrue(torch.equal(out,x/y))
            for actual,expected in zip(grads,ref):
                self.assertTrue(bool(torch.isfinite(actual)))
                torch.testing.assert_close(actual.double(),expected,atol=1e-6,rtol=1e-5)

    def test_broadcast_vjp_and_zero_receipt_boundary(self):
        torch.manual_seed(7)
        x=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
        r=torch.tensor([[0.],[.4]],dtype=torch.float64,requires_grad=True)
        fn=lambda a,b:ordered_divide(a,torch.where(b>0,b,torch.ones_like(b)))
        out=fn(x,r);gx,gr=torch.autograd.grad(out,(x,r),torch.ones_like(out))
        self.assertEqual(gr[0].item(),0.)
        self.assertTrue(torch.equal(gx[0],torch.ones_like(gx[0])))
        positive=torch.rand(2,1,dtype=torch.float64,requires_grad=True)+.2
        self.assertTrue(torch.autograd.gradcheck(ordered_divide,(x,positive)))

    def test_true_unrepresentable_vjp_is_not_silently_replaced(self):
        x=torch.tensor(5e-38,requires_grad=True);r=torch.tensor(1.2e-38,requires_grad=True)
        grads=torch.autograd.grad(ordered_divide(x,r),(x,r),torch.tensor(10.))
        self.assertTrue(any(not bool(torch.isfinite(g)) for g in grads))


if __name__=='__main__':unittest.main()
