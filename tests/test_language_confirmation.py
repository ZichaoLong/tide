"""Independent checks for the changed confirmation and pointwise-control paths."""
import unittest
import torch
from scripts.prepare_language_confirmation import select_confirmation
from tide_lab.language import LocalReplacement
from tide_lab.bottleneck_checks import reference


class ConfirmationTests(unittest.TestCase):
    def test_selection_excludes_observed_articles_and_is_order_independent(self):
        docs=[dict(id=str(i),split='test') for i in range(25)]
        docs+=[dict(id='train',split='train')]
        old={'selected_documents':{'test':[dict(id=str(i)) for i in range(5)]}}
        a=select_confirmation(docs,old);b=select_confirmation(list(reversed(docs)),old)
        self.assertEqual(a,b);self.assertEqual(len(a),16)
        self.assertTrue(all(d['id'] not in {'train','0','1','2','3','4'} for d in a))
        with self.assertRaisesRegex(ValueError,'insufficient'):select_confirmation(docs[:6],old)

    def test_pointwise_value_and_all_gradients_match_manual_chain_rule(self):
        torch.set_num_threads(1);torch.manual_seed(37)
        stats=dict(x_mean=torch.randn(6),y_mean=torch.randn(6),x_scale=torch.tensor(1.4),y_scale=torch.tensor(.7))
        m=LocalReplacement('bottleneck',{'width':3},stats,dimension=6,bottleneck_width=8).double()
        x=torch.randn(4,6,dtype=torch.double,requires_grad=True)
        torch.testing.assert_close(m(x),m.y_mean.expand_as(x),rtol=0,atol=0)
        with torch.no_grad():m.output.weight.normal_(std=.2)
        u=torch.randn_like(x);value=m(x);params=dict(m.named_parameters())
        actual=torch.autograd.grad((value*u).sum(),[x,*params.values()])
        expected,grads=reference(m,x,u)
        torch.testing.assert_close(value,expected,rtol=1e-12,atol=1e-12)
        for key,gradient in zip(['input',*params],actual):
            torch.testing.assert_close(gradient,grads[key],rtol=1e-10,atol=1e-12)

    def test_real_control_parameter_budget_and_same_rank_input(self):
        stats=dict(x_mean=torch.zeros(896),y_mean=torch.zeros(896),x_scale=torch.tensor(1.),y_scale=torch.tensor(1.))
        m=LocalReplacement('bottleneck',{'width':32},stats)
        self.assertEqual(sum(p.numel() for p in m.parameters()),190464)
        self.assertEqual(tuple(m.input.weight.shape),(32,896))
        self.assertEqual(tuple(m.output.weight.shape),(896,32))


if __name__=='__main__':unittest.main()
