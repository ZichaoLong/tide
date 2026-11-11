import math
import unittest

import torch
from torch.nn import functional as F

from tide_lab.direct_graph_cost import DenseRegion, DirectGraphLM


class DirectCostChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)

    def test_node_batch_matches_independent_single_node_formula_and_gradients(self):
        block = DenseRegion(3, 8, heads=2).double()
        x = torch.randn(3, 2, 5, 8, dtype=torch.float64, requires_grad=True)
        expected = []
        for i in range(3):
            w = {name: t[i] for name, t in block.weights.items()}
            z = x[i]/torch.sqrt((x[i]*x[i]).mean(-1, keepdim=True)+1e-6)*block.norm_att[i]
            q, k, v = [(z@w[name]).view(2, 5, 2, 4).transpose(1, 2) for name in ('q', 'k', 'v')]
            by_time = []
            for t in range(5):
                att = (q[:, :, t:t+1]@k[:, :, :t+1].transpose(-1, -2)/2).softmax(-1)
                by_time.append((att@v[:, :, :t+1]).transpose(1, 2).reshape(2, 1, 8))
            h = x[i]+torch.cat(by_time, 1)@w['o']
            z = h/torch.sqrt((h*h).mean(-1, keepdim=True)+1e-6)*block.norm_ffn[i]
            expected.append(h+(F.silu(z@w['gate'])*(z@w['up']))@w['down'])
        ref = torch.stack(expected); actual = block(x)
        roots = [x, *block.parameters()]
        cotangent = torch.randn_like(ref)
        a = torch.autograd.grad((actual*cotangent).sum(), roots, retain_graph=True)
        b = torch.autograd.grad((ref*cotangent).sum(), roots)
        torch.testing.assert_close(actual, ref, atol=1e-10, rtol=1e-9)
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-8)

    def test_causality_and_all_parameters_receive_gradients(self):
        graph = dict(nodes=4, regions=2, node_regions=[0, 0, 1, 1], ranks=[1, 2],
                     edges=[[0, 2, 1], [1, 2, 1], [0, 3, 1], [1, 3, 1]], inputs=[0, 1], outputs=[2, 3])
        model = DirectGraphLM(graph, width=8, vocab_size=16)
        ids = torch.tensor([[1, 2, 3, 4, 5], [4, 3, 2, 1, 0]])
        changed = ids.clone(); changed[:, 3:] = 9
        first, second = model(ids), model(changed)
        torch.testing.assert_close(first[:, :3], second[:, :3], atol=1e-6, rtol=1e-5)
        first.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertIs(model.embedding.weight, model.head.weight)


if __name__ == '__main__':
    unittest.main()
