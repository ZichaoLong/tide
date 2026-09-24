"""A small VJP tolerance alone cannot certify an AdamW update near zero."""
import pytest
import torch
from tidegraph import Graph, Node, Region, Continuation, External
from tidegraph.ops import Model
from tidegraph.native import Native
from tidegraph.reference import run
from tidegraph.compare import equivalent


@pytest.mark.parametrize('kind', ['weighted_mean', 'active_softmax', 'all_softmax'])
@pytest.mark.parametrize('sources', [1, 3, 7])
def test_equal_sources_normalization_preserves_optimizer_update(dtype, kind, sources):
    g = Graph((Node(0, aggregation=kind),), (), (Region(1),), (0,)*sources, (0,))
    def execute(native):
        m = Model(g, width=4, dtype=dtype)
        with torch.no_grad():
            for p in m.input_scale: p.fill_(1)
            for name, p in m.nodes[0].extra.items():
                if name.startswith('agg_'): p.fill_(.07)
        xs = [External(b, p, 0, 0, torch.tensor([.39, .73, -.37, .18], dtype=dtype)*(b+1))
              for b in range(7) for p in range(sources)]
        q = Continuation(g.identity, 7)
        r = Native(g, m, packed=True, aggregate_autograd='batched').run(q, xs, 1, sealed_until=1) if native else run(g, m, q, xs, 1, sealed_until=1)
        opt = torch.optim.AdamW(m.parameters(), lr=.01, eps=1e-8)
        sum(e['content'].square().sum() for e in r.trace).backward()
        gradients = {k: None if p.grad is None else p.grad.clone() for k, p in m.named_parameters()}
        opt.step()
        return r, gradients, m.state_dict(), opt.state_dict()
    equivalent(execute(False), execute(True))
