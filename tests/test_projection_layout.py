"""Physical QKV/output strides preserve owners, aliases, VJPs and updates."""
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.fiber_attention import PROFILE
from tidegraph.ops import Model
from tidegraph.native import Native
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.checkpoint import save, load


@pytest.mark.parametrize('schedule', ['streaming', 'frontier', 'chain', 'settle'])
def test_projection_strides_shared_owners_update_checkpoint(dtype, schedule, tmp_path):
    g = Graph(tuple(Node(i, memory=PROFILE, query_heads=2, kv_heads=2) for i in range(2)),
              (Edge(0, 1, 1),), (Region(1),)*2, (0,), (1,))
    left = Model(g, width=4, dtype=dtype); right = Model(g, width=4, dtype=dtype, projection_layout='linear')
    left.nodes[1] = left.nodes[0]; right.nodes[1] = right.nodes[0]
    assert left.nodes[0].extra['fiber_qkv'].stride() == (12, 1)
    assert right.nodes[0].extra['fiber_qkv'].stride() == (1, 4)
    identities = [(name, id(p), p.data_ptr(), p.stride()) for name, p in right.named_parameters()]
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, eps=1e-5) for m in (left, right)]
    for step in range(3):
        def execute(m, anchor):
            x = (torch.arange(24, dtype=dtype).reshape(2, 3, 4)/30+step/10).requires_grad_()
            q = Continuation(g.identity, 2)
            if schedule == 'settle':
                spec = SettleGraph(g, (1, 2))
                if anchor: result = settle(spec, m, q, x, mode='hst')
                else:
                    eg, em = spec.embed(m); eq = spec.embed_initial(q, eg); stop = spec.stride*3
                    engine = Native(eg, em, algorithm='frontier', packed=True, workers=3, mode='hst',
                                    attention_packing='single', fiber_pooling='csr', fiber_cache='owned', attention_layout='head')
                    result = spec.project(engine.run(eq, spec.external(x, encoded=True), stop, sealed_until=stop))
            else:
                xs = [External(b, 0, t, 2*t, x[b, t]) for b in range(2) for t in range(3)]
                result = run(g, m, q, xs, 7, sealed_until=7, mode='hst') if anchor else Native(
                    g, m, algorithm=schedule, packed=True, workers=3, mode='hst', attention_packing='single',
                    fiber_pooling='csr', fiber_cache='owned', attention_layout='head').run(q, xs, 7, sealed_until=7)
            return result, x
        a, ax = execute(left, True); b, bx = execute(right, False); equivalent(a, b)
        for m, optimizer in zip((left, right), optimizers): optimizer.zero_grad(set_to_none=True)
        objective(a).backward(); objective(b).backward()
        equivalent(ax.grad, bx.grad)
        equivalent({k:p.grad for k,p in left.named_parameters()}, {k:p.grad for k,p in right.named_parameters()})
        for optimizer in optimizers: optimizer.step()
        equivalent(dict(left.named_parameters()), dict(right.named_parameters()))
        assert identities == [(name,id(p),p.data_ptr(),p.stride()) for name,p in right.named_parameters()]
    path = tmp_path/'layout.pt'; save(path,g,right,b.continuation)
    restored = load(path,g,right); equivalent(b.continuation,restored)
    assert right.nodes[0] is right.nodes[1]


def test_nondefault_layout_rejects_inapplicable_graph(dtype):
    g = Graph((Node(0),), (), (Region(1),), (0,), (0,))
    with pytest.raises(ValueError, match='same-fiber'): Model(g,dtype=dtype,projection_layout='linear')
