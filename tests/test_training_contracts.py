import os
import random
import pytest
import torch
from cases import gradients, ring
from tidegraph import Continuation, Edge, External, Graph, Node, Region, State
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run


def share(model):
    model.nodes[1] = model.nodes[0]
    model.edge_scale[1] = model.edge_scale[0]


def test_shared_parameters_vjp_and_checkpoint(dtype, tmp_path):
    g, m, q, xs, x, initial = ring(dtype); share(m)
    expected = run(g, m, q, xs, 7, sealed_until=7, mode="hst")
    expected_grad = gradients(objective(expected), m, x, initial)
    g, m, q, xs, x, initial = ring(dtype); share(m)
    actual = Native(g, m, workers=3, packed=True, mode="hst").run(q, xs, 7, sealed_until=7)
    equivalent(expected, actual)
    equivalent(expected_grad, gradients(objective(actual), m, x, initial))
    path = tmp_path / "shared.pt"
    save(path, g, m, actual.continuation)
    _, restored, _, _, _, _ = ring(dtype)
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    with pytest.raises(ValueError, match="sharing"):
        load(path, g, restored)
    equivalent(before, restored.state_dict())
    share(restored)
    rq = load(path, g, restored)
    equivalent(actual.continuation, rq)
    assert restored.nodes[0].weight is restored.nodes[1].weight
    assert restored.edge_scale[0] is restored.edge_scale[1]


def test_inference_mode_reaches_native_workers(dtype):
    g, m, q, xs, _, _ = ring(dtype)
    engine = Native(g, m, workers=3, packed=True)
    with torch.inference_mode():
        result = engine.run(q, xs, 7, sealed_until=7)
    assert all(torch.is_inference(value) for _, _, _, value in result.outputs)


def test_explicit_truncation_detaches_states_and_pending(dtype):
    g, m, q, xs, x, _ = ring(dtype)
    prefix = run(g, m, q, xs, 4, sealed_until=4)
    assert prefix.continuation.pending
    engine = Native(g, m, workers=3, packed=True)
    full = engine.run(prefix.continuation, [], 9, sealed_until=9)
    detached = engine.run(prefix.continuation.detach(), [], 9, sealed_until=9)
    equivalent(full, detached)
    continuous_grad, = torch.autograd.grad(objective(full), (x,), allow_unused=True)
    truncated_grad, = torch.autograd.grad(objective(detached), (x,), allow_unused=True)
    assert continuous_grad is not None and continuous_grad.abs().sum() > 0
    assert truncated_grad is None


@pytest.mark.parametrize("seed", range(6))
def test_random_sparse_cycles_and_ragged_batch(dtype, seed):
    seed += int(os.environ.get("TIDE_TEST_SEED", "7"))
    def fixture():
        rng = random.Random(seed)
        nodes = tuple(Node(v % 2, clear=v == 3) for v in range(5))
        edges = [Edge(v, (v + 1) % 5, rng.randrange(1, 4)) for v in range(5)]
        edges += [Edge(rng.randrange(5), rng.randrange(5), rng.randrange(1, 4)) for _ in range(4)]
        g = Graph(nodes, tuple(edges), (Region(1), Region(1, observe_all=False)), (0, 1), (3, 4))
        m = Model(g, dtype=dtype, seed=seed)
        x = torch.randn(18, 3, dtype=dtype, generator=torch.Generator().manual_seed(seed)).requires_grad_()
        xs = []
        for b in range(3):
            for p in range(2):
                for i, time in enumerate(sorted(rng.sample(range(9), 3))):
                    xs.append(External(b, p, i, time, x[b * 6 + p * 3 + i]))
        initial = torch.full((3, 5, 3), 0.03, dtype=dtype, requires_grad=True)
        q = Continuation(g.identity, 3, states={(b, v): State(initial[b, v]) for b in range(3) for v in range(5)})
        return g, m, q, xs, x, initial
    g, m, q, xs, x, initial = fixture()
    expected = run(g, m, q, xs, 12, sealed_until=12, mode="hst")
    expected_grad = gradients(objective(expected), m, x, initial)
    g, m, q, xs, x, initial = fixture()
    actual = Native(g, m, workers=3, packed=True, mode="hst").run(q, xs, 12, sealed_until=12)
    equivalent(expected, actual)
    equivalent(expected_grad, gradients(objective(actual), m, x, initial))


@pytest.mark.parametrize("native", [False, True])
def test_input_position_holes_are_rejected(dtype, native):
    g, m, q, xs, _, _ = ring(dtype)
    xs[0].position = 10
    with pytest.raises(ValueError, match="contiguous"):
        if native:
            Native(g, m).run(q, xs, 7, sealed_until=7)
        else:
            run(g, m, q, xs, 7, sealed_until=7)
