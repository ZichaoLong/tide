import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region
from tidegraph.compare import equivalent
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from isolated_cases import fixture, roots, vjp


IMPLEMENTATIONS = ("python", "python-step", "native-stream", "native-serial",
                   "native-frontier", "native-unpacked", "native-step")


def execute(implementation, g, m, q, xs, mode):
    if implementation.startswith("python"):
        return frontier(g, m, q, xs, 4, sealed_until=4, mode=mode, prefill=implementation == "python")
    engine = Native(g, m, mode=mode, workers=1 if implementation == "native-serial" else 3,
                    packed=implementation != "native-unpacked",
                    algorithm="streaming" if implementation in {"native-stream", "native-serial"} else "frontier",
                    prefill=implementation != "native-step")
    return engine.run(q, xs, 4, sealed_until=4)


@pytest.mark.parametrize("kind", ["ema", "ssm", "attention", "linear", "delta"])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_separate_public_roots_preserve_connectivity(dtype, kind, mode, implementation):
    g, m, q, xs, variables = fixture(dtype, kind)
    expected = run(g, m, q, xs, 4, sealed_until=4, mode=mode)
    expected_roots = roots(expected)
    g, m, q, xs, actual_variables = fixture(dtype, kind)
    actual = execute(implementation, g, m, q, xs, mode)
    equivalent(expected, actual)
    actual_roots = roots(actual)
    for name, root in expected_roots.items():
        assert root.requires_grad == actual_roots[name].requires_grad, name
        # Query the same graph repeatedly, including a connected numerical zero.
        for zero in (False, True):
            grad = vjp(root, variables, zero)
            actual_grad = vjp(actual_roots[name], actual_variables, zero)
            equivalent(grad, actual_grad, name)
            assert grad["upstream.1"] is None
            if name.startswith("trace.") or name in {"output", "pending"}:
                assert all(grad[f"input.0.{n}.{t}"] is None for n in range(2) for t in (2, 3))
    assert actual.stats.get("semantic_full_replays", 0) > 0
    if implementation not in {"python-step", "native-step"}:
        assert actual.stats["semantic_state_replays"] == len(actual.trace)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_mixed_frozen_lanes_preserve_requires_grad(dtype, implementation):
    g = Graph((Node(0),), (), (Region(1),), (0,), (0,))
    m = Model(g, width=4, dtype=dtype).requires_grad_(False)
    q = Continuation(g.identity, 2)
    xs = [External(b, 0, t, t + 1, torch.full((4,), 0.2, dtype=dtype, requires_grad=b == 1))
          for b in range(2) for t in range(2)]
    expected = run(g, m, q, xs, 4, sealed_until=4)
    actual = execute(implementation, g, m, q, xs, "hard")
    equivalent(expected, actual)
    for (_, _, _, a), (_, _, _, b) in zip(expected.outputs, actual.outputs):
        assert a.requires_grad == b.requires_grad
    for a, b in zip(expected.trace, actual.trace):
        for name in ("proposal", "descriptor", "control", "comparison", "next", "full"):
            assert a[name].requires_grad == b[name].requires_grad, name


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_inference_has_no_semantic_replay(dtype, implementation, context):
    g, m, q, xs, _ = fixture(dtype, "attention")
    with context():
        expected = run(g, m, q, xs, 4, sealed_until=4)
        actual = execute(implementation, g, m, q, xs, "hard")
    equivalent(expected, actual)
    assert actual.stats.get("semantic_state_replays", 0) == 0
    assert actual.stats.get("semantic_full_replays", 0) == 0
    assert all(not x.requires_grad for _, _, _, x in actual.outputs)
