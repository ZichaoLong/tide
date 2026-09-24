"""Independent Python packed VJPs retain public-root connectivity and values."""
from dataclasses import replace
import pytest
import torch
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.streaming import run as packed_stream
from tidegraph.reference import run as scalar
from tidegraph.ops import Model
from isolated_cases import fixture as full_fixture, roots, vjp
from aggregate_cases import fixture as agg_fixture


@pytest.mark.parametrize("schedule", [frontier, packed_stream])
@pytest.mark.parametrize("mode", ["hard", "softp", "hst"])
@pytest.mark.parametrize("kind", ["sum", "mean", "weighted_mean", "active_softmax", "all_softmax"])
def test_source_roots(dtype, schedule, mode, kind):
    def execute(batched):
        g, m, q, xs, variables = agg_fixture(dtype, kind, budget=1)
        result = (schedule(g, m, q, xs, 5, sealed_until=5, mode=mode,
                           full_autograd="batched", aggregate_autograd="batched") if batched
                  else scalar(g, m, q, xs, 5, sealed_until=5, mode=mode))
        first = result.trace[0]
        rs = [objective(result), result.outputs[0][-1], first["content"],
              *first["contributions"].values(), result.continuation.pending[0].value]
        if batched:
            assert result.stats["batched_aggregate_events"] == result.stats["candidate_events"]
            assert not result.stats.get("semantic_aggregate_replays", 0)
            assert not result.stats.get("semantic_full_replays", 0)
        return result, [vjp(r, variables, zero) for r in rs for zero in (False, True)]
    equivalent(execute(False), execute(True))


@pytest.mark.parametrize("mode", ["hard", "softp", "hst"])
@pytest.mark.parametrize("full", ["tanh", "swiglu", "lh-silu-rms-v1"])
@pytest.mark.parametrize("emission", ["broadcast", "slot_affine"])
def test_full_roots(dtype, mode, full, emission):
    def execute(batched):
        g, _, q, xs, variables = full_fixture(dtype, "ssm")
        g = replace(g, nodes=tuple(replace(n, full=full, emission=emission) for n in g.nodes))
        m = Model(g, width=4, dtype=dtype)
        m.nodes[1].weight = m.nodes[0].weight
        q.identity = g.identity
        variables = dict(m.named_parameters()) | {k: v for k, v in variables.items()
                    if k.startswith(("initial.", "input.", "upstream."))}
        result = (frontier(g, m, q, xs, 4, sealed_until=4, mode=mode,
                           full_autograd="batched", aggregate_autograd="batched") if batched
                  else scalar(g, m, q, xs, 4, sealed_until=4, mode=mode))
        rs = roots(result)
        first = next(e for e in result.trace if e["batch"] == 0 and e["active"])
        rs["full"] = first["full"]
        rs.update({f"emit.{k}": v for k, v in first["emitted"].items()})
        return result, {(n, zero): vjp(r, variables, zero) for n, r in rs.items() for zero in (False, True)}
    equivalent(execute(False), execute(True))


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_nograd_numeric_path_unchanged(dtype, context):
    g, m, q, xs, _ = agg_fixture(dtype, "all_softmax")
    with context():
        a = frontier(g, m, q, xs, 5, sealed_until=5)
        b = frontier(g, m, q, xs, 5, sealed_until=5, full_autograd="batched", aggregate_autograd="batched")
    equivalent(a, b)
    assert not b.stats.get("batched_aggregate_events", 0)
    assert not b.stats.get("batched_full_events", 0)


def test_isolated_linear_disconnected_and_frozen_rows(dtype):
    from tidegraph.isolated_linear import linear
    rows = [torch.ones(3, dtype=dtype, requires_grad=True) for _ in range(2)]
    weight = torch.eye(3, dtype=dtype, requires_grad=True)
    outputs = linear(rows, weight)
    gradients = torch.autograd.grad(outputs[0].sum()*0, [*rows, weight], allow_unused=True)
    assert gradients[1] is None
    assert gradients[0] is not None and torch.count_nonzero(gradients[0]) == 0
    with pytest.raises(ValueError, match="first-order"):
        torch.autograd.grad(linear(rows, weight)[0].sum(), rows, create_graph=True)
    result = linear([r.detach() for r in rows], weight.detach())
    assert all(not r.requires_grad for r in result)
