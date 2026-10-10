"""Different live output slots in one batch must retain independent dependencies."""
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.reference import run
from isolated_cases import fixture, vjp


@pytest.mark.parametrize("kind", ["ema", "ssm", "attention", "linear", "delta", "delta-rule-v1"])
@pytest.mark.parametrize("implementation", ["python", "native-streaming", "native-frontier"])
def test_mixed_slot_cotangents_and_frozen_rows(dtype, kind, implementation):
    g, m, q, xs, leaves = fixture(dtype, kind)
    expected = run(g, m, q, xs, 4, sealed_until=4, mode="hst")
    if implementation == "python":
        actual = frontier(g, m, q, xs, 4, sealed_until=4, mode="hst")
    else:
        actual = Native(g, m, algorithm=implementation.split("-")[1], packed=True, mode="hst").run(
            q, xs, 4, sealed_until=4)
    equivalent(expected, actual)
    def root(result):
        a, b = [result.continuation.states[i, 0] for i in range(2)]
        if a.slots:
            names = sorted(a.slots)
            return a.slots[names[0]].sum()*.3 + b.value.sum()*.2 + b.slots[names[-1]].sum()*0
        return a.value.sum()*.3 + b.value.sum()*0
    equivalent(vjp(root(expected), leaves), vjp(root(actual), leaves))
    assert actual.stats["batched_state_events"] > 0
    assert actual.stats.get("semantic_state_replays", 0) == 0


def test_state_vjp_rejects_higher_order():
    g, m, q, xs, leaves = fixture(torch.float64, "ema")
    result = frontier(g, m, q, xs, 4, sealed_until=4, mode="hard")
    with pytest.raises((ValueError, RuntimeError), match="first-order"):
        torch.autograd.grad(result.continuation.states[0, 0].value.sum(),
                            list(leaves.values()), create_graph=True, allow_unused=True)
