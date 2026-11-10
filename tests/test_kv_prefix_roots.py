"""Raw sorted-time slices must retain every intermediate state as a VJP root."""
import pytest
import torch
from tidegraph import Continuation, External
from tidegraph.compare import equivalent
from tidegraph.reference import run as reference
from tidegraph.greedy import run
from tidegraph.native import Native
from test_weighted_packed import binding
from isolated_cases import vjp


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("clock", [0, 2**55+41])
def test_expiring_prefixes_preserve_intermediate_field_derivatives(dtype, native, clock):
    b = binding(dtype, "kv")
    q = Continuation(b.graph.identity, 1, cut=clock)
    xs = [External(0, 0, i, clock+t, torch.tensor([mass, .3, -.2], dtype=dtype, requires_grad=True))
          for i, (t, mass) in enumerate(((0, .4), (2, .5), (6, 0.), (7, .6)))]
    expected = reference(b.graph, b.model, q, xs, clock+8, sealed_until=clock+8)
    execute = Native(b.graph, b.model, algorithm="window", packed=True,
                     full_autograd="batched", aggregate_autograd="batched").run if native else lambda *a, **k: run(
                         b.graph, b.model, *a, component_window=True, full_autograd="batched", aggregate_autograd="batched", **k)
    actual = execute(q, xs, clock+8, sealed_until=clock+8)
    equivalent(expected, actual)
    def roots(result):
        return {f"{i}.{name}": tensor for i, event in enumerate(result.trace[:2])
                for name, tensor in event["proposal_slots"].items() if tensor.is_floating_point()}
    for index in (2, 3):
        for zero in (False, True):
            a = expected.trace[index]["proposal_slots"]["keys"]
            c = actual.trace[index]["proposal_slots"]["keys"]
            equivalent(vjp(a*0 if zero else a, roots(expected)), vjp(c*0 if zero else c, roots(actual)))
    equivalent(vjp(expected.outputs[-1][-1], roots(expected)), vjp(actual.outputs[-1][-1], roots(actual)))
    assert actual.continuation.states[0, 0].slots["times"].tolist() == [clock+7]
