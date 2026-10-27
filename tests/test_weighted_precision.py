"""Explicit half payload, FP32 scoring and masters; no implicit solver promotion."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.compare import equivalent, objective
from tidegraph.checkpoint import save, load
from tidegraph.placement import ExecutionPlacement, place_model
from tidegraph.precision import FP32MasterOptimizer
from tidegraph.reference import run as reference
from tidegraph.greedy import run as greedy
from tidegraph.native import Native
from tidegraph.weighted import bind
from test_weighted_recurrent_execution import model
from test_weighted_execution import setup, variables
from tidegraph.qualification_checks import vjp


class GDNAdamWParityError(AssertionError):
    """Only the retained parameter-update witness, not arbitrary test errors."""


def half_close(a, b):
    # FP32 descriptors and master slots inherit quantized payload error. The
    # declared half policy applies to all floating observables; discrete records
    # and None gradients are still exact under compare.equivalent.
    equivalent(a, b, atol=1e-3, rtol=2e-2)


def fixture(family):
    original, _, xs, stop = setup(torch.float32, "hb") if family == "kv" else model(torch.float32, family, topology="hb")
    b = bind(original.graph, memories=original.memories, regions=original.regions, alpha_agg=.2, dtype=torch.float16)
    f = bind(original.graph, memories=original.memories, regions=original.regions, alpha_agg=.2, dtype=torch.float32)
    for name, p in b.model.state_dict().items():
        torch.testing.assert_close(p, f.model.state_dict()[name].half(), rtol=0, atol=0)
    xs = [replace(x, value=x.value.detach().half().requires_grad_()) for x in xs]
    return b, xs, stop


def run(b, native):
    p = ExecutionPlacement(scoring_dtype="float32")
    options = dict(packed=True, aggregate_autograd="batched", full_autograd="batched")
    if native:
        return Native(b.graph, b.model, algorithm="greedy", placement=p, **options).run
    view = place_model(b.graph, b.model, p)
    return lambda *a, **kw: greedy(b.graph, view, *a, **options, **kw)


@pytest.mark.parametrize("family", ["kv", "affine", "delta", "ssd",
    pytest.param("gated_delta", marks=pytest.mark.xfail(strict=True, raises=GDNAdamWParityError, reason=
        "Recorded strict half GDN AdamW witness: near-zero gradients yield different updates; "
        "docs/weighted-precision.md, core5-half-clocks-dev. Not a passing training case."))])
@pytest.mark.parametrize("native", [False, True])
def test_half_payload_explicit_scoring_master_updates_and_resume(family, native, tmp_path):
    check_half_training(family, native, tmp_path, optimizer="adamw")


@pytest.mark.parametrize("native", [False, True])
def test_half_gdn_explicit_sgd_master_updates_and_resume(native, tmp_path):
    check_half_training("gated_delta", native, tmp_path, optimizer="sgd")


def check_half_training(family, native, tmp_path, *, optimizer):
    a, xa, stop = fixture(family)
    b, xb, _ = fixture(family)
    view = place_model(a.graph, a.model, ExecutionPlacement(scoring_dtype="float32"))
    q = Continuation(a.graph.identity, 1)
    ref = lambda state, xs, end: reference(a.graph, view, state, xs, end, sealed_until=end)
    execute = run(b, native)
    oa = FP32MasterOptimizer((p for p in a.model.parameters() if p.requires_grad), optimizer=optimizer, lr=.001, loss_scale=16)
    ob = FP32MasterOptimizer((p for p in b.model.parameters() if p.requires_grad), optimizer=optimizer, lr=.001, loss_scale=16)
    for step in range(3):
        oa.zero_grad(); ob.zero_grad()
        ra, rb = ref(q, xa, stop), execute(q, xb, stop, sealed_until=stop)
        half_close(ra, rb)
        assert all(e["control"].dtype == torch.float16 for e in rb.trace)
        assert all(v.dtype == torch.float32 for e in rb.trace for v in e["descriptor"].values())
        if step == 0:
            half_close(vjp(objective(ra), variables(a, xa)), vjp(objective(rb), variables(b, xb)))
            half_close(vjp(ra.outputs[0][-1].float().sum(), variables(a, xa)),
                       vjp(rb.outputs[0][-1].float().sum(), variables(b, xb)))
        oa.backward(objective(ra)); ob.backward(objective(rb))
        oa.step(); ob.step()
        try:
            equivalent(a.model.state_dict(), b.model.state_dict())
        except AssertionError as error:
            if (family == "gated_delta" and optimizer == "adamw"
                    and "root[nodes.5.kernel.conv.weight]" in str(error)):
                raise GDNAdamWParityError(str(error)) from error
            raise
        # Master gradients aggregate half-rounding errors; compare to the same
        # declared mixed-precision baseline, not to a fictitious FP32 forward.
        equivalent(oa.state_dict(), ob.state_dict(), atol=1e-3, rtol=2e-2)
    cut = 3
    first = execute(q, [x for x in xb if x.time < cut], cut, sealed_until=cut)
    path = tmp_path/"half.pt"; save(path, b.graph, b.model, first.continuation, ob)
    c, _, _ = fixture(family)
    oc = FP32MasterOptimizer((p for p in c.model.parameters() if p.requires_grad), optimizer=optimizer, lr=.001, loss_scale=1)
    restored = load(path, c.graph, c.model, oc)
    equivalent(first.continuation.detach(), restored); equivalent(ob.state_dict(), oc.state_dict())
    assert oc.loss_scale == 16
    for owner, state, opt, fn in ((b, first.continuation.detach(), ob, execute), (c, restored, oc, run(c, native))):
        opt.zero_grad(); r = fn(state, [x for x in xb if x.time >= cut], stop, sealed_until=stop)
        opt.backward(objective(r)); opt.step()
    equivalent(b.model.state_dict(), c.model.state_dict()); equivalent(ob.state_dict(), oc.state_dict())


def test_half_scoring_requires_explicit_supported_precision():
    b, xs, stop = fixture("kv")
    with pytest.raises(ValueError, match="FP32/FP64"):
        reference(b.graph, b.model, Continuation(b.graph.identity, 1), xs, stop, sealed_until=stop)
