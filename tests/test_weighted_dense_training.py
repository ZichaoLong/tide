"""Independent graph schedules, all roots, AdamW and retained/exact cut state."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, ExecutionPlacement
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.reference import run as reference
from tidegraph.weighted import KVConfig, RegionalConfig, bind
from tidegraph.weighted.execution import eager_runner
from test_weighted_execution import setup, variables
from test_weighted_dense_query import DEVICE
from dense_checks import Checks
from isolated_cases import vjp


def fixture(dtype, family, device):
    original, _, inputs, stop = setup(dtype, family)
    b = bind(original.graph, memories=KVConfig(width=2, summary_width=2, adapter_rank=1, heads=1,
             ffn_width=3, window=7, full_frequencies=((.2,),), memory_read="qry"),
             regions=tuple(RegionalConfig(k=len(r.members), gamma_max=1.3) for r in original.graph.region_layouts),
             dtype=dtype, alpha_agg=.2)
    # One explicit shared query Parameter, many independent snapshots.
    for node in b.model.nodes:
        node.full_program.content.query = node.kernel.read_q
    b.model.to(device)
    xs = [External(i, x.port, x.position, x.time,
                   (x.value.detach()+x.value.new_tensor([0., .04*i, -.02*i])).to(device).requires_grad_())
          for i in range(2) for x in inputs]
    return b, xs, stop


def runner(b, native):
    return eager_runner(b, "native" if native else "python",
        ExecutionPlacement(preset="cpu" if DEVICE.type == "cpu" else "mixed-c",
                           scoring_dtype=str(b.model.nodes[0].bias.dtype).split(".")[-1]),
        query_execution="dense-v1", packing="packed")


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("family", ["fc", "hb", "pdg"])
def test_dense_complete_training_and_cuts(dtype, native, family, tmp_path, request):
    check = Checks(request.node.nodeid)
    a, xa, stop = fixture(dtype, family, "cpu")
    b, xb, _ = fixture(dtype, family, DEVICE)
    execute = runner(b, native)
    oa = torch.optim.AdamW([p for p in a.model.parameters() if p.requires_grad], lr=.001)
    ob = torch.optim.AdamW([p for p in b.model.parameters() if p.requires_grad], lr=.001)
    for step in range(3):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        ra = reference(a.graph, a.model, Continuation(a.graph.identity, 2), xa, stop, sealed_until=stop)
        rb = execute(Continuation(b.graph.identity, 2), xb, stop, sealed_until=stop)
        check(rb, ra, f"step{step}.complete")
        if step == 0:
            for name in ra.trace[0]["descriptor"]:
                check(vjp(rb.trace[0]["descriptor"][name], variables(b, xb)),
                      vjp(ra.trace[0]["descriptor"][name], variables(a, xa)), f"root.{name}")
        objective(ra).backward(); objective(rb).backward()
        check([p.grad for p in b.model.parameters()], [p.grad for p in a.model.parameters()], f"step{step}.gradients")
        oa.step(); ob.step()
        check(b.model.state_dict(), a.model.state_dict(), f"step{step}.parameters")
        check(ob.state_dict(), oa.state_dict(), f"step{step}.adamw")
    cut = 2
    head = execute(Continuation(b.graph.identity, 2), [x for x in xb if x.time < cut], cut, sealed_until=cut)
    assert head.continuation.pending
    tail = [x for x in xb if x.time >= cut]
    retained = execute(head.continuation, tail, stop, sealed_until=stop)
    oracle = reference(b.graph, b.model, head.continuation, tail, stop, sealed_until=stop)
    check(retained, oracle, "retained.complete")
    check(vjp(objective(retained), variables(b, xb)), vjp(objective(oracle), variables(b, xb)), "retained.roots")
    save(tmp_path/"cut.pt", b.graph, b.model, head.continuation, ob)
    c, _, _ = fixture(dtype, family, DEVICE)
    oc = torch.optim.AdamW([p for p in c.model.parameters() if p.requires_grad], lr=.001)
    restored = load(tmp_path/"cut.pt", c.graph, c.model, oc)
    equivalent(restored, head.continuation.detach(), atol=0, rtol=0)
    ob.zero_grad(set_to_none=True); oc.zero_grad(set_to_none=True)
    left = execute(head.continuation.detach(), tail, stop, sealed_until=stop)
    right = runner(c, native)(restored, tail, stop, sealed_until=stop)
    equivalent(right, left, atol=0, rtol=0)
    objective(left).backward(); objective(right).backward()
    equivalent([p.grad for p in c.model.parameters()], [p.grad for p in b.model.parameters()], atol=0, rtol=0)
    ob.step(); oc.step()
    equivalent(c.model.state_dict(), b.model.state_dict(), atol=0, rtol=0)
    equivalent(oc.state_dict(), ob.state_dict(), atol=0, rtol=0)
    check.finish()
