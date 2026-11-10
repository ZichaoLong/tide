"""Five memory families: ragged B×L, independent schedule, updates and raw resume.

Local formula anchors live in test_weighted_{kv,recurrent}_formulas; this gate
checks that the public default composes them without changing causal records.
"""
import os
import pytest
import torch
from tidegraph import Continuation, External, ExecutionPlacement
from tidegraph.runtime import resolve_device
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.reference import run as reference
from tidegraph.weighted.execution import eager_runner
from test_weighted_execution import setup, variables
from test_weighted_recurrent_execution import model
from isolated_cases import vjp
from dense_checks import Checks


DEVICE, _ = resolve_device(os.environ.get("TIDE_WINDOW_DEVICE", "cpu"))


def fixture(dtype, family, topology, device):
    binding, spec, old, stop = setup(dtype, topology) if family == "kv" else model(dtype, family, topology=topology)
    stride = spec.stride if spec else 1
    origin = 1 if spec else 0
    xs = []
    for batch in range(2):
        for port in range(len(binding.graph.inputs)):
            # Independent sample lengths and real logical-time gaps.
            for position, t in enumerate((0, 1, 3, 5)[:4-batch]):
                mass = 0. if batch == 1 and position == 2 else .35+.1*port
                value = torch.tensor([mass, .2+.02*t, -.15+.03*batch], dtype=dtype, device=device)
                xs.append(External(batch, port, position, origin+t*stride, value.requires_grad_()))
    binding.model.to(device)
    return binding, xs, 7*stride


def runner(binding, native):
    return eager_runner(binding, "native" if native else "python",
                        ExecutionPlacement(preset="cpu" if DEVICE.type == "cpu" else "mixed-c",
                                           scoring_dtype=str(binding.model.nodes[0].bias.dtype).split(".")[-1]))


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("topology", ["hb", "dag", "pdg"])
@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd"])
def test_default_ragged_memory_update_and_resume(dtype, native, topology, family, tmp_path, request):
    # Fixed 47704c9: same first evidence root in four CPU affine witnesses.
    # Same FP32 material promoted to FP64 passes; all other full-graph checks,
    # updates and exact restore passed. Keep the original strict failure named.
    witnessed = (DEVICE.type == "cpu" and dtype == torch.float32
                 and family == "affine" and topology in {"dag", "pdg"})
    check = Checks(request.node.nodeid,
        known_numeric={"root.evidence[nodes.0.read_program.evidence_weight]": 1.6e-6} if witnessed else None,
        failure_reason=("affine first evidence FP32 strict witness; diagnose_window_describe.py"
                        if witnessed else "component-window strict numeric difference retained; cause pending bounded analysis"))
    a, xa, stop = fixture(dtype, family, topology, "cpu")
    b, xb, _ = fixture(dtype, family, topology, DEVICE)
    execute = runner(b, native)
    oa, ob = [torch.optim.SGD(x.model.parameters(), lr=.0001, momentum=.3) for x in (a, b)]
    for step in range(2):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        expected = reference(a.graph, a.model, Continuation(a.graph.identity, 2), xa, stop, sealed_until=stop)
        actual = execute(Continuation(b.graph.identity, 2), xb, stop, sealed_until=stop)
        check(actual, expected, f"step{step}.complete")
        assert actual.stats["window_components"] > 0
        if topology == "hb":
            assert actual.stats["max_state_batch"] == 2
            assert actual.stats["max_state_sequence"] >= 3
        if step == 0:
            for name in ("score", "evidence"):
                check(vjp(actual.trace[0]["descriptor"][name], variables(b, xb)),
                      vjp(expected.trace[0]["descriptor"][name], variables(a, xa)), f"root.{name}")
        objective(expected).backward(); objective(actual).backward()
        check([p.grad for p in b.model.parameters()], [p.grad for p in a.model.parameters()], f"step{step}.gradients")
        oa.step(); ob.step()
        check(b.model.state_dict(), a.model.state_dict(), f"step{step}.parameters")
        check(ob.state_dict(), oa.state_dict(), f"step{step}.optimizer")
    cut = stop//2
    head = execute(Continuation(b.graph.identity, 2), [x for x in xb if x.time < cut], cut, sealed_until=cut)
    save(tmp_path/"window.pt", b.graph, b.model, head.continuation, ob)
    c, _, _ = fixture(dtype, family, topology, DEVICE)
    oc = torch.optim.SGD(c.model.parameters(), lr=.0001, momentum=.3)
    restored = load(tmp_path/"window.pt", c.graph, c.model, oc)
    equivalent(head.continuation.detach(), restored, atol=0, rtol=0)
    tail = [x for x in xb if x.time >= cut]
    ob.zero_grad(set_to_none=True); oc.zero_grad(set_to_none=True)
    left = execute(head.continuation.detach(), tail, stop, sealed_until=stop)
    right = runner(c, native)(restored, tail, stop, sealed_until=stop)
    equivalent(left, right, atol=0, rtol=0)
    objective(left).backward(); objective(right).backward()
    equivalent([p.grad for p in b.model.parameters()], [p.grad for p in c.model.parameters()], atol=0, rtol=0)
    ob.step(); oc.step()
    equivalent(b.model.state_dict(), c.model.state_dict(), atol=0, rtol=0)
    equivalent(ob.state_dict(), oc.state_dict(), atol=0, rtol=0)
    check.finish()
