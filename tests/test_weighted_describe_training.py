"""General default Describe: finite public training/continuation compositions.

Independent scalar scheduling anchors full records and gradients. Same-source
auto execution separately checks exact checkpoint recovery and the next update.
CPU FP64 is the first gate; TIDE_DESCRIBE_DEVICE and --dtype select other scopes.
"""
import os

import pytest
import torch

from tidegraph import Continuation, ExecutionPlacement, External
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.reference import run as reference
from tidegraph.runtime import resolve_device
from tidegraph.weighted import KVConfig, RecurrentConfig, RegionalConfig, bind
from tidegraph.weighted.execution import eager_runner
from test_weighted_describe_capability import linear_vjp
from test_weighted_execution import setup, variables


DEVICE, _ = resolve_device(os.environ.get("TIDE_DESCRIBE_DEVICE", "cpu"))


def fixture(dtype, case, device):
    original, _, inputs, stop = setup(dtype, "fc")
    common = dict(width=2, summary_width=2, ffn_width=3, epsilon=.2)
    if case == "recurrent-grouped":
        memory = RecurrentConfig(memory_family="ssd", memory_read="state", key_width=2,
                                 value_width=2, conv_size=2, **common)
        regions = tuple(RegionalConfig(k=len(r.members), gamma_max=1.3, local_score="match",
                         region_score="att", region_key="content", region_evidence="grouped",
                         selection_width=2, hidden_width=3, region_control_width=3)
                        for r in original.graph.region_layouts)
    else:
        memory = KVConfig(memory_read="qry", adapter_rank=1, window=7, **common)
        regions = RegionalConfig(selection_mode="direct_attention", send_gain="context", gamma_max=1.5,
                                 region_score="att", region_key="content_stats", selection_width=2, hidden_width=3)
    b = bind(original.graph, memories=memory, regions=regions, alpha_agg=.2, dtype=dtype)
    if case == "recurrent-grouped":
        for weights in b.model.nodes:
            head = weights.read_program
            head.key = head.evidence.encoders["mem"] = head.score.encoders["mem"]
    b.model.to(device)
    xs = [External(i, x.port, x.position, x.time,
                   (x.value.detach()+x.value.new_tensor([0., .04*i, -.02*i])).to(device).requires_grad_())
          for i in range(2) for x in inputs]
    return b, xs, stop


def runner(binding, native):
    dtype = str(binding.model.nodes[0].bias.dtype).split(".")[-1]
    return eager_runner(binding, "native" if native else "python", ExecutionPlacement(
        preset="cpu" if DEVICE.type == "cpu" else "mixed-c", scoring_dtype=dtype),
        query_execution="auto", packing="packed")


def optimizer(binding):
    return torch.optim.SGD([p for p in binding.model.parameters() if p.requires_grad], lr=.001, momentum=.1)


@pytest.mark.parametrize("case", ["recurrent-grouped", "kv-context-gain"])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_default_general_describe_training_and_exact_next_window(dtype, case, native, tmp_path):
    a, xa, stop = fixture(dtype, case, "cpu")
    b, xb, _ = fixture(dtype, case, DEVICE)
    oa, ob, execute = optimizer(a), optimizer(b), runner(b, native)
    def check(left, right):
        equivalent(left, right, check_device=False)
    for step in range(2):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        for item in (*xa, *xb):
            item.value.grad = None
        expected = reference(a.graph, a.model, Continuation(a.graph.identity, 2), xa, stop, sealed_until=stop)
        actual = execute(Continuation(b.graph.identity, 2), xb, stop, sealed_until=stop)
        check(actual, expected)
        assert actual.stats["describe_batched_nodes"] == len(b.model.nodes)
        assert actual.stats["describe_scalar_nodes"] == 0
        assert actual.stats["query_"+("legacy" if case == "recurrent-grouped" else "packed-v2")+"_nodes"] == len(b.model.nodes)
        if step == 0:
            for name in expected.trace[0]["descriptor"]:
                check(linear_vjp(actual.trace[0]["descriptor"][name], variables(b, xb)),
                      linear_vjp(expected.trace[0]["descriptor"][name], variables(a, xa)))
        objective(expected).backward(); objective(actual).backward()
        check({name: p.grad for name, p in variables(b, xb).items()},
              {name: p.grad for name, p in variables(a, xa).items()})
        oa.step(); ob.step()
        check(b.model.state_dict(), a.model.state_dict())
        check(ob.state_dict(), oa.state_dict())
    cut = 2
    head = execute(Continuation(b.graph.identity, 2), [x for x in xb if x.time < cut], cut, sealed_until=cut)
    assert head.continuation.pending
    tail = [x for x in xb if x.time >= cut]
    continued = execute(head.continuation, tail, stop, sealed_until=stop)
    oracle = reference(b.graph, b.model, head.continuation, tail, stop, sealed_until=stop)
    check(continued, oracle)
    check(linear_vjp(objective(continued), variables(b, xb)), linear_vjp(objective(oracle), variables(b, xb)))
    save(tmp_path/"columns.pt", b.graph, b.model, head.continuation, ob)
    c, _, _ = fixture(dtype, case, DEVICE)
    oc = optimizer(c)
    restored = load(tmp_path/"columns.pt", c.graph, c.model, oc)
    equivalent(restored, head.continuation.detach(), atol=0, rtol=0)
    if case == "recurrent-grouped":
        for weights in c.model.nodes:
            assert weights.read_program.key is weights.read_program.evidence.encoders["mem"] is weights.read_program.score.encoders["mem"]
    ob.zero_grad(set_to_none=True); oc.zero_grad(set_to_none=True)
    left = execute(head.continuation.detach(), tail, stop, sealed_until=stop)
    right = runner(c, native)(restored, tail, stop, sealed_until=stop)
    equivalent(right, left, atol=0, rtol=0)
    equivalent(linear_vjp(objective(right), variables(c, tail)),
               linear_vjp(objective(left), variables(b, tail)), atol=0, rtol=0)
    objective(left).backward(); objective(right).backward()
    equivalent([p.grad for p in c.model.parameters()], [p.grad for p in b.model.parameters()], atol=0, rtol=0)
    ob.step(); oc.step()
    equivalent(c.model.state_dict(), b.model.state_dict(), atol=0, rtol=0)
    equivalent(oc.state_dict(), ob.state_dict(), atol=0, rtol=0)
