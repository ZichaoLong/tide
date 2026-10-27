"""Architecture §3: complete Describe fields from actual int64 snapshots.

Scalar local formulas anchor values; fixtures distinguish initial/old/proposal,
expired KV entries and sparse absolute convolution slots without scalar replay.
"""
from dataclasses import replace
import pytest
import torch
from tidegraph import Graph, Node, Region
from tidegraph.compare import equivalent
from tidegraph.content import Content
from tidegraph.readout import ReadInput
from tidegraph.weighted import KVConfig, RecurrentConfig, RegionalConfig, bind


CASES = [("kv", "sum"), ("kv", "qry")]+[(f, r) for f in ("affine", "delta", "gated_delta", "ssd") for r in ("sum", "state")]


def fixture(dtype, family, read, grouped):
    if family == "kv":
        c = KVConfig(width=2, summary_width=2, memory_read=read, window=5,
                     read_frequencies=(.3,) if read == "qry" else ())
    else:
        c = RecurrentConfig(memory_family=family, width=2, summary_width=2, memory_read=read,
                            heads=2 if family in {"gated_delta", "ssd"} else 1,
                            key_width=2, value_width=3, conv_size=3 if family in {"gated_delta", "ssd"} else 1,
                            conv_stride=2 if family in {"gated_delta", "ssd"} else 1)
    rc = RegionalConfig(local_score="match", region_score="att", region_key="content", region_evidence="grouped",
                        selection_width=2, region_control_width=3) if grouped else RegionalConfig()
    b = bind(Graph((Node(0),), (), (Region(1),), (0,), (0,)), memories=c, regions=rc, dtype=dtype)
    w = b.model.nodes[0]; m = w.kernel; clock = 2**54+17
    requests = []
    for i, stamps in enumerate(((), (0,), (0, 2), (0, 2), (0,), (0, 2, 3))):
        state = m.initial(w)
        for j in stamps:
            state = m.step(w, state, Content(torch.tensor([.2+.1*i, .3+.1*j, -.4+.1*i], dtype=dtype)), clock+j)
        content = Content(torch.tensor([.4+.1*i, .2, -.3+.1*i], dtype=dtype))
        time = clock+(20 if i == 4 else 4)
        requests.append(ReadInput(state, time, content))
    # Read^0 binds initial even when neighboring queries have populated memory.
    requests.append(replace(requests[1], state=None))
    return w, requests


@pytest.mark.parametrize("family,read", CASES)
@pytest.mark.parametrize("grouped", [False, True])
def test_describe_all_memories_and_field_layout(dtype, family, read, grouped, monkeypatch):
    w, requests = fixture(dtype, family, read, grouped)
    p = w.read_program
    with torch.no_grad():
        expected = [p.step(w, r) for r in requests]
    def forbidden(*args, **kwargs):
        raise AssertionError("Read batch called a scalar memory/Describe function")
    monkeypatch.setattr(p, "step", forbidden)
    monkeypatch.setattr(w.kernel, "mem", forbidden)
    with torch.no_grad():
        actual = p.batch(w, requests)
    equivalent(expected, actual)
    assert all({k: tuple(t.shape) for k, t in row.items()} == p.field_layout for row in actual)
    assert p.batch(w, []) == [] and p.joint_batch


@pytest.mark.parametrize("family,read", CASES)
def test_full_comparison_batch_broadcast_and_unscaled_payload(dtype, family, read, monkeypatch):
    from tidegraph.full import FullInput
    w, reads = fixture(dtype, family, read, False)
    requests = [FullInput(r.state or w.initial(), r.time, r.content,
                          r.content.value.new_tensor(.1+.1*i)) for i, r in enumerate(reads)]
    p = w.full_program
    with torch.no_grad():
        expected = [p.step(w, r, 3, "hard", 1.) for r in requests]
    def forbidden(*args, **kwargs):
        raise AssertionError("Full batch called scalar Full/content/features")
    monkeypatch.setattr(p, "step", forbidden)
    monkeypatch.setattr(p.content, "forward", forbidden)
    if hasattr(w.kernel, "features"):
        monkeypatch.setattr(w.kernel, "features", forbidden)
    with torch.no_grad():
        actual = p.batch(w, requests, 3, "hard", 1.)
    equivalent(expected, actual)
    for r, out in zip(requests, actual):
        assert out.value[0] == r.control
        assert all(value is out.value for value in out.emitted.values())
    assert p.batch(w, [], 3, "hard", 1.) == [] and p.joint_batch
    with pytest.raises(ValueError, match="legacy Emit"):
        p.batch(w, requests, 3, "hst", 1.)


@pytest.mark.parametrize("case", [0, 3, 4])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_joint_queries_complete_records_and_autograd_boundary(dtype, case, native, monkeypatch):
    from tidegraph import Continuation, External
    from tidegraph.compare import objective
    from tidegraph.greedy import run as greedy
    from tidegraph.native import Native
    from tidegraph.reference import run as reference
    from test_weighted_scoring_execution import configured
    from test_weighted_execution import variables
    from isolated_cases import vjp
    b, original, stop = configured(dtype, case, "pdg", read_mode=("0", "-", "+")[case % 3])
    xs = [External(i, x.port, x.position, x.time, (x.value.detach()+x.value.new_tensor([0., .1*i, -.03*i])).requires_grad_())
          for i in range(3) for x in original]
    q = Continuation(b.graph.identity, 3)
    expected = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
    runner = Native(b.graph, b.model, algorithm="greedy", packed=True).run if native else lambda *a, **kw: greedy(b.graph, b.model, *a, **kw)
    actual = runner(q, xs, stop, sealed_until=stop)
    equivalent(expected, actual)
    roots = variables(b, xs)
    equivalent(vjp(objective(expected), roots), vjp(objective(actual), roots))
    for name in expected.trace[0]["descriptor"]:
        equivalent(vjp(expected.trace[0]["descriptor"][name], roots), vjp(actual.trace[0]["descriptor"][name], roots))
    assert actual.stats.get("read_scalar_batch_steps", 0) == actual.stats.get("full_scalar_fallback_steps", 0) == 0
    assert actual.stats["batched_read_events"] > 0 and actual.stats.get("semantic_read_replays", 0) == 0
    assert actual.stats["semantic_full_replays"] > 0  # Default policy remains explicit replay.
    if not native:
        def forbidden(*a, **kw):
            raise AssertionError("numeric query batch called scalar Read or Full")
        for node in b.model.nodes:
            monkeypatch.setattr(node.read_program, "step", forbidden)
            monkeypatch.setattr(node.full_program, "step", forbidden)
    with torch.no_grad():
        equivalent(expected, runner(q, xs, stop, sealed_until=stop))
