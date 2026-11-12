"""Independent LH/HB tensor formula versus scalar scheduling and hand formulas."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from lh_tensor_baseline import StaticKV
from lh_tensor_math import _Quadratic
from lh_tensor_state import detached, FLOATS
from tidegraph import Continuation, External
from tidegraph.records import State
from tidegraph.reference import run as scalar
from tidegraph.compare import equivalent
from tidegraph.settle import SettleGraph


def parameters(anchor):
    return [p for p in anchor.binding.model.parameters() if p.requires_grad]


def fixture(dtype, topology="lh", profile="textbook", depth=1, region_width=2):
    from tidegraph.planning.recipes import prepare
    from tidegraph.planning.materialize import materialize
    plan, _ = prepare(dict(topology=topology, family="kv", width=4, depth=depth,
        region_width=region_width, sparse_denominator=2, profile=profile,
        region_overrides=dict(allocation_temperature=.7)), samples=2, positions=3, seed=7)
    bound, _, _ = materialize(plan, max_bytes=2**26, dtype=dtype, include_inputs=False)
    spec = SettleGraph(bound.graph, tuple(plan["graph"]["ranks"]), binding="ports", time_stride=plan["stride"])
    # Shared weights are distinct from the nodes' private memories.
    bound.model.nodes[1].full_program.content.down = bound.model.nodes[0].full_program.content.down
    return StaticKV(bound, spec)


def oracle(anchor, x, support, *, start=0, old=None):
    binding, spec = anchor.binding, anchor.spec
    q = Continuation(binding.graph.identity, x.shape[0], cut=start*spec.stride)
    if old is not None:
        for (b, v), state in anchor.decode(dict(states=old)).items():
            if state["observations"]:
                q.states[b, v] = State(torch.zeros_like(binding.model.nodes[v].bias),
                    state["last_time"], state["observations"], state["slots"])
    external = []
    for b in range(x.shape[0]):
        for p, v in enumerate(binding.graph.inputs):
            occurrence = 0
            for t in range(x.shape[1]):
                if support[p, b, t]:
                    external.append(External(b, p, occurrence, (start+t)*spec.stride+spec.rank(v),
                        torch.cat((x.new_ones(1), x[b, t]))))
                    occurrence += 1
    stop = (start+x.shape[1])*spec.stride
    return scalar(binding.graph, binding.model, q, external, stop, sealed_until=stop)


def output_projection(anchor, result, x, start=0):
    out = torch.zeros((len(anchor.binding.graph.outputs), *x.shape), dtype=x.dtype)
    for batch, time, port, value in result.outputs:
        t = (time-anchor.spec.rank(anchor.binding.graph.outputs[port]))//anchor.spec.stride-start
        out[port, batch, t] = value[0]*value[1:]
    return out


def state_objective(states):
    return sum(v.square().sum() for state in states.values() for v in state["slots"].values() if v.is_floating_point())*.001


@pytest.mark.parametrize("profile", ["baseline", "textbook"])
def test_lh_output_controls_raw_state_shared_gradients_and_sparse_ports(dtype, profile):
    actual, expected = fixture(dtype, profile=profile), fixture(dtype, profile=profile)
    x = torch.linspace(-.2, .4, 24, dtype=dtype).reshape(2, 3, 4).requires_grad_()
    ox = x.detach().clone().requires_grad_()
    support = torch.tensor([[[True, False, True], [True, True, False]],
                            [[True, True, False], [False, True, False]]])
    result = actual(x, support=support, records=True)
    reference = oracle(expected, ox, support)
    projected = result["output_controls"][..., None]*result["output_contents"]
    target = output_projection(expected, reference, ox)
    equivalent(projected, target)
    raw = actual.decode(result)
    for event in reference.trace:
        b, v = event["batch"], event["node"]
        r, i = actual.location[v]
        t = (event["time"]-actual.spec.rank(v))//actual.spec.stride
        equivalent(result["regions"][r]["controls"][i, b, t], event["control"])
        assert bool(result["regions"][r]["active"][i, b, t]) == event["active"]
    for key, state in reference.continuation.states.items():
        equivalent(raw[key]["slots"], state.slots)
        assert (raw[key]["last_time"], raw[key]["observations"]) == (state.last_time, state.observations)
    assert int(result["counts"][0]) == len(reference.trace)
    assert int(result["counts"][1]) == sum(e["active"] for e in reference.trace)
    loss = projected.square().sum()+state_objective(raw)
    target_loss = target.square().sum()+.001*sum(v.square().sum() for s in reference.continuation.states.values()
                                                for v in s.slots.values() if v.is_floating_point())
    ga = torch.autograd.grad(loss, [x, *parameters(actual)], allow_unused=True)
    ge = torch.autograd.grad(target_loss, [ox, *parameters(expected)], allow_unused=True)
    for a, e, root in zip(ga, ge, [x, *parameters(actual)]):
        equivalent(torch.zeros_like(root) if a is None else a, torch.zeros_like(root) if e is None else e)


def test_quad_empty_full_and_interior_implicit_gradients(dtype):
    scores = torch.tensor([[.8, .3], [9., 1.], [1., 2.]], dtype=dtype, requires_grad=True)
    caps = torch.tensor([[1., 1.], [.5, .75], [0., 0.]], dtype=dtype, requires_grad=True)
    target = torch.tensor([.7, 1.25, 0.], dtype=dtype, requires_grad=True)
    actual = _Quadratic.apply(scores, caps, target, 1.)
    equivalent(actual, torch.tensor([[.6, .1], [.5, .75], [0., 0.]], dtype=dtype))
    bar = torch.tensor([[2., 4.], [3., 5.], [8., 9.]], dtype=dtype)
    ds, dc, dt = torch.autograd.grad((actual*bar).sum(), (scores, caps, target))
    equivalent(ds, torch.tensor([[-1., 1.], [0., 0.], [0., 0.]], dtype=dtype))
    equivalent(dc, torch.tensor([[0., 0.], [3., 5.], [0., 0.]], dtype=dtype))
    equivalent(dt, torch.tensor([3., 0., 0.], dtype=dtype))


def test_independent_raw_left_roots_and_candidate_continuation(dtype):
    a, other = fixture(dtype), fixture(dtype)
    x = torch.linspace(-.2, .4, 16, dtype=dtype).reshape(2, 2, 4).requires_grad_()
    left = detached(a(x.detach())["states"])
    old = detached(left)
    for banks in (left, old):
        for bank in banks.values():
            for name in FLOATS:
                bank[name].requires_grad_()
    support = torch.ones((2, 2, 2), dtype=torch.bool); support[0, 1, 0] = False
    ox = x.detach().clone().requires_grad_()
    result = a(x, support=support, start=5, left=left)
    reference = oracle(other, ox, support, start=5, old=old)
    y = result["output_controls"][..., None]*result["output_contents"]
    target = output_projection(other, reference, ox, start=5)
    equivalent(y, target)
    raw = a.decode(result)
    for key, state in reference.continuation.states.items():
        equivalent(raw[key]["slots"], state.slots)
    roots = [x, *(bank[name] for bank in left.values() for name in FLOATS), *parameters(a)]
    oroots = [ox, *(bank[name] for bank in old.values() for name in FLOATS), *parameters(other)]
    ga = torch.autograd.grad(y.square().sum()+state_objective(raw), roots, allow_unused=True)
    state_loss = .001*sum(v.square().sum() for s in reference.continuation.states.values()
                          for v in s.slots.values() if v.is_floating_point())
    ge = torch.autograd.grad(target.square().sum()+state_loss, oroots, allow_unused=True)
    for g, h, root in zip(ga, ge, roots):
        equivalent(torch.zeros_like(root) if g is None else g, torch.zeros_like(root) if h is None else h)


def test_duplicate_edges_same_class_are_distinct_atoms(dtype):
    from tidegraph import Graph, Node, Edge, Region
    from tidegraph.source_class import SourceClasses
    from tidegraph.weighted import bind
    base = fixture(dtype)
    g, original = base.binding.graph, base.binding
    duplicate = Graph(tuple(Node(n.region) for n in g.nodes), tuple(g.edges)+(g.edges[0],),
                      tuple(Region(1) for _ in g.regions), g.inputs, g.outputs)
    classes = SourceClasses(original.source_classes.edges+(original.source_classes.edges[0],),
                            original.source_classes.inputs)
    def build():
        binding = bind(duplicate, memories=original.memories, regions=original.regions,
                       source_classes=classes, source_hidden_width=4, dtype=dtype, seed=7)
        return StaticKV(binding, SettleGraph(binding.graph, base.spec.ranks, binding="ports", time_stride=base.spec.stride))
    actual, expected = build(), build()
    assert any(float(s.incidence.max()) == 2 for s in actual.sources)
    x = torch.linspace(-.2, .4, 8, dtype=dtype).reshape(1, 2, 4)
    result = actual(x)
    reference = oracle(expected, x, torch.ones((2, 1, 2), dtype=torch.bool))
    equivalent(result["output_controls"][..., None]*result["output_contents"], output_projection(expected, reference, x))
    for key, state in reference.continuation.states.items():
        equivalent(actual.decode(result)[key]["slots"], state.slots)


def test_retained_windows_large_int64_time_and_silent_state(dtype):
    anchor = fixture(dtype)
    start = 2**53//anchor.spec.stride+2
    x = torch.linspace(-.2, .4, 24, dtype=dtype).reshape(2, 3, 4).requires_grad_()
    all_result = anchor(x, start=start)
    first = anchor(x[:, :1], start=start)
    second = anchor(x[:, 1:], start=start+1, left=first["states"])
    equivalent(torch.cat((first["hidden"], second["hidden"]), 1), all_result["hidden"])
    equivalent(anchor.decode(second), anchor.decode(all_result))
    zero = torch.zeros((len(anchor.binding.graph.inputs), 2, 1), dtype=torch.bool)
    silence = anchor(x[:, :1], start=start+10000, support=zero, left=second["states"])
    equivalent(anchor.decode(silence), anchor.decode(second), atol=0, rtol=0)
    ga = torch.autograd.grad(all_result["hidden"].square().sum(), [x, *parameters(anchor)], allow_unused=True)
    gb = torch.autograd.grad(first["hidden"].square().sum()+second["hidden"].square().sum(),
                             [x, *parameters(anchor)], allow_unused=True)
    for a, b, root in zip(ga, gb, [x, *parameters(anchor)]):
        equivalent(torch.zeros_like(root) if a is None else a, torch.zeros_like(root) if b is None else b)


def test_three_optimizer_steps_and_exact_raw_checkpoint(tmp_path, dtype):
    a, restored = fixture(dtype), fixture(dtype)
    optimizer = torch.optim.AdamW(a.binding.model.parameters(), lr=1e-4, foreach=False)
    other = torch.optim.AdamW(restored.binding.model.parameters(), lr=1e-4, foreach=False)
    x = torch.linspace(-.2, .4, 16, dtype=dtype).reshape(2, 2, 4)
    def step(anchor, opt, left, start):
        opt.zero_grad(set_to_none=True)
        result = anchor(x, left=left, start=start)
        (result["output_controls"][..., None]*result["output_contents"]).square().sum().backward()
        opt.step()
        return result["hidden"].detach(), detached(result["states"])
    _, left = step(a, optimizer, None, 0)
    path = tmp_path/"projection.pt"
    torch.save(dict(model=a.binding.model.state_dict(), optimizer=optimizer.state_dict(), left=left), path)
    saved = torch.load(path, weights_only=True)
    restored.binding.model.load_state_dict(saved["model"]); other.load_state_dict(saved["optimizer"])
    right = saved["left"]
    for start in (2, 4):
        av, left = step(a, optimizer, left, start)
        bv, right = step(restored, other, right, start)
        equivalent(av, bv, atol=0, rtol=0)
        equivalent(left, right, atol=0, rtol=0)
        equivalent(a.binding.model.state_dict(), restored.binding.model.state_dict(), atol=0, rtol=0)
        equivalent(optimizer.state_dict(), other.state_dict(), atol=0, rtol=0)


def test_domain_rejections_and_many_node_recipe(dtype):
    anchor = fixture(dtype, depth=3, region_width=8)
    assert len(anchor.binding.graph.nodes) == 176
    x = torch.ones((1, 1, 4), dtype=dtype)
    assert torch.isfinite(anchor(x)["hidden"]).all()
    with pytest.raises(ValueError, match="bool P,B,L"):
        anchor(x, support=torch.ones((8, 1, 1), dtype=dtype))
    with pytest.raises(ValueError, match="int64"):
        anchor(x, start=2**63)
