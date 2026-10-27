"""Complete device observations, independent roots, training and exact decode."""
from dataclasses import replace
import torch
from tidegraph import Continuation
from tidegraph.compare import objective
from tidegraph.checkpoint import save, load
from tidegraph.reference import run as reference
from tidegraph.qualification_checks import compare_finite, assert_placement, vjp, coverage, plain
from tidegraph.placement import ExecutionPlacement, place_model
from weighted_device_cases import fixture, inputs_to, leaves


def runner(binding, implementation, placement, native_library=None, streaming=False):
    options = dict(packed=True, aggregate_autograd="batched", full_autograd="batched")
    if implementation == "native":
        from tidegraph.native import Native
        from tidegraph.native_loader import load_native
        load_native(native_library, backend=binding.model.nodes[0].bias.device.type)
        return Native(binding.graph, binding.model,
                      algorithm="streaming" if streaming else "greedy", placement=placement, **options).run
    from tidegraph.greedy import run as greedy
    from tidegraph.streaming import run as stream
    view = place_model(binding.graph, binding.model, placement)
    schedule = stream if streaming else greedy
    if streaming:
        options = {}
    return lambda *a, **kw: schedule(binding.graph, view, *a, **options, **kw)


def check_live(result, placement, device):
    assert_placement(result.continuation, device)
    assert_placement(result.outputs, device)
    assert_placement(result.messages, device)
    target = placement.resolve(device)
    for event in result.trace:
        assert_placement(event["descriptor"], target["read"])
        assert_placement(event["control"], device)


def locate(binding, xs, devices):
    """Regional sharing requires co-location; private regions can have peers."""
    from tidegraph.ownership import place_payloads
    owners = tuple(devices[n.region % len(devices)] for n in binding.graph.nodes)
    place_payloads(binding.graph, binding.model, owners)
    copies = {}
    result = []
    for x in xs:
        target = owners[binding.graph.inputs[x.port]]
        key = (id(x.value), target)
        if key not in copies:
            copies[key] = x.value.detach().to(target).clone().requires_grad_()
        result.append(replace(x, value=copies[key]))
    return result, owners


def check_owners(binding, result, placement):
    from tidegraph.ownership import region_reference
    owner = lambda node: binding.model.nodes[node].bias.device
    for event in result.trace:
        device = owner(event["node"])
        for name in ("content", "control", "proposal", "comparison", "next"):
            assert_placement(event[name], device)
        assert_placement(event["descriptor"], placement.resolve(device)["read"])
    for (_, node), state in result.continuation.states.items():
        assert_placement(state, owner(node))
    for (_, region), history in result.continuation.history.items():
        assert_placement(history, region_reference(binding.graph, binding.model, region).device)
    for atom in result.messages+result.continuation.pending:
        assert_placement(atom.value, owner(atom.node))
    for _, _, port, value in result.outputs:
        assert_placement(value, owner(binding.graph.outputs[port]))


def check_case(case, device, implementation, placement, out, native_library=None, training=True, peers=None):
    a, xa, stop = fixture(case)
    b, _, _ = fixture(case)
    b.model.load_state_dict(a.model.state_dict())
    devices = peers or (device,)
    xb, owners = locate(b, xa, devices)
    compare_finite(a.model.state_dict(), b.model.state_dict(), "starting-parameters")
    torch.save(dict(weights=a.model.state_dict(), inputs=plain(xa)), out/"fixture.pt")
    q = Continuation(a.graph.identity, 2)
    ref_view = place_model(a.graph, a.model, ExecutionPlacement(scoring_dtype=placement.scoring_dtype))
    ref = lambda state, xs, end, **kw: reference(a.graph, ref_view, state, xs, end, **kw)
    execute = runner(b, implementation, placement, native_library)
    expected = ref(q, xa, stop, sealed_until=stop)
    actual = execute(q, xb, stop, sealed_until=stop)
    check_owners(b, actual, placement)
    compare_finite(expected, actual)
    left, right = leaves(a, xa), leaves(b, xb)
    roots = [("all", objective(expected), objective(actual))]
    if expected.outputs:
        roots.append(("first-output", expected.outputs[0][-1].sum(), actual.outputs[0][-1].sum()))
    for key in expected.trace[0]["descriptor"]:
        roots.append(("first-descriptor-"+key, expected.trace[0]["descriptor"][key].sum(), actual.trace[0]["descriptor"][key].sum()))
    roots.append(("first-control", expected.trace[0]["control"].sum(), actual.trace[0]["control"].sum()))
    for name, x, y in roots:
        compare_finite(vjp(x, left), vjp(y, right), "vjp."+name)
    with torch.no_grad():
        inference = execute(q, xb, stop, sealed_until=stop)
        scalar = runner(b, implementation, placement, native_library, streaming=True)(q, xb, stop, sealed_until=stop)
    compare_finite(expected, inference, "inference")
    compare_finite(expected, scalar, "streaming")
    cut = 3
    first = execute(q, [x for x in xb if x.time < cut], cut, sealed_until=cut)
    tail = [x for x in xb if x.time >= cut]
    rest = execute(first.continuation, tail, stop, sealed_until=stop)
    compare_finite(actual.continuation, rest.continuation, "retained-state")
    joined = replace(rest, outputs=first.outputs+rest.outputs)
    compare_finite(vjp(objective(actual), right), vjp(objective(joined), right), "retained-vjp")
    # Independent raw float-state roots; metadata remains exact, non-differentiable.
    head_a = ref(q, [x for x in xa if x.time < cut], cut, sealed_until=cut)
    warm_a, warm_b = head_a.continuation.detach(), first.continuation.detach()
    warm_roots = []
    for warm in (warm_a, warm_b):
        roots_here = {}
        for owner, state in warm.states.items():
            state.value = state.value.detach().requires_grad_()
            roots_here[f"{owner}.value"] = state.value
            for name, value in tuple(state.slots.items()):
                if value.is_floating_point():
                    state.slots[name] = value.detach().requires_grad_()
                    roots_here[f"{owner}.{name}"] = state.slots[name]
        warm_roots.append(roots_here)
    wa = ref(warm_a, [x for x in xa if x.time >= cut], stop, sealed_until=stop)
    wb = execute(warm_b, tail, stop, sealed_until=stop)
    compare_finite(wa, wb, "raw-state-roots.records")
    compare_finite(vjp(objective(wa), warm_roots[0]), vjp(objective(wb), warm_roots[1]), "raw-state-roots.all")
    for key in wa.trace[0]["descriptor"]:
        compare_finite(vjp(wa.trace[0]["descriptor"][key].sum(), warm_roots[0]),
                       vjp(wb.trace[0]["descriptor"][key].sum(), warm_roots[1]), "raw-state-roots."+key)
    report = dict(coverage=coverage(actual, b.graph), roots=[name for name, _, _ in roots],
                  pending_at_cut=len(first.continuation.pending), retained_cut=cut,
                  streaming=True, prefill=True, inference=True, training_updates=0)
    report["independent_raw_state_roots"] = len(warm_roots[0])
    report["node_owners"] = [str(d) for d in owners]
    report["transport_stats"] = {k: v for k, v in actual.stats.items() if "cross_device" in k}
    if len(devices) > 1 and not actual.stats.get("cross_device_rows", 0):
        raise AssertionError("peer check performed no cross-device transport")
    if not training:
        return report
    oa = torch.optim.AdamW(a.model.parameters(), lr=.001)
    ob = torch.optim.AdamW(b.model.parameters(), lr=.001)
    for step in range(3):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        ra = ref(q, xa, stop, sealed_until=stop)
        rb = execute(q, xb, stop, sealed_until=stop)
        compare_finite(ra, rb, f"train.{step}")
        objective(ra).backward(); objective(rb).backward()
        compare_finite({n: p.grad for n, p in a.model.named_parameters()},
                       {n: p.grad for n, p in b.model.named_parameters()}, f"train-gradient.{step}")
        oa.step(); ob.step()
        compare_finite(a.model.state_dict(), b.model.state_dict(), f"updated.{step}")
        compare_finite(oa.state_dict(), ob.state_dict(), f"optimizer.{step}")
    first = execute(q, [x for x in xb if x.time < cut], cut, sealed_until=cut)
    path = out/"checkpoint.pt"
    save(path, b.graph, b.model, first.continuation, ob)
    # Restore the actual computed raw record, on the target and on CPU.
    targets = [("same-device", devices), ("cpu-handoff", (torch.device("cpu"),))]
    if len(devices) > 1:
        targets.append(("remapped-peers", tuple(reversed(devices))))
    for label, target in targets:
        c, xc, _ = fixture(case)
        xc, _ = locate(c, xc, target)
        oc = torch.optim.AdamW(c.model.parameters(), lr=.001)
        restored = load(path, c.graph, c.model, oc)
        compare_finite(first.continuation.detach(), restored, label+".raw")
        check_owners(c, replace(first, continuation=restored, trace=[], outputs=[], messages=[]),
                     placement if target[0].type != "cpu" else ExecutionPlacement())
        compare_finite(ob.state_dict(), oc.state_dict(), label+".optimizer")
        # Independent owner restored again to avoid applying the fourth step twice.
        d, xd, _ = fixture(case)
        xd, _ = locate(d, xd, devices)
        od = torch.optim.AdamW(d.model.parameters(), lr=.001)
        ds = load(path, d.graph, d.model, od)
        cp = placement if target[0].type != "cpu" else ExecutionPlacement(scoring_dtype=placement.scoring_dtype)
        for owner, xs, opt, state, location in ((c, xc, oc, restored, cp), (d, xd, od, ds, placement)):
            opt.zero_grad(set_to_none=True)
            r = runner(owner, implementation, location, native_library)(state, [x for x in xs if x.time >= cut], stop, sealed_until=stop)
            objective(r).backward(); opt.step()
        compare_finite(c.model.state_dict(), d.model.state_dict(), label+".fourth-update")
        compare_finite(oc.state_dict(), od.state_dict(), label+".fourth-optimizer")
    report.update(training_updates=4, same_device_resume=True, cpu_checkpoint_handoff=True)
    return report
