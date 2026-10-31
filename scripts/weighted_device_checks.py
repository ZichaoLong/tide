"""Complete device observations, independent roots, training and exact decode."""
from dataclasses import replace
import hashlib
import torch
from tidegraph import Continuation
from tidegraph.compare import objective
from tidegraph.checkpoint import save, load
from tidegraph.reference import run as reference
from tidegraph.qualification_checks import compare_finite, assert_placement, vjp, coverage, plain
from tidegraph.placement import ExecutionPlacement, place_model
from weighted_device_cases import fixture, inputs_to, leaves
from weighted_device_precision import Precision


def runner(binding, implementation, placement, native_library=None, streaming=False, *, workers=1, packing="auto"):
    from tidegraph.weighted.execution import eager_runner
    return eager_runner(binding, implementation, placement, native_library, streaming=streaming,
                        workers=workers, packing="auto" if streaming and implementation == "python" else packing)


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


def check_case(case, device, implementation, placement, out, native_library=None, training=True, peers=None,
               precision=Precision(), workers=1, packing="auto"):
    compare = precision.compare
    a, xa, stop = fixture(case, precision.dtype)
    b, _, _ = fixture(case, precision.dtype)
    b.model.load_state_dict(a.model.state_dict())
    devices = peers or (device,)
    xb, owners = locate(b, xa, devices)
    compare_finite(a.model.state_dict(), b.model.state_dict(), "starting-parameters", atol=0, rtol=0)
    torch.save(dict(weights=a.model.state_dict(), inputs=plain(xa)), out/"fixture.pt")
    q = Continuation(a.graph.identity, 2)
    ref_view = place_model(a.graph, a.model, ExecutionPlacement(scoring_dtype=placement.scoring_dtype))
    ref = lambda state, xs, end, **kw: reference(a.graph, ref_view, state, xs, end, **kw)
    execute = runner(b, implementation, placement, native_library, workers=workers, packing=packing)
    expected = ref(q, xa, stop, sealed_until=stop)
    actual = execute(q, xb, stop, sealed_until=stop)
    check_owners(b, actual, placement)
    compare(expected, actual)
    left, right = leaves(a, xa), leaves(b, xb)
    roots = [("all", objective(expected), objective(actual))]
    if expected.outputs:
        roots.append(("first-output", expected.outputs[0][-1].sum(), actual.outputs[0][-1].sum()))
    for key in expected.trace[0]["descriptor"]:
        roots.append(("first-descriptor-"+key, expected.trace[0]["descriptor"][key].sum(), actual.trace[0]["descriptor"][key].sum()))
    roots.append(("first-control", expected.trace[0]["control"].sum(), actual.trace[0]["control"].sum()))
    for name, x, y in roots:
        compare(vjp(x, left), vjp(y, right), "vjp."+name)
    with torch.no_grad():
        inference = execute(q, xb, stop, sealed_until=stop)
        scalar = runner(b, implementation, placement, native_library, streaming=True,
                        workers=workers, packing=packing)(q, xb, stop, sealed_until=stop)
    compare(expected, inference, "inference")
    compare(expected, scalar, "streaming")
    cut = 3
    first = execute(q, [x for x in xb if x.time < cut], cut, sealed_until=cut)
    tail = [x for x in xb if x.time >= cut]
    rest = execute(first.continuation, tail, stop, sealed_until=stop)
    compare(actual.continuation, rest.continuation, "retained-state")
    joined = replace(rest, outputs=first.outputs+rest.outputs)
    compare(vjp(objective(actual), right), vjp(objective(joined), right), "retained-vjp")
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
    compare(wa, wb, "raw-state-roots.records")
    compare(vjp(objective(wa), warm_roots[0]), vjp(objective(wb), warm_roots[1]), "raw-state-roots.all")
    for key in wa.trace[0]["descriptor"]:
        compare(vjp(wa.trace[0]["descriptor"][key].sum(), warm_roots[0]),
                       vjp(wb.trace[0]["descriptor"][key].sum(), warm_roots[1]), "raw-state-roots."+key)
    report = dict(coverage=coverage(actual, b.graph), roots=[name for name, _, _ in roots],
                  fixture_sha256=hashlib.sha256((out/"fixture.pt").read_bytes()).hexdigest(),
                  pending_at_cut=len(first.continuation.pending), retained_cut=cut,
                  streaming=True, prefill=True, inference=True, training_updates=0)
    report["independent_raw_state_roots"] = len(warm_roots[0])
    report["node_owners"] = [str(d) for d in owners]
    report["transport_stats"] = {k: v for k, v in actual.stats.items() if "cross_device" in k}
    report["preparation_stats"] = {k: v for k, v in actual.stats.items()
                                   if "preparation" in k or "projection" in k}
    if len(devices) > 1 and not actual.stats.get("cross_device_rows", 0):
        raise AssertionError("peer check performed no cross-device transport")
    if not training:
        return report
    oa = precision.make_optimizer(a.model)
    ob = precision.make_optimizer(b.model)
    for step in range(3):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        ra = ref(q, xa, stop, sealed_until=stop)
        rb = execute(q, xb, stop, sealed_until=stop)
        compare(ra, rb, f"train.{step}")
        precision.backward(oa, objective(ra)); precision.backward(ob, objective(rb))
        compare(precision.gradients(a.model), precision.gradients(b.model), f"train-gradient.{step}")
        oa.step(); ob.step()
        compare(a.model.state_dict(), b.model.state_dict(), f"updated.{step}")
        compare(oa.state_dict(), ob.state_dict(), f"optimizer.{step}")
    first = execute(q, [x for x in xb if x.time < cut], cut, sealed_until=cut)
    path = out/"checkpoint.pt"
    save(path, b.graph, b.model, first.continuation, ob)
    # Restore the actual computed raw record, on the target and on CPU.
    targets = [("same-device", devices), ("cpu-handoff", (torch.device("cpu"),))]
    if len(devices) > 1:
        targets.append(("remapped-peers", tuple(reversed(devices))))
    for label, target in targets:
        c, xc, _ = fixture(case, precision.dtype)
        xc, _ = locate(c, xc, target)
        oc = precision.make_optimizer(c.model)
        restored = load(path, c.graph, c.model, oc)
        compare_finite(first.continuation.detach(), restored, label+".raw", atol=0, rtol=0)
        check_owners(c, replace(first, continuation=restored, trace=[], outputs=[], messages=[]),
                     placement if target[0].type != "cpu" else ExecutionPlacement())
        compare_finite(ob.state_dict(), oc.state_dict(), label+".optimizer", atol=0, rtol=0)
        # Independent owner restored again to avoid applying the fourth step twice.
        d, xd, _ = fixture(case, precision.dtype)
        xd, _ = locate(d, xd, devices)
        od = precision.make_optimizer(d.model)
        ds = load(path, d.graph, d.model, od)
        cp = placement if target[0].type != "cpu" else ExecutionPlacement(scoring_dtype=placement.scoring_dtype)
        resumed = []
        for owner, xs, opt, state, location in ((c, xc, oc, restored, cp), (d, xd, od, ds, placement)):
            opt.zero_grad(set_to_none=True)
            r = runner(owner, implementation, location, native_library, workers=workers, packing=packing)(
                state, [x for x in xs if x.time >= cut], stop, sealed_until=stop)
            precision.backward(opt, objective(r)); opt.step()
            resumed.append((r, precision.gradients(owner.model), owner.model.state_dict(), opt.state_dict()))
        if label == "same-device":
            compare_finite(*resumed, label+".fourth-complete", atol=0, rtol=0)
        else:
            compare(*resumed, label+".fourth-complete")
    report.update(training_updates=4, same_device_resume=True, same_device_resume_exact=True,
                  cpu_checkpoint_handoff=True)
    return report
