"""Finite installed base checks; scope is explicit, never a backend matrix."""
from copy import deepcopy
from pathlib import Path
import torch
from .config import resolve
from .records import now, write_json


def _selected(resolved, **changes):
    request = deepcopy(resolved["requested"])
    # The resolved material is self-contained; a caller may have changed cwd
    # since resolving a relative model.path. Never reopen an ambient path.
    request["model"] = dict(definition={key: deepcopy(resolved["plan"][key]) for key in
        ("graph", "memories", "regions", "classes", "source_binding", "source_hidden_width", "alpha_agg")})
    request["task"] = deepcopy(resolved["task"])
    request["site"] = deepcopy(resolved["site"])
    request.setdefault("execution", {}).update(return_mode="complete")
    request.setdefault("measurement", {}).update(validation="strict", **changes)
    request.setdefault("resources", {}).pop("logical_windows", None)
    return resolve(request)


def _leaves(owner):
    result = {"parameter."+n: p for n, p in owner.parameter_owner.named_parameters() if p.requires_grad}
    for x in owner.last_inputs:
        result[f"input.{x.batch}.{x.port}.{x.position}.{x.time}"] = x.value
    return result


def _vjp(loss, leaves):
    from .returns import snapshot
    gradients = torch.autograd.grad(loss, tuple(leaves.values()), allow_unused=True, retain_graph=True)
    return snapshot(dict(zip(leaves, gradients)))


def eager_roots(resolved, *, full, progress=None):
    from .eager import Eager
    from ..compare import objective, equivalent
    from ..planning.precision import tolerances
    dtype = getattr(torch, resolved["execution"]["effective"]["dtype"])
    atol, rtol = tolerances(dtype)
    check = lambda a, b: equivalent(a, b, check_device=False, atol=atol, rtol=rtol)
    checked = []
    def report(stage, root=None):
        if progress is not None:
            progress(dict(stage=stage, root=root, completed_roots=list(checked)))
    def compare_vjp(name, x, y, a, b):
        report("reference-vjp", name)
        expected = _vjp(x, a)
        report("candidate-vjp", name)
        actual = _vjp(y, b)
        report("compare-vjp", name)
        check(expected, actual)
        checked.append(name)
        report("root-passed", name)
    report("construction")
    left, right = Eager(resolved, reference=True), Eager(resolved)
    try:
        report("reference-forward")
        expected = left.forward(0)
        report("candidate-forward")
        actual = right.forward(0)
        report("compare-forward")
        check(expected, actual)
        a, b = _leaves(left), _leaves(right)
        report("construct-roots")
        roots = [("all", objective(expected), objective(actual))]
        if getattr(left.task, "module", None) is not None:
            roots.append(("downstream-task-loss", left.task.loss(expected, training=True), right.task.loss(actual, training=True)))
        if expected.outputs:
            roots += [("first-output", expected.outputs[0][-1].sum(), actual.outputs[0][-1].sum()),
                      ("connected-zero", expected.outputs[0][-1].sum()*0, actual.outputs[0][-1].sum()*0)]
        if expected.trace:
            roots += [("first-control", expected.trace[0]["control"].sum(), actual.trace[0]["control"].sum())]
            for key in expected.trace[0]["descriptor"]:
                roots.append(("first-descriptor-"+key, expected.trace[0]["descriptor"][key].sum(), actual.trace[0]["descriptor"][key].sum()))
        if full and expected.messages:
            roots.append(("first-physical-message", expected.messages[0].value.sum(), actual.messages[0].value.sum()))
        for name, x, y in roots:
            compare_vjp(name, x, y, a, b)
        if full:
            from .. import Continuation
            cut = max(1, min(left.span-1, resolved["plan"]["stride"]-1))
            q = Continuation(left.binding.graph.identity, resolved["plan"]["samples"])
            states, raw = [], []
            for owner in (left, right):
                report("reference-private-state" if owner is left else "candidate-private-state")
                placed = owner.place_inputs(owner.last_inputs)
                head = [x for x in placed if x.time < cut]
                first = owner.execute(q, head, cut, sealed_until=cut)
                state = first.continuation.detach()
                leaves = {}
                for key, s in state.states.items():
                    s.value = s.value.detach().requires_grad_()
                    leaves[f"{key}.value"] = s.value
                    for name, value in tuple(s.slots.items()):
                        if value.is_floating_point():
                            s.slots[name] = value.detach().requires_grad_()
                            leaves[f"{key}.{name}"] = s.slots[name]
                tail = [x for x in placed if x.time >= cut]
                states.append(owner.execute(state, tail, owner.span, sealed_until=owner.span))
                raw.append(leaves)
            report("compare-private-state")
            check(*states)
            if raw[0]:
                compare_vjp("private-raw-state-roots-after-nonposition-cut",
                            objective(states[0]), objective(states[1]), raw[0], raw[1])
            else:
                checked.append("private-raw-state-roots-empty-at-selected-cut")
        report("complete")
    finally:
        left.close()
        right.close()
    return dict(state="passed", roots=checked, parameters_and_external_roots=True,
                scope="actual selected Python/native eager schedule; independent scalar CPU schedule")


def resident_roots(resolved):
    """Output and connected-zero roots; no claim of all raw-field VJPs."""
    from .eager import Eager
    from .resident import Resident
    from ..compare import equivalent
    from ..planning.precision import tolerances
    dtype = getattr(torch, resolved["execution"]["effective"]["dtype"])
    atol, rtol = tolerances(dtype)
    checked = []
    for scale in (.07, 0.):
        reference, candidate = Eager(resolved, reference=True), Resident(resolved)
        try:
            expected = reference.forward(0)
            loss = sum(x.float().sum()*scale for *_, x in expected.outputs)
            leaves = {n: p for n, p in reference.binding.model.named_parameters() if p.requires_grad}
            want = _vjp(loss, leaves)
            with torch.no_grad():
                xs = candidate.task.inputs(candidate.inputs, 0, device="cpu", training=False)
                window = candidate.owner.advance(xs, stop=candidate.span, sealed_until=candidate.span)
                f = window.fields["outputs"]
                bar = torch.where(f.present[..., None], torch.full_like(f.values, scale, dtype=torch.float32), torch.zeros_like(f.values, dtype=torch.float32))
                gradients = candidate.owner.backward([candidate.owner.cotangents(window, {"outputs": bar})])
                got = candidate.layout.decode(gradients.parameters, candidate.owner.checkpoint())["gradients"]
                got = {name: got[name] for name in leaves}
                equivalent(got, want, check_device=False, atol=atol, rtol=rtol)
            checked.append("output" if scale else "connected-zero")
        finally:
            candidate.close()
            reference.close()
    return dict(state="passed", roots=checked, scope="independent public resident parameter VJP; full raw-field matrix remains separately qualified")


def check(configuration, output, scope):
    if scope not in {"smoke", "related", "full"}:
        raise ValueError("scope: choose smoke, related or full")
    original = resolve(configuration)
    if original["inventory"]["totals"]["trainable"] > 5_000_000:
        raise ValueError("base checks require an explicit model <=5M parameters; supply a small representative, use run for scale validation")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema="tide-base-check-v1", state="running", started=now(), scope=scope,
        config_id=original["config_id"], checks=[],
        coverage="selected finite configuration only; not every module/topology/device/dtype or complete repository qualification",
        untested=["all-configuration backend matrix", "higher-order gradients", "multi-card resident without hardware capability"])
    write_json(out/"result.json", record)
    def phase(name):
        record["phase"] = name
        write_json(out/"result.json", record)
    def root_progress(detail):
        record["root_progress"] = detail
        write_json(out/"result.json", record)
    try:
        from .engine import run, _configure
        inference = _selected(original, mode="inference", steps=1 if scope == "smoke" else 2, warmup=0, checkpoint_check=True)
        phase("inference")
        result = run(inference, out/"inference")
        record["checks"].append(dict(name="complete-independent-inference-and-exact-resume", state=result["state"], result="inference/result.json"))
        if result["state"] != "passed":
            raise AssertionError("selected inference strict qualification failed")
        if scope != "smoke":
            training = _selected(original, mode="training", steps=2 if scope == "related" else 3, warmup=0, checkpoint_check=True)
            e = training["execution"]["effective"]
            _configure(training)
            if e["client"] != "libtorch":
                phase("independent-roots")
                roots = resident_roots(training) if e["placement"] == "resident" else eager_roots(training, full=scope == "full", progress=root_progress)
                record["checks"].append(dict(name="independent-roots", **roots))
            cpp_roots = e["client"] == "libtorch" and e["placement"] != "resident"
            if e["client"] == "libtorch" and not cpp_roots:
                record["untested"].append("standalone resident independent roots use the separate installed device component gates")
            phase("training")
            result = run(training, out/"training", check_roots=cpp_roots)
            if cpp_roots:
                roots = result.get("independent_roots", {})
                record["checks"].append(dict(name="independent-roots", **roots))
                if roots.get("state") != "passed":
                    raise AssertionError("standalone independent-root diagnostic missing or failed")
            record["checks"].append(dict(name="complete-independent-training-and-exact-resume", state=result["state"], result="training/result.json"))
            if result["state"] != "passed":
                raise AssertionError("selected training strict qualification failed")
        record["state"] = "passed"
        phase("complete")
    except BaseException as error:
        record.update(state="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        record["finished"] = now()
        write_json(out/"result.json", record)
    return record
