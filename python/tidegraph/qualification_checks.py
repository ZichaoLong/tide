"""Complete observable and independent root checks for one finite workload."""
from dataclasses import fields, is_dataclass
import torch
from .blocks import canonicalize
from .compare import equivalent, objective
from .records import Result


def plain(value):
    """Weights-only-loadable, detached snapshot; preserve None and owner keys."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value):
        return {f.name:plain(getattr(value, f.name)) for f in fields(value) if f.name != "stats"}
    if isinstance(value, dict):
        return {key:plain(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)):
        return type(value)(plain(x) for x in value)
    return value


def finite(value, path="root"):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise AssertionError(f"{path}: nonfinite qualification tensor")
    elif is_dataclass(value):
        for field in fields(value):
            if field.name != "stats":
                finite(getattr(value, field.name), path + "." + field.name)
    elif isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{path}[{key}]")
    elif isinstance(value, (tuple,list)):
        for i, item in enumerate(value):
            finite(item, f"{path}[{i}]")


def compare_finite(a, b, path="root", **tolerances):
    finite(a, path)
    finite(b, path)
    equivalent(a, b, path, **tolerances)


def vjp(root, leaves):
    gradients = (torch.autograd.grad(root, tuple(leaves.values()), allow_unused=True, retain_graph=True)
                 if root.requires_grad else [None] * len(leaves))
    return dict(zip(leaves, gradients))


def probe_loss(result, root="all"):
    """Mean-scaled diagnostic loss so its magnitude does not grow with graph size."""
    tensors = []
    if root in {"all", "output"}:
        tensors += [x for _,_,_,x in result.outputs]
    if root in {"all", "state"}:
        tensors += [s.value for s in result.continuation.states.values()]
        tensors += [v for s in result.continuation.states.values() for v in s.slots.values()]
    if root in {"all", "history"}:
        tensors += [v for h in result.continuation.history.values() for v in h.tensors.values()]
    if root in {"all", "pending"}:
        tensors += [a.value for a in result.continuation.pending]
    return objective(result, root) / max(1, sum(v.numel() for v in tensors))


def roots(result):
    yield "all", probe_loss(result)
    for name in ("output", "state", "history", "pending"):
        try:
            yield name, probe_loss(result, name)
        except ValueError:
            continue
    if result.trace:
        yield "first-content", result.trace[0]["content"].square().mean()
    if result.outputs:
        yield "first-output", result.outputs[0][-1].square().mean()
    if result.continuation.pending:
        yield "first-pending", result.continuation.pending[0].value.square().mean()


def compare_gradients(a, ar, ap, b, br, bp, *, compare=equivalent):
    left = {k:v for k,v in ar.model.named_parameters() if v.requires_grad} | ap.leaves()
    right = {k:v for k,v in br.model.named_parameters() if v.requires_grad} | bp.leaves()
    ra, rb = dict(roots(a)), dict(roots(b))
    if ra.keys() != rb.keys():
        raise AssertionError("gradient root categories differ")
    for key in ra:
        compare(vjp(ra[key], left), vjp(rb[key], right), path=f"gradient.{key}")
    all_gradients = vjp(ra["all"], left)
    return dict(roots=list(ra), leaves=len(left), disconnected=sum(v is None for v in all_gradients.values()))


def chunked(runtime, probe):
    session = runtime.session(probe.batch_size)
    middle = probe.stop // 2
    first = probe.advance(session, 0, middle)
    last = probe.advance(session, middle)
    return canonicalize(runtime.graph, Result(last.continuation, first.trace + last.trace,
                        first.outputs + last.outputs, first.messages + last.messages, {}))


def coverage(result, graph):
    touched = {event["node"] for event in result.trace}
    return dict(nodes=len(graph.nodes), edges=len(graph.edges), regions=len(graph.regions),
                observed_nodes=len(touched), unobserved_nodes=len(graph.nodes)-len(touched),
                events=len(result.trace), outputs=len(result.outputs), pending=len(result.continuation.pending),
                state_owners=len(result.continuation.states), history_owners=len(result.continuation.history))
