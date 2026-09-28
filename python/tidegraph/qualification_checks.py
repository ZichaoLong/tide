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


def vjp(root, leaves):
    gradients = (torch.autograd.grad(root, tuple(leaves.values()), allow_unused=True, retain_graph=True)
                 if root.requires_grad else [None] * len(leaves))
    return dict(zip(leaves, gradients))


def roots(result):
    yield "all", objective(result)
    for name in ("output", "state", "history", "pending"):
        try:
            yield name, objective(result, name)
        except ValueError:
            continue
    if result.trace:
        yield "first-content", result.trace[0]["content"].square().sum()
    if result.outputs:
        yield "first-output", result.outputs[0][-1].square().sum()
    if result.continuation.pending:
        yield "first-pending", result.continuation.pending[0].value.square().sum()


def compare_gradients(a, ar, ap, b, br, bp):
    left = {k:v for k,v in ar.model.named_parameters() if v.requires_grad} | ap.leaves()
    right = {k:v for k,v in br.model.named_parameters() if v.requires_grad} | bp.leaves()
    ra, rb = dict(roots(a)), dict(roots(b))
    if ra.keys() != rb.keys():
        raise AssertionError("gradient root categories differ")
    for key in ra:
        equivalent(vjp(ra[key], left), vjp(rb[key], right), path=f"gradient.{key}")
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
