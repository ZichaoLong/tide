"""Complete tagged fibers, local source parameters and optional per-source content."""
from dataclasses import dataclass
from collections import defaultdict
import torch
from . import autograd
from .records import Atom
from .origins import view


@dataclass(frozen=True)
class SourceInput:
    slot: int
    atom: Atom
    scale: torch.Tensor


@dataclass(frozen=True)
class AggregateInput:
    time: int
    slots: int
    sources: tuple[SourceInput, ...]  # Canonical program-visible atom order, no padding.


@dataclass
class AggregateResult:
    value: torch.Tensor
    contributions: dict[int, torch.Tensor]


class AggregateProgram(torch.nn.Module):
    profile = "custom"  # Version this name in Node.aggregation.
    joint_batch = False

    def step(self, weights, request):
        raise NotImplementedError

    def batch(self, weights, requests):
        return [self.step(weights, r) for r in requests]


class SourceAggregate(AggregateProgram):
    joint_batch = True

    def __init__(self, kind="sum"):
        super().__init__()
        if kind not in {"sum", "mean", "weighted_mean", "active_softmax", "all_softmax"}:
            raise ValueError("unknown Aggregate profile")
        self.kind = kind

    def coefficients(self, w, request):
        slots = [s.slot for s in request.sources]
        if self.kind in {"sum", "mean"}:
            return None
        if self.kind == "weighted_mean":
            masses = torch.nn.functional.softplus(torch.stack([w.extra[f"agg_mass_{s}"] for s in slots]))
            denominator = masses.sum()
            if not bool(denominator.detach() > 0):
                raise ValueError("Aggregate weighted mean has zero mass")
            return masses / denominator
        domain = range(request.slots) if self.kind == "all_softmax" else slots
        probabilities = torch.stack([w.extra[f"agg_logit_{s}"] for s in domain]).softmax(0)
        return probabilities[slots] if self.kind == "all_softmax" else probabilities

    def combine(self, w, request, values):
        coefficients = self.coefficients(w, request)
        terms = [value if self.kind == "sum" else value / len(values) if self.kind == "mean"
                 else value * coefficients[i] for i, value in enumerate(values)]
        h = terms[0]
        for value in terms[1:]:
            h = h + value
        return h, terms

    def step(self, w, request):
        h, terms = self.combine(w, request, [s.atom.value * s.scale for s in request.sources])
        return AggregateResult(h, dict(sorted((s.slot, t) for s, t in zip(request.sources, terms))))

    def batch(self, w, requests):
        groups = defaultdict(list)
        for i, request in enumerate(requests):
            groups[tuple(s.slot for s in request.sources)].append(i)
        result = [None] * len(requests)
        for rows in groups.values():
            request = requests[rows[0]]
            values = [torch.stack([requests[i].sources[j].atom.value for i in rows]) *
                      torch.stack([requests[i].sources[j].scale for i in rows]).unsqueeze(-1)
                      for j in range(len(request.sources))]
            h, terms = self.combine(w, request, values)
            for row, i in enumerate(rows):
                result[i] = AggregateResult(h[row], dict(sorted((s.slot, t[row]) for s, t in zip(request.sources, terms))))
        return result


def validate_program(weights, spec, slots):
    program = weights.aggregate_program
    if not isinstance(program, SourceAggregate):
        if spec.identity or program.profile != spec.aggregation:
            raise ValueError("custom Aggregate program does not match graph profile")
        return
    if program.kind != spec.aggregation:
        raise ValueError("shared Aggregate program does not match graph profile")
    prefix = "agg_mass_" if program.kind == "weighted_mean" else "agg_logit_" if "softmax" in program.kind else None
    if prefix:
        names = {name for name in weights.extra if name.startswith(prefix)}
        if names != {f"{prefix}{i}" for i in range(slots)} or any(weights.extra[name].shape != () for name in names):
            raise ValueError("Aggregate parameter slot domain mismatch")


def request(graph, model, fiber):
    if not fiber:
        raise ValueError("Aggregate requires a nonempty fiber")
    node, time = fiber[0].node, fiber[0].time
    offsets = graph.port_indexes[0].offsets
    sources = [SourceInput(graph.ports.input[a.source] if a.kind == 0 else graph.ports.edge_target[a.source],
                           view(graph, a), model.input_scale[a.source] if a.kind == 0 else model.agg_scale[a.source]) for a in fiber]
    if graph.origins:
        sources.sort(key=lambda s: s.atom.key())
    return AggregateInput(time, offsets[node+1] - offsets[node], tuple(sources))


def validate(result, request):
    if not isinstance(result, AggregateResult):
        raise ValueError("Aggregate returned an invalid result")
    keys = list(result.contributions)
    present = {s.slot for s in request.sources}
    if keys != sorted(set(keys)) or any(type(k) is not int or k not in present for k in keys):
        raise ValueError("Aggregate returned invalid contribution slots")
    ref = request.sources[0].atom.value
    for value in [result.value, *result.contributions.values()]:
        if not isinstance(value, torch.Tensor) or (value.shape, value.dtype, value.device) != (ref.shape, ref.dtype, ref.device):
            raise ValueError("Aggregate returned incompatible tensor metadata")


def evaluate(graph, model, events, packed=False):
    if not events:
        return
    w = model.nodes[events[0]["node"]]
    requests = [request(graph, model, e["fiber"]) for e in events]
    program = w.aggregate_program
    if packed:
        with torch.no_grad():
            results = program.batch(w, requests)
        if len(results) != len(events):
            raise ValueError("Aggregate batch changed event count")
        for i, r in enumerate(requests):
            validate(results[i], r)
            if torch.is_grad_enabled():
                ref = program.step(w, r); validate(ref, r)
                result = results[i]
                if result.contributions.keys() != ref.contributions.keys():
                    raise ValueError("Aggregate batch changed contribution presence")
                value = autograd.value(result.value, ref.value)
                terms = {slot: autograd.value(t, value if ref.contributions[slot] is ref.value else ref.contributions[slot])
                         for slot, t in result.contributions.items()}
                results[i] = AggregateResult(value, terms)
    else:
        results = [program.step(w, r) for r in requests]
    for event, result, r in zip(events, results, requests):
        validate(result, r)
        event.update(content=result.value, contributions=result.contributions)
