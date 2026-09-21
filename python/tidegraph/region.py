"""Functional selectors with complete candidate controls and region-owned history."""
from dataclasses import dataclass
from types import MappingProxyType
import torch
from .history import History, increment, validate as validate_history


@dataclass(frozen=True)
class RegionLayout:
    spec: object
    members: tuple[int, ...]
    slots: object

    @classmethod
    def create(cls, spec, members):
        members = tuple(members)
        return cls(spec, members, MappingProxyType({v: i for i, v in enumerate(members)}))


@dataclass(frozen=True)
class RegionInput:
    history: History
    time: int
    candidates: tuple[tuple[int, torch.Tensor], ...]
    layout: RegionLayout
    payload_dtype: torch.dtype
    payload_device: torch.device


@dataclass
class Selection:
    active: set[int]
    controls: dict[int, torch.Tensor]
    history: History


class RegionProgram(torch.nn.Module):
    profile = "custom"

    def initial(self, layout, reference):
        return History()

    def step(self, request):
        raise NotImplementedError

    def validate_history(self, history, layout):
        pass

    def validate_weights(self, layout):
        pass


class CountSelector(RegionProgram):
    profile = "count-v1"

    def initial(self, layout, reference):
        return History(node_maps={"selected": {}})

    def scores(self, request):
        return [d for _, d in request.candidates]

    def eligible(self, score):
        return True

    def step(self, r):
        nodes = [v for v, _ in r.candidates]
        scores = self.scores(r)
        if any(not torch.isfinite(d).all() for d in scores):
            raise ValueError("nonfinite region selector score")
        counts = r.history.node_maps["selected"]
        order = sorted((i for i, d in enumerate(scores) if self.eligible(d)), key=lambda i: (
            counts.get(nodes[i], 0) if r.layout.spec.count_priority else 0, -float(scores[i].detach()), nodes[i]))
        active = {nodes[i] for i in order[:r.layout.spec.budget]}
        probs = torch.stack(scores).softmax(0).to(dtype=r.payload_dtype, device=r.payload_device)
        history = r.history.fork()
        history.last_time = r.time
        for v in active:
            history.node_maps["selected"][v] = increment(counts.get(v, 0))
        return Selection(active, dict(zip(nodes, probs.unbind(0))), history)

    def validate_history(self, history, layout):
        if (history.scalars or history.node_maps.keys() != {"selected"}
                or any(c < 0 for c in history.node_maps["selected"].values())):
            raise ValueError("invalid selected-count history layout")
        if history.tensors:
            raise ValueError("unexpected count-selector history tensors")


class PositiveSelector(CountSelector):
    """Simple empty-selection anchor: only strictly positive scores are eligible."""
    profile = "positive-v1"

    def eligible(self, score):
        return float(score.detach()) > 0


class TensorHistorySelector(CountSelector):
    profile = "tensor-history-v1"

    def __init__(self, members, dtype):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(.5, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.linspace(-.2, .2, members, dtype=dtype))

    def initial(self, layout, reference):
        return History(node_maps={"selected": {}}, tensors={"memory": reference.new_zeros(())})

    def scores(self, r):
        memory = r.history.tensors["memory"]
        return [d + memory*self.bias[r.layout.slots[v]] for v, d in r.candidates]

    def step(self, r):
        result = super().step(r)
        result.history.tensors["memory"] = self.alpha*r.history.tensors["memory"] + torch.stack(
            [d for _, d in r.candidates]).sum().to(dtype=r.payload_dtype, device=r.payload_device)
        return result

    def validate_weights(self, layout):
        if self.alpha.shape != () or self.bias.shape != (len(layout.members),):
            raise ValueError("region parameter membership layout mismatch")

    def validate_history(self, history, layout):
        if (history.scalars or history.node_maps.keys() != {"selected"}
                or any(c < 0 for c in history.node_maps["selected"].values())
                or history.tensors.keys() != {"memory"} or history.tensors["memory"].shape != ()):
            raise ValueError("invalid tensor-selector history layout")


def program(layout, dtype):
    profile = layout.spec.selector
    if profile == CountSelector.profile:
        return CountSelector()
    if profile == PositiveSelector.profile:
        return PositiveSelector()
    if profile == TensorHistorySelector.profile:
        return TensorHistorySelector(len(layout.members), dtype)
    from .lh_selector import LHSelector
    if profile == LHSelector.profile:
        return LHSelector()
    raise ValueError("unknown region selector profile")


def validate_program(program, layout, reference, *, native=False):
    from .lh_selector import LHSelector
    if native and type(program) not in {CountSelector, PositiveSelector, TensorHistorySelector, LHSelector}:
        raise ValueError("Python custom region selector has no native implementation")
    if not isinstance(program, RegionProgram) or program.profile != layout.spec.selector:
        raise ValueError("region program does not match graph profile")
    for value in (*program.parameters(), *program.buffers()):
        if (value.dtype, value.device) != (reference.dtype, reference.device) or not torch.isfinite(value).all():
            raise ValueError("incompatible region parameter dtype/device/value")
    program.validate_weights(layout)


def evaluate(graph, model, q, batch, region, time, candidates):
    """Only called for a complete, canonical nonempty candidate set."""
    layout, p = graph.region_layouts[region], model.regions[region]
    reference = model.nodes[0].bias
    nodes = [v for v, _ in candidates]
    if not nodes or nodes != sorted(set(nodes)) or any(v not in layout.slots for v in nodes):
        raise ValueError("invalid region candidate domain/order")
    old = q.history.get((batch, region))
    if old is None:
        old = p.initial(layout, reference)
        validate_history(old, layout, reference, time-1)
        p.validate_history(old, layout)
    result = p.step(RegionInput(old, time, tuple(candidates), layout, reference.dtype, reference.device))
    if (not isinstance(result, Selection) or not isinstance(result.active, set)
            or any(type(v) is not int for v in result.active) or not result.active <= set(nodes)
            or len(result.active) > layout.spec.budget):
        raise ValueError("region selector returned invalid active subset/capacity")
    if (not isinstance(result.controls, dict) or result.controls.keys() != set(nodes)
            or any(type(v) is not int for v in result.controls)):
        raise ValueError("region selector returned invalid control domain")
    for control in result.controls.values():
        if not isinstance(control, torch.Tensor) or (control.dtype, control.device) != (reference.dtype, reference.device):
            raise ValueError("region selector returned incompatible control dtype/device")
        if not torch.isfinite(control).all():
            raise ValueError("region selector returned nonfinite control")
    validate_history(result.history, layout, reference, time)
    p.validate_history(result.history, layout)
    q.history[batch, region] = result.history
    return result.active, result.controls, result.history
