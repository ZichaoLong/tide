"""Functional Full programs return sparse families indexed by local output slots."""
from dataclasses import dataclass
import torch
from . import autograd
from .records import State
from .content import Content


@dataclass(frozen=True)
class FullInput:
    comparison: State
    time: int
    content: Content
    control: torch.Tensor


@dataclass
class FullResult:
    value: torch.Tensor | None
    emitted: dict[int, torch.Tensor]


class FullProgram(torch.nn.Module):
    profile = "custom"  # Declare a versioned name in Node.emission for persistence.
    joint_batch = False

    def step(self, weights, request, slots, mode, zeta):
        raise NotImplementedError

    def batch(self, weights, requests, slots, mode, zeta):
        return [self.step(weights, r, slots, mode, zeta) for r in requests]


class ProjectionEmit(FullProgram):
    joint_batch = True

    def __init__(self, spec=None):
        super().__init__()
        self.kind = "broadcast" if spec is None else spec.emission
        self.period = 1 if spec is None else spec.emit_period
        self.phases = () if spec is None else spec.emit_phases
        self.identity = False if spec is None else spec.identity
        if self.kind not in {"broadcast", "slot_affine"}:
            raise ValueError("unknown emission program")

    def present(self, slot, time):
        phase = self.phases[slot] if self.phases else -1
        return phase == -1 or (phase >= 0 and time % self.period == phase)

    def step(self, w, request, slots, mode, zeta):
        from .ops import emit
        h, p = request.content.value, request.control
        if p.ndim != 0:
            raise ValueError("projection Emit requires scalar control")
        fresh = w.fresh(request.comparison.value, h)
        value = h if self.identity else emit(h, fresh, p, mode, zeta)
        outputs = {}
        for slot in range(slots):
            if not self.present(slot, request.time):
                continue
            if self.kind == "broadcast":
                outputs[slot] = value
            else:
                weight, bias = w.extra[f"emit_w_{slot}"], w.extra[f"emit_b_{slot}"]
                outputs[slot] = emit(h @ weight, fresh @ weight + bias, p, mode, zeta)
        return FullResult(value, outputs)

    def batch(self, w, requests, slots, mode, zeta):
        from .ops import emit
        if any(r.control.ndim != 0 for r in requests):
            raise ValueError("projection Emit requires scalar control")
        h = torch.stack([r.content.value for r in requests]); p = torch.stack([r.control for r in requests])
        comparison = torch.stack([r.comparison.value for r in requests])
        fresh = w.fresh(comparison, h)
        values = h if self.identity else emit(h, fresh, p, mode, zeta)
        results = [FullResult(value, {}) for value in values]
        for slot in range(slots):
            rows = [i for i, r in enumerate(requests) if self.present(slot, r.time)]
            if not rows:
                continue
            if self.kind == "broadcast":
                for i in rows:
                    results[i].emitted[slot] = results[i].value
            else:
                weight, bias = w.extra[f"emit_w_{slot}"], w.extra[f"emit_b_{slot}"]
                projected = emit(h[rows] @ weight, fresh[rows] @ weight + bias, p[rows], mode, zeta)
                for i, value in zip(rows, projected):
                    results[i].emitted[slot] = value
        return results


def validate_program(weights, spec, slots):
    program = weights.full_program
    if not isinstance(program, ProjectionEmit):
        if spec.identity or program.profile != spec.emission:
            raise ValueError("custom Full program does not match graph profile")
        return  # Python custom programs own their scalar/batch contract.
    if (program.kind, program.period, program.phases, program.identity) != (
            spec.emission, spec.emit_period, spec.emit_phases, spec.identity):
        raise ValueError("shared Full program does not match graph policy")
    if not spec.identity and weights.full_kind != spec.full:
        raise ValueError("shared Full backbone does not match graph profile")
    from . import lh_full
    if not spec.identity and weights.full_kind in lh_full.PROFILES:
        lh_full.validate(weights)
    if program.kind == "slot_affine":
        width = weights.bias.numel()
        for slot in range(slots):
            for name, shape in ((f"emit_w_{slot}", (width, width)), (f"emit_b_{slot}", (width,))):
                if name not in weights.extra or weights.extra[name].shape != shape:
                    raise ValueError("Full program parameter slot domain mismatch")


def validate(result, request, slots):
    keys = list(result.emitted)
    if keys != sorted(set(keys)) or any(type(k) is not int or not 0 <= k < slots for k in keys):
        raise ValueError("Full returned invalid output slots")
    for value in ([result.value] if result.value is not None else []) + list(result.emitted.values()):
        if not isinstance(value, torch.Tensor) or (value.shape, value.dtype, value.device) != (
                request.content.value.shape, request.content.value.dtype, request.content.value.device):
            raise ValueError("Full returned incompatible tensor metadata")


def bind(packed, reference):
    if packed.emitted.keys() != reference.emitted.keys() or (packed.value is None) != (reference.value is None):
        raise ValueError("Full batch changed output presence")
    value = None if packed.value is None else autograd.value(packed.value, reference.value)
    emitted = {slot: autograd.value(t, value if reference.emitted[slot] is reference.value else reference.emitted[slot])
               for slot, t in packed.emitted.items()}
    return FullResult(value, emitted)


def evaluate(weights, requests, slots, mode, zeta, packed=False, full_autograd="replay"):
    program = weights.full_program
    if full_autograd not in {"replay", "batched"} or (full_autograd == "batched" and not packed):
        raise ValueError("invalid Full autograd policy or unpacked execution")
    if full_autograd == "batched" and type(program) is not ProjectionEmit:
        raise ValueError("Full program has no batched autograd implementation")
    if packed and full_autograd == "batched" and torch.is_grad_enabled():
        from .full_rows import evaluate as batch_grad
        results = batch_grad(program, weights, requests, slots, mode, zeta)
    elif not packed:
        results = [program.step(weights, r, slots, mode, zeta) for r in requests]
    else:
        with torch.no_grad():
            results = program.batch(weights, requests, slots, mode, zeta)
        if len(results) != len(requests):
            raise ValueError("Full batch changed event count")
        for r, result in zip(requests, results):
            validate(result, r, slots)
        if torch.is_grad_enabled():
            for i, r in enumerate(requests):
                reference = program.step(weights, r, slots, mode, zeta)
                validate(reference, r, slots)
                results[i] = bind(results[i], reference)
    for request, result in zip(requests, results):
        validate(result, request, slots)
    return results
