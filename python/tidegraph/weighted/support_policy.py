"""Explicit Shannon positive storage and unfloored response VJP (TL-REQ-003).

The strict path is unchanged. This is a first-order numerical contract, not
the derivative of rounding/flooring. See docs/positive-support-policy.md.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import torch

FLOOR = "positive-floor-v1"
_observer = ContextVar("weighted_support_observer", default=None)


def validate(policy, kind, alpha):
    if policy not in {"strict", FLOOR}:
        raise ValueError("support_policy must be strict or positive-floor-v1")
    if policy != "strict" and (kind != "ent" or alpha != 1):
        raise ValueError("positive-floor-v1 currently requires Shannon (ent_alpha=1)")


def smallest(reference):
    if reference.dtype not in {torch.float32, torch.float64}:
        raise ValueError("positive-floor-v1 requires FP32 or FP64")
    f = torch.finfo(reference.dtype)
    return reference.new_tensor(f.tiny*f.eps)


def repair(controls, response, caps, temperature, lower, upper, reject):
    """Store only lost positive controls; reject material mass/cap distortion.

    Rows include only ordinary (not zero/caps endpoint) allocations. The
    relative perturbation check is additional to the existing residual check.
    """
    log_response = response/temperature-1
    reject(~torch.isfinite(log_response).all(-1), "positive-floor-v1 response exceeds finite log range")
    lost = (caps > 0) & (controls == 0)
    delta = smallest(controls)
    result = torch.where(lost, delta, controls)
    reject(((lost & (result <= 0)) | (result > caps)).any(-1),
           "positive-floor-v1 storage cannot preserve positive support/caps")
    added = (result-controls).sum(-1)
    scale = torch.maximum(controls.sum(-1), lower)
    bound = 64*controls.shape[-1]*torch.finfo(controls.dtype).eps*scale
    # Reject all-underflow/weak-total rows: an absolute tolerance alone would
    # hide order-one changes in a very small allocation.
    reject((added > bound) | (result.sum(-1) > upper+bound),
           "positive-floor-v1 mass perturbation exceeds relative budget")
    observer = _observer.get()
    if observer is not None:
        observer.add(lost, controls.dtype)
    return result, lost


def signed_exp_product(log_scale, value):
    safe = torch.where(value != 0, value.abs(), torch.ones_like(value))
    result = (log_scale+safe.log()).exp()*value.sign()
    return torch.where(value != 0, result, torch.zeros_like(result))


def shannon_vjp(response, caps, temperature, grad, constrained):
    """KKT response VJP from x/tau, never from the floored control.

    Leading dimensions are independent rows. Log contractions retain a tiny
    slope when its product with a representable cotangent is representable.
    """
    log_c = response/temperature-1
    log_cap = torch.where(caps > 0, caps, torch.ones_like(caps)).log()
    free = (caps > 0) & (log_c < log_cap)
    saturated = (caps > 0) & ~free
    log_a = torch.where(free, log_c-temperature.log(), -torch.inf)
    any_free = free.any(-1)
    peak = torch.where(any_free, log_a.max(-1).values, torch.zeros_like(any_free, dtype=grad.dtype))
    scaled = (log_a-peak.unsqueeze(-1)).exp()
    total = scaled.sum(-1)
    anchor = grad.gather(-1, log_a.argmax(-1).unsqueeze(-1)).squeeze(-1)
    relative = grad-anchor.unsqueeze(-1)
    weighted = signed_exp_product(log_a-peak.unsqueeze(-1), relative)
    offset = weighted.sum(-1)/torch.where(total > 0, total, torch.ones_like(total))
    use = constrained & any_free
    mean = torch.where(use, anchor+offset, torch.zeros_like(anchor))
    centered = torch.where(use.unsqueeze(-1), relative-offset.unsqueeze(-1), grad)
    ds = signed_exp_product(log_a, centered)
    dc = saturated.to(grad.dtype)*centered
    ratio = response/temperature
    safe = torch.where(ratio != 0, ratio.abs(), torch.ones_like(ratio))
    dt = -signed_exp_product(log_a+safe.log(), centered)*ratio.sign()
    dt = dt.sum(-1)
    if not all(torch.isfinite(x).all() for x in (ds, dc, mean, dt)):
        raise ArithmeticError("positive-floor-v1 allocation VJP exceeds requested dtype range")
    return ds, dc, mean, dt


class SupportObservation:
    def __init__(self):
        self.counts = {}
        self.positions = {}

    def add(self, lost, dtype):
        key = (str(lost.device), str(dtype), lost.shape[-1])
        value = torch.stack((lost.sum(), lost.any(-1).sum())).detach()
        self.counts[key] = value if key not in self.counts else self.counts[key]+value
        # Bounded local candidate-index histogram, no per-call host transfer.
        if key in self.positions or len(self.positions) < 16:
            counts = lost.reshape(-1,lost.shape[-1])[:, :32].sum(0).detach()
            self.positions[key] = counts if key not in self.positions else self.positions[key]+counts

    def metrics(self):
        """Synchronize only on explicit collection, normally at a run boundary."""
        values = [v.cpu().tolist() for v in self.counts.values()]
        return dict(policy=FLOOR, repaired_coordinates=sum(v[0] for v in values),
                    repaired_rows=sum(v[1] for v in values),
                    positions=[dict(device=k[0],dtype=k[1],candidates=k[2],
                                    first_candidate_counts=v.cpu().tolist()) for k,v in self.positions.items()],
                    position_limit=dict(groups=16,candidates_per_group=32),
                    scope="actual allocation calls, including autograd replays")


@contextmanager
def observe_support():
    observer = SupportObservation()
    token = _observer.set(observer)
    try:
        yield observer
    finally:
        _observer.reset(token)
