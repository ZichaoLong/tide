"""0770593 allocation §6/S.2: response, inverse bracket and response derivatives.

The inverse bracket is never used as the backward response point. Alpha is a
fixed shape, not a learnable tensor. No activity threshold or near-zero clip.
"""
from dataclasses import dataclass
import torch
from .config import finite


@dataclass(frozen=True)
class Regularizer:
    kind: str
    alpha: float | None = None

    def __post_init__(self):
        if self.kind not in {"quad", "bern", "ent"}:
            raise ValueError("regularizer must be quad/bern/ent")
        if self.kind == "ent":
            finite("ent_alpha", self.alpha, 1)
            if self.alpha > 2:
                raise ValueError("ent_alpha must lie in [1,2]")
        elif self.alpha is not None:
            raise ValueError("ent_alpha applies only to ent")

    def response(self, x, caps, temperature):
        z = x/temperature
        if self.kind == "quad":
            return torch.minimum(z.clamp_min(0), caps)
        if self.kind == "bern":
            return caps*torch.sigmoid(z)
        safe_caps = torch.where(caps > 0, caps, torch.ones_like(caps))
        log_cap = safe_caps.log()
        if self.alpha == 1:
            log_value = z-1
        else:
            delta = self.alpha-1
            log_value = torch.log1p((delta*(z-1/self.alpha)).clamp_min(-1))/delta
        c = torch.where(log_value >= log_cap, caps, torch.exp(torch.minimum(log_value, log_cap)))
        return torch.where(caps > 0, c, torch.zeros_like(c))

    def inverse_bracket(self, fraction, caps, temperature):
        if self.kind == "quad":
            return temperature*fraction*caps
        if self.kind == "bern":
            return temperature*(fraction.log()-torch.log1p(-fraction)).expand_as(caps)
        log_c = fraction.log()+torch.where(caps > 0, caps, torch.ones_like(caps)).log()
        if self.alpha == 1:
            return temperature*(1+log_c)
        delta = self.alpha-1
        return temperature*(torch.expm1(delta*log_c)/delta+1/self.alpha)

    def derivatives(self, x, caps, temperature, controls):
        if self.kind == "bern":
            z = x/temperature
            p, opposite = torch.sigmoid(z), torch.sigmoid(-z)
            # Using 1-p would erase representable positive slopes near p=1.
            a, e = caps*p*opposite/temperature, torch.where(caps > 0, p, torch.zeros_like(p))
        else:
            free = (controls > 0) & (controls < caps)
            a = (free.to(controls.dtype)/temperature if self.kind == "quad" else
                 torch.where(free, controls.pow(2-self.alpha)/temperature, torch.zeros_like(controls)))
            e = ((controls == caps) & (caps > 0)).to(controls.dtype)
        # Saturated coordinates have zero derivative even if x/tau overflowed.
        temperature_slope = -a*torch.where(a != 0, x/temperature, torch.zeros_like(x))
        return a, e, temperature_slope

    def positive_support(self, x, caps, temperature):
        if self.kind == "bern" or self.kind == "ent" and self.alpha == 1:
            return caps > 0
        threshold = 0 if self.kind == "quad" else -temperature/(self.alpha*(self.alpha-1))
        return (caps > 0) & (x > threshold)
