"""Explicit numerical policy for target checks and complete training profiles."""
from dataclasses import dataclass
import math
import torch
from tidegraph.precision import FP32MasterOptimizer
from tidegraph.qualification_checks import compare_finite


@dataclass(frozen=True)
class Precision:
    dtype: torch.dtype = torch.float32
    optimizer: str = "adamw"
    loss_scale: float | None = None

    def __post_init__(self):
        if self.dtype not in {torch.float16, torch.float32}:
            raise ValueError("target checks require float16 or float32")
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be adamw or sgd")
        if self.loss_scale is not None and (not math.isfinite(self.loss_scale) or self.loss_scale <= 0):
            raise ValueError("loss scale must be positive and finite")
        if self.dtype != torch.float16 and self.loss_scale not in (None, 1.):
            raise ValueError("loss scaling requires explicit float16 payload")

    @property
    def scale(self):
        return (16. if self.dtype == torch.float16 else 1.) if self.loss_scale is None else self.loss_scale

    @property
    def tolerances(self):
        return dict(atol=1e-3, rtol=2e-2) if self.dtype == torch.float16 else dict(atol=1e-6, rtol=1e-5)

    def compare(self, a, b, path="root"):
        # FP32 scoring/master values inherit half operand error. Int/bool/None
        # and route/record identities remain exact in the common comparator.
        compare_finite(a, b, path, **self.tolerances)

    def make_optimizer(self, model):
        parameters = [p for p in model.parameters() if p.requires_grad]
        if self.dtype == torch.float16:
            return FP32MasterOptimizer(parameters, optimizer=self.optimizer, lr=.001, loss_scale=self.scale)
        factory = torch.optim.AdamW if self.optimizer == "adamw" else torch.optim.SGD
        return factory(parameters, lr=.001)

    def backward(self, optimizer, loss):
        if self.dtype == torch.float16:
            optimizer.backward(loss)
        else:
            loss.backward()

    def gradients(self, model):
        return {n: None if p.grad is None else p.grad.float()/self.scale
                for n, p in model.named_parameters()}

    def record(self):
        return dict(payload_dtype=str(self.dtype), optimizer=self.optimizer, learning_rate=.001,
                    loss_scale=self.scale, master_dtype="torch.float32" if self.dtype == torch.float16 else None,
                    tolerances=self.tolerances)
