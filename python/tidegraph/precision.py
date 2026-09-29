"""Explicit FP16 payload / FP32 master training, independent of scheduling."""
import copy
import math
import torch


class FP32MasterOptimizer:
    """SGD/AdamW on FP32 copies of unique FP16 parameters.

    Call ``backward(loss)`` (or scaled ``backward`` for each accumulated loss),
    then ``step()``. The loss must be FP32. Scaling is static and recorded;
    nonfinite gradients fail before any optimizer mutation. No update is skipped.
    Session checkpoints include masters, optimizer slots and the scale.
    """
    def __init__(self, parameters, *, optimizer="adamw", loss_scale=128., **options):
        self.payload = list(parameters)
        if (not self.payload or len({id(p) for p in self.payload}) != len(self.payload)
                or any(not isinstance(p, torch.Tensor) or not p.is_leaf or p.dtype != torch.float16
                       or not p.requires_grad for p in self.payload)):
            raise ValueError("unique trainable FP16 leaf parameters required")
        if optimizer not in {"adamw", "sgd"}:
            raise ValueError("FP32 master optimizer must be adamw or sgd")
        self._validate_scale(loss_scale)
        self.loss_scale = float(loss_scale)
        self.kind = optimizer
        self.masters = [p.detach().float().requires_grad_() for p in self.payload]
        factory = torch.optim.AdamW if optimizer == "adamw" else torch.optim.SGD
        self._inner = factory(self.masters, **options)
        self.param_groups = [dict(self._inner.param_groups[0], params=self.payload)]
        self._scaled_backward = False

    @staticmethod
    def _validate_scale(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("loss_scale must be positive and finite")

    @property
    def state(self):
        return self._inner.state

    def zero_grad(self, set_to_none=True):
        self._inner.zero_grad(set_to_none=set_to_none)
        for p in self.payload:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.detach_(); p.grad.zero_()
        self._scaled_backward = False

    def backward(self, loss, **kwargs):
        if not isinstance(loss, torch.Tensor) or loss.dtype != torch.float32 or loss.numel() != 1:
            raise ValueError("mixed-precision training requires a scalar FP32 loss")
        (loss * self.loss_scale).backward(**kwargs)
        self._scaled_backward = True

    @torch.no_grad()
    def step(self):
        if any(p.grad is not None for p in self.payload) and not self._scaled_backward:
            raise ValueError("call optimizer.backward(loss) before step to declare gradient scaling")
        checks = {}
        for p, m in zip(self.payload, self.masters):
            m.grad = None if p.grad is None else p.grad.float() / self.loss_scale
            if m.grad is not None:
                checks.setdefault(m.device, []).append(torch.isfinite(m.grad).all())
        if any(not torch.stack(values).all().item() for values in checks.values()):
            raise FloatingPointError("nonfinite gradient; reduce loss_scale or use float32")
        # Honor caller learning-rate/group changes while keeping owner identity.
        for key, value in self.param_groups[0].items():
            if key != "params": self._inner.param_groups[0][key] = value
        self._inner.step()
        checks = {}
        for p, m in zip(self.payload, self.masters):
            if m.grad is not None:
                p.copy_(m)
                checks.setdefault(p.device, []).append(torch.isfinite(p).all())
        if any(not torch.stack(values).all().item() for values in checks.values()):
            raise FloatingPointError("updated FP16 payload overflow; restore checkpoint and change precision/lr")
        self._scaled_backward = False

    def state_dict(self):
        state = copy.deepcopy(self._inner.state_dict())
        ids = state["param_groups"][0]["params"]
        state["param_groups"][0] = copy.deepcopy({k:v for k,v in self.param_groups[0].items() if k!="params"})
        state["param_groups"][0]["params"] = ids
        state["precision"] = dict(schema="tide-fp32-masters-v1", optimizer=self.kind,
                                  loss_scale=self.loss_scale,
                                  masters=[p.detach().clone() for p in self.masters])
        return state

    def validate_state_dict(self, state):
        from .checkpoint_ownership import validate_optimizer_state
        if not isinstance(state, dict) or set(state) != {"state", "param_groups", "precision"}:
            raise ValueError("invalid FP32 master checkpoint structure")
        policy = state["precision"]
        if (not isinstance(policy, dict) or set(policy) != {"schema", "optimizer", "loss_scale", "masters"}
                or policy["schema"] != "tide-fp32-masters-v1" or policy["optimizer"] != self.kind):
            raise ValueError("FP32 master checkpoint policy mismatch")
        self._validate_scale(policy["loss_scale"])
        values = policy["masters"]
        if not isinstance(values, list) or len(values) != len(self.masters):
            raise ValueError("FP32 master checkpoint owner inventory mismatch")
        for actual, expected in zip(values, self.masters):
            if (not isinstance(actual, torch.Tensor) or actual.dtype != torch.float32
                    or actual.shape != expected.shape or not torch.isfinite(actual).all()):
                raise ValueError("invalid FP32 master checkpoint tensor")
        validate_optimizer_state(self._inner, {k: state[k] for k in ("state", "param_groups")})

    def validate_checkpoint_weights(self, model, weights, state):
        self.validate_state_dict(state)
        names = {id(p): name for name, p in model.named_parameters()}
        for p, master in zip(self.payload, state["precision"]["masters"]):
            value = weights[names[id(p)]]
            if not torch.equal(master.detach().cpu().to(p.dtype), value.detach().cpu()):
                raise ValueError("checkpoint master and payload weights disagree")

    @torch.no_grad()
    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self._inner.load_state_dict({k: state[k] for k in ("state", "param_groups")})
        for live, saved in zip(self.masters, state["precision"]["masters"]):
            live.copy_(saved)
        self.loss_scale = float(state["precision"]["loss_scale"])
        self.param_groups = [dict(self._inner.param_groups[0], params=self.payload)]
        self._scaled_backward = False
