"""Explicit first-order resident training storage; no runtime imports."""
from dataclasses import asdict, dataclass
import math
import re


@dataclass(frozen=True)
class ResidentTrainingLimits:
    windows: int = 8
    retained_bytes: int = 128 * 1024 * 1024
    backward_bytes: int = 512 * 1024 * 1024
    optimizer_bytes: int = 128 * 1024 * 1024
    program_workspace_bytes: int = 64 * 1024 * 1024
    reverse_chunk_rows: int = 16

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int or not 0 < value < 1 << 63:
                raise ValueError(f"invalid resident training capacity: {key}")

    def to_dict(self):
        return asdict(self)


GROUP_FIELDS = ("parameters", "lr", "weight_decay", "momentum", "dampening",
                "beta1", "beta2", "eps", "nesterov", "amsgrad", "maximize")
STATE_FIELDS = ("values", "first", "second", "maximum", "steps", "corrections")


def optimizer_groups(core, groups):
    if groups is None:
        return []
    if not isinstance(groups, (tuple, list)):
        raise ValueError("optimizer groups require a list of named option dictionaries")
    result = []
    for spec in groups:
        if not isinstance(spec, dict) or set(spec) - set(GROUP_FIELDS):
            raise ValueError("unknown optimizer group fields")
        group = core.OptimizerGroup()
        for name, value in spec.items():
            if name == "parameters":
                if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) for x in value):
                    raise ValueError("optimizer group parameters require names")
            elif name in {"nesterov", "amsgrad", "maximize"}:
                if type(value) is not bool:
                    raise ValueError(f"optimizer option {name} requires bool")
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"optimizer option {name} requires a finite number")
            setattr(group, name, value)
        result.append(group)
    return result


@dataclass(frozen=True)
class ResidentPlacement:
    """Logical owners of one accelerator family; empty means one device."""
    devices: tuple[str, ...] = ()
    policy: str = "locality"
    full_owners: tuple[int, ...] = ()
    state_owners: tuple[int, ...] = ()

    def __post_init__(self):
        for name in ("devices", "full_owners", "state_owners"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"resident {name} requires a sequence")
            object.__setattr__(self, name, tuple(value))
        if self.policy not in {"memory", "locality"}:
            raise ValueError("resident placement policy must be memory or locality")
        if (len(self.devices) > 16 or any(not isinstance(d, str) or not re.fullmatch(r"(?:npu|cuda):(0|[1-9][0-9]*)", d)
                                         or int(d.split(":")[1]) > 127 for d in self.devices)
                or len(set(self.devices)) != len(self.devices)
                or len({d.split(":")[0] for d in self.devices}) > 1):
            raise ValueError("resident placement requires distinct explicit logical devices of one backend")
        for name in ("full_owners", "state_owners"):
            if any(type(x) is not int or not 0 <= x < len(self.devices) for x in getattr(self, name)):
                raise ValueError(f"invalid resident {name} indices")

    def to_dict(self):
        return {"devices": list(self.devices), "policy": self.policy,
                "full_owners": list(self.full_owners), "state_owners": list(self.state_owners)}

    def native(self, module, coordinator, nodes):
        import torch
        if self.devices and torch.device(self.devices[0]) != coordinator:
            raise ValueError("resident placement must start with the runtime coordinator")
        for name in ("full_owners", "state_owners"):
            if getattr(self, name) and len(getattr(self, name)) != nodes:
                raise ValueError(f"resident {name} must describe every execution node")
        value = module.TrainingPlacement()
        value.devices = [torch.device(d) for d in self.devices]
        value.policy, value.full_owners, value.state_owners = self.policy, self.full_owners, self.state_owners
        return value
