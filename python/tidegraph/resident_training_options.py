"""Explicit first-order resident training storage; no runtime imports."""
from dataclasses import asdict, dataclass
import math


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
