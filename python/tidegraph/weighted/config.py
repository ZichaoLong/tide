"""Concrete bindings for 0770593 KV §1–4 and allocation §2–3/S.1.

Frozen configurations are part of graph identity. Defaults name actual
functions; an unimplemented candidate is never substituted for another one.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


def identity(prefix, *bindings):
    raw = json.dumps([asdict(x) if hasattr(x, "__dataclass_fields__") else x for x in bindings],
                     sort_keys=True, separators=(",", ":"), allow_nan=False)
    return prefix+":"+hashlib.sha256(raw.encode()).hexdigest()


def positive_int(name, value, minimum=1):
    if type(value) is not int or not minimum <= value < 2**63:
        raise ValueError(f"{name} must be an int64 >= {minimum}")


def finite(name, value, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (value < minimum or strict and value == minimum):
        raise ValueError(f"invalid {name} range")


@dataclass(frozen=True)
class KVConfig:
    width: int = 4
    summary_width: int = 2
    adapter_rank: int = 2
    heads: int = 1
    ffn_width: int = 8
    window: int = 0  # 0 denotes infinity; otherwise original logical time.
    memory_read: str = "sum"
    beta_short: float = .1
    epsilon: float = 1e-5
    alpha_att: float = 1.
    alpha_ff: float = 1.
    position_origin: float = 0.
    position_scale: float = 1.
    full_frequencies: tuple[tuple[float, ...], ...] = ()
    read_frequencies: tuple[float, ...] = ()
    distance_slopes: tuple[float, ...] = ()

    def __post_init__(self):
        for name in ("width", "summary_width", "adapter_rank", "heads", "ffn_width"):
            positive_int(name, getattr(self, name))
        positive_int("window", self.window, 0)
        if self.width % self.heads or self.memory_read not in {"sum", "qry"}:
            raise ValueError("invalid KV head layout or memory_read")
        for name in ("beta_short", "epsilon", "position_scale"):
            finite(name, getattr(self, name), 0, True)
        for name in ("alpha_att", "alpha_ff"):
            finite(name, getattr(self, name), 0)
        finite("position_origin", self.position_origin)
        freqs = tuple(tuple(row) for row in self.full_frequencies) or ((),)*self.heads
        slopes = tuple(self.distance_slopes) or (0.,)*self.heads
        read = tuple(self.read_frequencies)
        if (len(freqs) != self.heads or len(slopes) != self.heads
                or any(2*len(row) > self.width//self.heads for row in freqs)
                or 2*len(read) > self.summary_width):
            raise ValueError("invalid KV position layout")
        for value in (*read, *(v for row in freqs for v in row)):
            finite("frequency", value)
        for value in slopes:
            finite("distance slope", value, 0)
        if self.memory_read == "sum" and read:
            raise ValueError("sum memory_read does not use read_frequencies")
        object.__setattr__(self, "full_frequencies", freqs)
        object.__setattr__(self, "distance_slopes", slopes)
        object.__setattr__(self, "read_frequencies", read)


@dataclass(frozen=True)
class RegionalConfig:
    read_mode: str = "+"
    local_score: str = "cat"
    region_score: str = "ind"
    selection_mode: str = "regional_total"
    region_control_mode: str = "context"
    region_evidence: str | None = "cat"
    regularizer: str = "quad"
    hidden_width: int = 4
    region_control_width: int = 2
    gamma_max: float = 1.
    k: int = 1
    allocation_temperature: float = 1.
    total_fraction: float | None = None

    def __post_init__(self):
        for name in ("hidden_width", "region_control_width", "k"):
            positive_int(name, getattr(self, name))
        if self.read_mode not in {"0", "-", "+"}:
            raise ValueError("read_mode must be 0/-/+")
        if (self.local_score, self.region_score, self.selection_mode, self.regularizer) != (
                "cat", "ind", "regional_total", "quad"):
            raise ValueError("this binding currently implements cat/ind/regional_total/quad")
        finite("gamma_max", self.gamma_max, 1)
        finite("allocation_temperature", self.allocation_temperature, 0, True)
        if self.region_control_mode == "context":
            if self.region_evidence != "cat" or self.total_fraction is not None:
                raise ValueError("context requires cat evidence and computes total_fraction")
        elif self.region_control_mode == "fixed":
            if self.region_evidence is not None or self.total_fraction is None:
                raise ValueError("fixed control needs total_fraction and no evidence function")
            finite("total_fraction", self.total_fraction, 0)
            if self.total_fraction > 1:
                raise ValueError("total_fraction exceeds one")
        else:
            raise ValueError("invalid region_control_mode")
