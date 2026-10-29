"""Concrete bindings for 0770593 KV §1–4 and allocation §2–3/S.1.

Frozen configurations are part of graph identity. Defaults name actual
functions; an unimplemented candidate is never substituted for another one.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


def identity(prefix, *bindings):
    raw = json.dumps([x.identity_fields() if hasattr(x, "identity_fields") else
                      asdict(x) if hasattr(x, "__dataclass_fields__") else x for x in bindings],
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
    region_control_mode: str | None = None
    region_evidence: str | None = None
    regularizer: str | None = None
    ent_alpha: float | None = None
    hidden_width: int = 4
    region_control_width: int | None = None
    gamma_max: float = 1.
    k: int | None = None
    allocation_temperature: float | None = None
    total_fraction: float | None = None
    eta: float | None = None
    send_gain: str | None = None
    fixed_gain: float | None = None
    gain_logit: float | None = None
    allocation_base: str | None = None
    selection_width: int | None = None
    region_key: str | None = None
    query_weight: str | None = None
    alpha_match: float | None = None
    alpha_res: float | None = None
    alpha_loc: float | None = None
    alpha_sel: float | None = None

    def __post_init__(self):
        positive_int("hidden_width", self.hidden_width)
        if self.read_mode not in {"0", "-", "+"}:
            raise ValueError("read_mode must be 0/-/+")
        if self.local_score not in {"cat", "match"} or self.region_score not in {"ind", "att"}:
            raise ValueError("invalid local_score/region_score")
        if self.selection_mode not in REGIONAL | GAIN:
            raise ValueError("unknown selection_mode")
        finite("gamma_max", self.gamma_max, 1)
        if self.selection_mode in ALLOCATED | ATTENTION:
            self._default("allocation_temperature", 1.)
            finite("allocation_temperature", self.allocation_temperature, 0, True)
        else:
            self._absent("allocation_temperature")
        if self.selection_mode in ALLOCATED:
            self._default("regularizer", "quad")
            validate_regularizer(self.regularizer, self.ent_alpha)
        else:
            self._absent("regularizer", "ent_alpha")
        if self.selection_mode not in ATTENTION:
            self._default("k", 1)
            positive_int("k", self.k)
            self._absent("allocation_base")
        else:
            self._absent("k", "eta")
            self._default("allocation_base", "uniform" if self.selection_mode == "unit_attention" else "mu")
            allowed = {"direct_attention": {"mu", "received"}, "unit_attention": {"uniform", "mu"},
                       "pooled_attention": {"mu"}}
            if self.allocation_base not in allowed[self.selection_mode]:
                raise ValueError("inapplicable allocation_base")
        if self.regional:
            self._absent("send_gain", "fixed_gain", "gain_logit")
            self._default("region_control_mode", "context")
            self._default("region_control_width", 2)
            positive_int("region_control_width", self.region_control_width)
            if self.region_control_mode == "context":
                self._default("region_evidence", "cat")
                self._absent("eta", "total_fraction")
                if self.region_evidence not in {"cat", "grouped"}:
                    raise ValueError("region_evidence must be cat/grouped")
            elif self.region_control_mode == "fixed":
                self._absent("region_evidence")
                if self.selection_mode in TOTAL:
                    finite("total_fraction", self.total_fraction, 0)
                    if self.total_fraction > 1:
                        raise ValueError("total_fraction exceeds one")
                else:
                    self._default("eta", 0.)
                    finite("eta", self.eta)
            else:
                raise ValueError("invalid region_control_mode")
            self._absent("eta" if self.selection_mode in TOTAL else "total_fraction")
        else:
            self._absent("region_control_mode", "region_control_width", "region_evidence", "total_fraction")
            self._default("send_gain", "fixed")
            if self.send_gain == "context":
                self._absent("fixed_gain", "gain_logit")
            elif self.send_gain == "fixed":
                if self.gain_logit is not None:
                    self._absent("fixed_gain")
                    finite("gain_logit", self.gain_logit)
                else:
                    self._default("fixed_gain", self.gamma_max)
                    finite("fixed_gain", self.fixed_gain, 1)
                    if self.fixed_gain > self.gamma_max:
                        raise ValueError("fixed_gain exceeds gamma_max")
            else:
                raise ValueError("send_gain must be fixed/context")
            if self.selection_mode in {"constrained", "hard_selection"}:
                self._default("eta", 0.)
                finite("eta", self.eta)
            else:
                self._absent("eta")
        if self.local_score == "match" or self.region_score == "att" or self.region_evidence == "grouped":
            self._default("selection_width", 4)
            positive_int("selection_width", self.selection_width)
        else:
            self._absent("selection_width")
        if self.local_score == "match":
            for field in ("alpha_match", "alpha_res"):
                self._default(field, 1.)
                finite(field, getattr(self, field), 0)
        else:
            self._absent("alpha_match", "alpha_res")
        if self.region_score == "att":
            self._default("region_key", "content_stats")
            self._default("query_weight", "received" if self.regional else "mu")
            if (self.region_key not in {"content_stats", "content"} or self.query_weight not in {"mu", "received"}
                    or self.regional and self.query_weight != "received"):
                raise ValueError("invalid region_key/query_weight")
            for field in ("alpha_loc", "alpha_sel"):
                self._default(field, 1.)
                finite(field, getattr(self, field), 0)
        else:
            self._absent("region_key", "query_weight", "alpha_loc", "alpha_sel")

    @property
    def regional(self):
        return self.selection_mode in REGIONAL

    def capacity(self, size):
        if self.k is not None and self.k > size:
            raise ValueError("sending k exceeds region size")
        return self.k if self.selection_mode in HARD else size

    @property
    def backward_identity(self):
        if self.selection_mode in ALLOCATED:
            return self.regularizer+"-exact-map-v1"
        if self.selection_mode in ATTENTION:
            return "weighted-attention-map-v1"
        if self.selection_mode == "proportional":
            return "weighted-proportional-map-v1"
        return "weighted-hard-cap-branch-v1" if self.selection_mode in HARD else "weighted-sorted-fill-branch-v1"

    def _default(self, name, value):
        if getattr(self, name) is None:
            object.__setattr__(self, name, value)

    def _absent(self, *names):
        for name in names:
            if getattr(self, name) is not None:
                raise ValueError(f"{name} is inapplicable to {self.selection_mode}/{self.region_control_mode}")

    def identity_fields(self):
        fields = asdict(self)
        # Preserve pre-entropy quad graph/checkpoint identities. An absent
        # fixed entropy shape has no semantics; an actual shape always enters.
        for name in ("ent_alpha", "eta", "send_gain", "fixed_gain", "gain_logit", "allocation_base",
                     "selection_width", "region_key", "query_weight", "alpha_match", "alpha_res", "alpha_loc", "alpha_sel"):
            if fields[name] is None:
                fields.pop(name)
        return fields


TOTAL = {"regional_total", "regional_total_zero"}
HARD = {"regional_threshold_hard", "hard_selection"}
REGIONAL = TOTAL | {"regional_threshold", "regional_threshold_hard"}
ATTENTION = {"direct_attention", "unit_attention", "pooled_attention"}
ALLOCATED = {"regional_total", "regional_threshold", "constrained"}
GAIN = ATTENTION | {"constrained", "hard_selection", "proportional"}


def region_profile(config):
    return identity("weighted-regional-total-v1" if config.selection_mode == "regional_total" else
                    "weighted-region-v1", config)


def validate_regularizer(kind, alpha):
    """Allocation §6/S.2, shared by static configuration and numeric responses."""
    if kind not in {"quad", "bern", "ent"}:
        raise ValueError("regularizer must be quad/bern/ent")
    if kind == "ent":
        finite("ent_alpha", alpha, 1)
        if alpha > 2:
            raise ValueError("ent_alpha must lie in [1,2]")
    elif alpha is not None:
        raise ValueError("ent_alpha applies only to ent")
