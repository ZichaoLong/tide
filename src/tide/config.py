"""The finite current instance menu, not a general local-program language."""
from dataclasses import asdict, dataclass, field, replace
import math
from .topology import TopologyConfig, _resolve


@dataclass(frozen=True)
class ClockConfig:
    """Local logical clocks; None retains the selected binding's defaults.

    The input schedule remains TopologyConfig.period. Positions are used only
    relatively by KV, so a node-constant origin cancels before rotation.
    ``absolute`` explicitly declares Pos(theta)=theta/position_period for a
    comparison with identical local clocks on both sides. Long-summary decay
    and SSD passive matrix decay remain zero.
    """
    position_period: float | None = None
    position_origin: str = "binding"
    kv_window: int | None = None
    short_decay: float | None = None
    ssd_stride: int | None = None

    def __post_init__(self):
        if self.position_origin not in ("binding", "absolute"):
            raise ValueError("position_origin must be binding or absolute")
        for name, maximum in (("kv_window", 2**63-1), ("ssd_stride", (2**63-1)//3)):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or not 1 <= value <= maximum):
                raise ValueError(f"{name} must be a positive int64-safe logical interval")
        for name in ("position_period", "short_decay"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                                      or value < 0 or name == "position_period" and value == 0):
                raise ValueError(f"{name} must be finite and {'positive' if name == 'position_period' else 'nonnegative'}")

    def resolve(self, period, settle):
        return replace(self,
                       position_period=period if self.position_period is None else self.position_period,
                       kv_window=128*period if self.kv_window is None else self.kv_window,
                       short_decay=math.log(2)/(8*period) if self.short_decay is None else self.short_decay,
                       ssd_stride=(period if settle else 1) if self.ssd_stride is None else self.ssd_stride)


@dataclass(frozen=True)
class ModelConfig:
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    d: int = 128
    memory: str = "K"
    aggregation: str = "source_residual"
    allocation: str = "quad"
    sharing: str = "independent"
    adapter_rank: int = 8
    budget: int | None = None
    attention_backend: str = "auto"
    attention_chunk: int = 128
    input_dim: int | None = None
    checkpoint_regions: bool = False
    checkpoint_reception: bool = False
    reception_qkv_backend: str = "separate"
    # Legacy serialized field; both values retain mandatory topology masking.
    # Keep explicit False readable without changing stored checkpoint config.
    reception_mask_padding: bool = True
    parameter_indexing: str = "gather"
    # Execution-only: compact valid rows within each Full FFN owner.
    ffn_packing: str = "rectangular"
    ffn_weight_access: str = "gather"
    ssd_summary_backend: str = "block"
    ssd_scan_backend: str = "block"
    region_batch_size: int = 1
    local_attention_backend: str | None = None
    hub_allocation: str = "bounded"
    clock: ClockConfig = field(default_factory=ClockConfig)
    # Explicit instance extensions: content/summary widths stay unchanged.
    kv_width: int | None = None
    ffn_width: int | None = None
    # None preserves the selected whole-node profile. The explicit extension
    # overrides memory parameters only; FFN/reception/description retain sharing.
    memory_sharing: str | None = None
    # A missing field in an older checkpoint resolves to its original policy.
    shannon_support_policy: str = "strict"
    pool_support_policy: str = "strict"

    def __post_init__(self):
        if type(self.checkpoint_regions) is not bool:
            raise ValueError("checkpoint_regions must be boolean")
        if type(self.checkpoint_reception) is not bool:
            raise ValueError("checkpoint_reception must be boolean")
        if self.reception_qkv_backend not in ("separate", "fused"):
            raise ValueError("reception_qkv_backend must be separate or fused")
        if type(self.reception_mask_padding) is not bool:
            raise ValueError("reception_mask_padding must be boolean")
        if self.parameter_indexing not in ("gather", "static_view"):
            raise ValueError("parameter_indexing must be gather or static_view")
        if self.ffn_packing not in ("rectangular", "power2"):
            raise ValueError("ffn_packing must be rectangular or power2")
        if self.ffn_weight_access not in ("gather", "grouped"):
            raise ValueError("ffn_weight_access must be gather or grouped")
        if self.ssd_summary_backend not in ("block", "prefix"):
            raise ValueError("ssd_summary_backend must be block or prefix")
        if self.ssd_scan_backend not in ("block", "incremental", "doubling"):
            raise ValueError("ssd_scan_backend must be block, incremental or doubling")
        if type(self.region_batch_size) is not int or self.region_batch_size < 1:
            raise ValueError("region_batch_size must be positive")
        if not isinstance(self.clock, ClockConfig):
            raise TypeError("clock must be ClockConfig")
        if self.hub_allocation not in ("bounded", "fixed_pass"):
            raise ValueError("hub_allocation must be bounded or fixed_pass")
        if self.hub_allocation == "fixed_pass" and (self.topology.kind not in ("exchange", "multiscale")
                or self.topology.partition != "group" or not self.topology.hubs
                or self.topology.hub_mask == "none"):
            raise ValueError("fixed_pass requires an HB group partition with hubs")
        if self.d not in (64, 128, 192, 256, 384, 512, 768):
            raise ValueError("d must be a registered main-instance width")
        if self.memory not in ("K", "S", "X", "X_reverse"):
            raise ValueError("memory must be K, S, X or X_reverse")
        if self.kv_width is not None:
            if type(self.kv_width) is not int or not 0 < self.kv_width <= self.d or self.kv_width % 32:
                raise ValueError("kv_width must be a positive multiple of 32 <= d")
            if self.memory == "S":
                raise ValueError("kv_width is unused by all-SSD memory")
        if self.ffn_width is not None and (type(self.ffn_width) is not int or
                                          self.ffn_width < 1 or self.ffn_width % 16):
            raise ValueError("ffn_width must be a positive multiple of 16")
        if self.aggregation not in ("source_residual", "incoming_light_64", "incoming_light_128", "incoming_wide"):
            raise ValueError("unknown reception profile")
        if self.allocation not in ("quad", "shannon", "pool"):
            raise ValueError("unknown allocation profile")
        if self.pool_support_policy not in ("strict", "positive-floor-v1") or (
                self.pool_support_policy != "strict" and self.allocation != "pool"):
            raise ValueError("positive-floor-v1 pool policy requires pool allocation")
        if self.shannon_support_policy not in ("strict", "positive-floor-v1", "weak-log-v1"):
            raise ValueError("unknown Shannon support policy")
        if self.shannon_support_policy != "strict" and self.allocation != "shannon":
            raise ValueError("a Shannon support policy requires Shannon allocation")
        if self.sharing not in ("independent", "region_shared", "layer_shared"):
            raise ValueError("unknown sharing profile")
        if self.memory_sharing not in (None, "region_shared"):
            raise ValueError("memory_sharing must be None or region_shared")
        if self.adapter_rank not in (4, 8, 16):
            raise ValueError("adapter_rank must be 4, 8 or 16")
        if self.budget is not None and (type(self.budget) is not int or
                self.budget not in (1, 2, 4, 8, 16, 32, 64) or self.budget > self.topology.n):
            raise ValueError("ordinary budget must be a registered power of two <= min(n,64)")
        if self.attention_backend not in ("auto", "math", "sdpa", "tiled", "npu_fusion"):
            raise ValueError("unknown Attention backend")
        if self.local_attention_backend not in (None, "auto", "math", "sdpa", "tiled", "npu_fusion"):
            raise ValueError("unknown local Attention backend")
        if type(self.attention_chunk) is not int or self.attention_chunk < 1:
            raise ValueError("attention_chunk must be positive")
        if self.input_dim is not None and (type(self.input_dim) is not int or self.input_dim < 1):
            raise ValueError("input_dim must be positive")
        if self.allocation == "pool" and self.topology.partition == "mixed" and self.topology.hubs:
            raise ValueError("pool does not preserve the mixed-region Hub bounds")

    @property
    def allocation_support_policy(self):
        return self.pool_support_policy if self.allocation == "pool" else self.shannon_support_policy

    def to_dict(self):
        result = asdict(self)
        if self.pool_support_policy == "strict":
            del result["pool_support_policy"]
        if self.shannon_support_policy == "strict":
            # The omitted default preserves existing resolved recipe identities.
            del result["shannon_support_policy"]
        return result

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        topology = dict(data["topology"])
        if topology.get("probes") is not None:
            topology["probes"] = tuple(topology["probes"])
        if isinstance(topology.get("hub_mask"), list):
            topology["hub_mask"] = tuple(tuple(row) for row in topology["hub_mask"])
        data["topology"] = TopologyConfig(**topology)
        if "clock" in data:
            data["clock"] = ClockConfig(**data["clock"])
        return cls(**data)


def edge_delay_pair(config: ModelConfig, *, period: int) -> tuple[ModelConfig, ModelConfig]:
    """Return (Settle, Timed) differing only in edge-delay interpretation.

    Explicit identical local clocks are required; this does not assert equal
    trajectories, arrival grouping, state writes or output slot layouts.
    """
    if config.topology.kind != "multiscale":
        raise ValueError("an edge-delay pair requires Multiscale")
    clock = config.clock
    if clock.position_origin != "absolute" or any(getattr(clock, name) is None for name in
            ("position_period", "kv_window", "short_decay", "ssd_stride")):
        raise ValueError("an edge-delay pair requires fully explicit absolute local clocks")
    topology, _ = _resolve(replace(config.topology, timing="settle", period=period))
    return (replace(config, topology=topology),
            replace(config, topology=replace(topology, timing="timed")))
