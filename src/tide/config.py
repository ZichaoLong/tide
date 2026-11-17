"""The finite current instance menu, not a general local-program language."""
from dataclasses import asdict, dataclass, field
from .topology import TopologyConfig


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

    def __post_init__(self):
        if self.d not in (64, 128, 192, 256, 384, 512, 768):
            raise ValueError("d must be a registered main-instance width")
        if self.memory not in ("K", "S", "X"):
            raise ValueError("memory must be K, S or X")
        if self.aggregation not in ("source_residual", "incoming_light_64", "incoming_light_128", "incoming_wide"):
            raise ValueError("unknown reception profile")
        if self.allocation not in ("quad", "shannon", "pool"):
            raise ValueError("unknown allocation profile")
        if self.sharing not in ("independent", "region_shared", "layer_shared"):
            raise ValueError("unknown sharing profile")
        if self.adapter_rank not in (4, 8, 16):
            raise ValueError("adapter_rank must be 4, 8 or 16")
        if self.budget is not None and (type(self.budget) is not int or
                self.budget not in (1, 2, 4, 8, 16, 32, 64) or self.budget > self.topology.n):
            raise ValueError("ordinary budget must be a registered power of two <= min(n,64)")
        if self.attention_backend not in ("auto", "math", "sdpa", "tiled", "npu_fusion"):
            raise ValueError("unknown Attention backend")
        if type(self.attention_chunk) is not int or self.attention_chunk < 1:
            raise ValueError("attention_chunk must be positive")
        if self.input_dim is not None and (type(self.input_dim) is not int or self.input_dim < 1):
            raise ValueError("input_dim must be positive")
        if self.allocation == "pool" and self.topology.partition == "mixed" and self.topology.hubs:
            raise ValueError("pool does not preserve the mixed-region Hub bounds")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        topology = dict(data["topology"])
        if topology.get("probes") is not None:
            topology["probes"] = tuple(topology["probes"])
        if isinstance(topology.get("hub_mask"), list):
            topology["hub_mask"] = tuple(tuple(row) for row in topology["hub_mask"])
        data["topology"] = TopologyConfig(**topology)
        return cls(**data)
