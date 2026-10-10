"""Bounded resident storage; physical chunk sizes do not change logical groups."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResidentLimits:
    queue: int = 1024
    arrivals: int = 1024
    outputs: int = 1024
    trace: int = 4096
    stages: int = 4096
    workspace_bytes: int = 64 * 1024 * 1024
    chunk_policy: str = "conservative"
    full_chunk_rows: int = 16
    emission_chunk_rows: int = 16
    aggregate_chunk_rows: int = 8
    attention_chunk_rows: int = 8
    attention_key_rows: int = 128
    kv_rows: int = 128
    kv_trace_rows: int = 4096
    max_repeat_ticks: int = 65536

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "chunk_policy":
                if value not in {"conservative", "aggressive"}:
                    raise ValueError("unknown resident chunk policy")
            elif type(value) is not int or value < (0 if key == "trace" else 1) or value > (1 << 63) - 1:
                raise ValueError(f"invalid resident capacity: {key}")

    def to_dict(self):
        return asdict(self)


def validate_resident(config, options, device, placement):
    if device.type not in {"npu", "cuda"} or config.dtype not in {"float32", "float16"}:
        raise ValueError("resident backend requires CUDA/NPU FP32/FP16")
    if config.dtype == "float16" and placement["scoring_dtype"] == "payload":
        raise ValueError("resident FP16 payload requires explicit FP32 or profile scoring")
    if any(placement[key] != device for key in ("read", "control", "selection", "events")):
        raise ValueError("resident backend requires Read/control/selection/events on the payload device")
    if placement["scoring_dtype"] not in {"profile", "payload", "float32"}:
        raise ValueError("resident backend requires FP32 scoring")
    if options.implementation != "native":
        raise ValueError("resident backend currently requires a native implementation")
    if options.schedule not in {"streaming", "greedy"}:
        raise ValueError("resident backend requires streaming or general greedy schedule")
    if not options.packed or options.workers != 1:
        raise ValueError("resident backend requires packed execution and one host submitter")
    from .execution_options import ExecutionOptions
    defaults = ExecutionOptions()
    for key in ("full_autograd", "aggregate_autograd", "parallel_regions", "compact_events",
                "defer_state_release", "packed_sources", "batch_next", "attention_packing",
                "fiber_pooling", "fiber_cache", "attention_layout", "max_events"):
        if getattr(options, key) != getattr(defaults, key):
            raise ValueError(f"host scheduler option {key} is unavailable for resident execution")
    if options.trace and options.resident_limits.trace == 0:
        raise ValueError("resident diagnostics require a positive trace capacity")
    from .placement import read_dtype
    for node in config.graph.nodes:
        import torch
        read_dtype(node, torch.float32, placement)


def continuation_device_budgets(values):
    if values is None:
        return {}
    if (type(values) is not dict or any(type(k) is not int or not 0 <= k <= 127
            or type(v) is not int or not 0 <= v < 2**63 for k,v in values.items())):
        raise ValueError("invalid device continuation per-device budgets")
    return values
