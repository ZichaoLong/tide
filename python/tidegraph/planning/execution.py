"""Execution and measurement identities, distinct from mathematical models."""
from dataclasses import asdict, dataclass
import re
from .legacy_topology import integer
from .model import fingerprint


@dataclass(frozen=True)
class ExecutionSpec:
    client: str = "pytorch"
    device: str = "cpu"
    placement: str = "native"
    dtype: str = "float32"
    scoring_dtype: str = "float32"
    schedule: str = "prefill"
    operator: str = "aten"
    packing: str = "auto"
    workers: int = 1
    intraop_threads: int = 1
    interop_threads: int = 1
    peer_devices: tuple[str, ...] = ()

    def __post_init__(self):
        if self.client not in {"pytorch", "python-native", "libtorch"}:
            raise ValueError("unknown client")
        if self.schedule not in {"streaming", "prefill"}:
            raise ValueError("unknown schedule")
        if self.placement not in {"native", "cpu", "mixed-a", "mixed-b", "mixed-c", "resident"}:
            raise ValueError("unknown placement")
        if not re.fullmatch(r"cpu|(?:npu|cuda):[0-9]+", self.device):
            raise ValueError("device must be explicit cpu, npu:N or cuda:N")
        if self.dtype not in {"float16", "float32", "float64"} or self.scoring_dtype not in {"float32", "float64"}:
            raise ValueError("unknown explicit precision")
        if self.operator != "aten" or self.packing not in {"auto", "scalar", "packed"}:
            raise ValueError("unknown operator/packing implementation; no implicit substitution")
        for k in ("workers", "intraop_threads", "interop_threads"):
            integer(getattr(self, k), k)
        peers = tuple(self.peer_devices)
        if peers and (len(peers) < 2 or len(set(peers)) != len(peers) or peers[0] != self.device
                      or any(not re.fullmatch(self.device.split(":")[0]+r":[0-9]+", p) for p in peers)):
            raise ValueError("peer devices must be distinct same-backend accelerators, first equal to device")
        object.__setattr__(self, "peer_devices", peers)
        if self.device == "cpu" and self.placement not in {"native", "cpu"}:
            raise ValueError("accelerator placement requires an accelerator")
        if self.device != "cpu" and self.placement == "cpu":
            raise ValueError("CPU placement requires CPU payload")
        if self.dtype == "float64" and (self.device != "cpu" or peers):
            raise ValueError("FP64 benchmark requires CPU eager execution")
        if self.client == "pytorch" and (self.workers != 1 or self.placement == "resident"):
            raise ValueError("pure PyTorch owns an independent host scheduler; resident uses a native client")
        if self.client == "pytorch" and self.schedule == "streaming" and self.packing == "packed":
            raise ValueError("Python streaming has scalar dispatch; use prefill for packed execution")
        if self.placement == "resident" and (self.packing == "scalar" or self.workers != 1):
            raise ValueError("resident controls its device batches; host workers/scalar override do not apply")
        if self.placement == "resident" and self.scoring_dtype != "float32":
            raise ValueError("resident scoring requires float32")
        if self.device.startswith("npu:") and self.scoring_dtype == "float64" and self.placement != "mixed-a":
            raise ValueError("NPU FP64 scoring requires explicit CPU Read/control (mixed-a)")

    def record(self):
        requested = asdict(self)
        effective = dict(requested)
        if self.packing == "auto":
            effective["packing"] = "scalar" if self.client == "pytorch" and self.schedule == "streaming" else "packed"
        if self.placement == "resident":
            effective["operator"] = "aten+device-control"
        body = dict(schema="tide-weighted-execution-v1", requested=requested, effective=effective,
                    qualification="implementation selection only; exact-device evidence is separate")
        return dict(body, execution_id=fingerprint(body))


@dataclass(frozen=True)
class MeasurementSpec:
    mode: str = "training"
    steps: int = 2
    warmup: int = 1
    checkpoint_check: bool = True
    timeout_seconds: int = 1800

    def record(self, workload_id, execution_id):
        if self.mode not in {"inference", "training"} or self.warmup != 1:
            raise ValueError("unknown mode or unsupported warmup policy")
        integer(self.steps, "steps", maximum=8)
        integer(self.timeout_seconds, "timeout seconds")
        if type(self.checkpoint_check) is not bool or self.checkpoint_check and self.mode != "training":
            raise ValueError("checkpoint check requires training")
        body = dict(schema="tide-weighted-measurement-v1", workload_id=workload_id,
                    execution_id=execution_id, **asdict(self))
        return dict(body, measurement_id=fingerprint(body))
