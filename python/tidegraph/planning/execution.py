"""Execution and measurement identities, distinct from mathematical models."""
from dataclasses import asdict, dataclass
import re
from .legacy_topology import integer
from .model import fingerprint
from ..column_policy import POSITION_POLICY as COLUMN_POSITION_POLICY, validate_column_position_record


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
    return_packing: str = "scalar"
    workers: int = 1
    intraop_threads: int = 1
    interop_threads: int = 1
    peer_devices: tuple[str, ...] = ()
    node_devices: tuple[str, ...] = ()
    communication: str = "ordinary"
    return_mode: str = "complete"
    finite_check_packing: str = "scalar"
    query_execution: str = "auto"
    prefill_schedule: str = "auto"
    input_interface: str = "auto"

    def __post_init__(self):
        if self.client not in {"pytorch", "python-native", "libtorch"}:
            raise ValueError("unknown client")
        if self.schedule not in {"streaming", "prefill"}:
            raise ValueError("unknown schedule")
        if self.input_interface not in {"auto", "records"}:
            raise ValueError("input_interface must be auto or records")
        if self.prefill_schedule not in {"auto", "greedy", "window"}:
            raise ValueError("unknown prefill_schedule")
        if self.prefill_schedule != "auto" and (self.schedule != "prefill" or self.placement == "resident"):
            raise ValueError("explicit prefill_schedule requires host prefill")
        if self.placement not in {"native", "cpu", "mixed-a", "mixed-b", "mixed-c", "resident"}:
            raise ValueError("unknown placement")
        if not re.fullmatch(r"cpu|(?:npu|cuda):[0-9]+", self.device):
            raise ValueError("device must be explicit cpu, npu:N or cuda:N")
        if self.dtype not in {"float16", "float32", "float64"} or self.scoring_dtype not in {"float32", "float64"}:
            raise ValueError("unknown explicit precision")
        if self.operator != "aten" or self.packing not in {"auto", "scalar", "packed"}:
            raise ValueError("unknown operator/packing implementation; no implicit substitution")
        if self.query_execution not in {"auto", "legacy", "dense-v1", "packed-v2"}:
            raise ValueError("unknown query_execution")
        if self.query_execution not in {"legacy", "auto"} and (self.placement == "resident"
                or self.schedule != "prefill" or self.packing == "scalar" or self.workers != 1
                or self.peer_devices or self.dtype == "float16"):
            raise ValueError("dense-v1 requires single-device FP32/FP64 packed eager prefill, one worker")
        if self.return_packing not in {"scalar", "packed"}:
            raise ValueError("unknown complete-return packing")
        if self.finite_check_packing not in {"scalar", "packed"}:
            raise ValueError("finite_check_packing must be scalar or packed")
        if self.finite_check_packing != "scalar" and (self.client == "libtorch" or self.placement == "resident"):
            raise ValueError("finite_check_packing=packed applies to installed Python eager clients; standalone/resident keep their existing checks")
        if self.placement == "resident" and self.return_packing != "scalar":
            raise ValueError("packed complete return applies only to eager execution")
        for k in ("workers", "intraop_threads", "interop_threads"):
            integer(getattr(self, k), k)
        peers = tuple(self.peer_devices)
        if peers and (len(peers) < 2 or len(set(peers)) != len(peers) or peers[0] != self.device
                      or any(not re.fullmatch(self.device.split(":")[0]+r":[0-9]+", p) for p in peers)):
            raise ValueError("peer devices must be distinct same-backend accelerators, first equal to device")
        object.__setattr__(self, "peer_devices", peers)
        owners = tuple(self.node_devices)
        available = peers or (self.device,)
        if any(d not in available for d in owners):
            raise ValueError("node_devices must use device/peer_devices; provide one owner for every node")
        if owners and self.placement != "resident" and owners[0] != self.device:
            raise ValueError("eager node_devices[0] must equal device")
        object.__setattr__(self, "node_devices", owners)
        if self.communication not in {"ordinary", "blocking", "cpu_bridge"}:
            raise ValueError("communication must be ordinary, blocking or cpu_bridge")
        if self.return_mode not in {"complete", "outputs", "metrics"}:
            raise ValueError("return_mode must be complete, outputs or metrics")
        if self.placement == "resident" and self.communication != "ordinary":
            raise ValueError("resident requires its device peer protocol; eager communication policies do not apply")
        if self.device == "cpu" and self.placement not in {"native", "cpu"}:
            raise ValueError("accelerator placement requires an accelerator")
        if self.device != "cpu" and self.placement == "cpu":
            raise ValueError("CPU placement requires CPU payload")
        if self.dtype == "float64":
            if self.placement == "resident" or peers or self.device.startswith("npu:"):
                raise ValueError("FP64 benchmark requires CPU or single-card CUDA eager execution")
            if self.device.startswith("cuda:") and self.scoring_dtype != "float64":
                raise ValueError("CUDA FP64 benchmark requires explicit float64 scoring")
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

    @classmethod
    def from_record(cls, record):
        """Decode omitted historical defaults without adopting newer policies."""
        validate_column_position_record(record)
        requested = dict(record.get("requested", record))
        if "requested" in record:
            requested.setdefault("input_interface", "records")
            requested.setdefault("query_execution", "legacy")
            host_prefill = (requested.get("schedule", "prefill") == "prefill"
                            and requested.get("placement", "native") != "resident")
            requested.setdefault("prefill_schedule", "greedy" if host_prefill else "auto")
        return cls(**requested)

    def record(self):
        requested = asdict(self)
        columns_policy = (self.input_interface == "auto" and self.return_mode != "complete"
            and self.placement != "resident" and self.schedule == "prefill"
            and self.prefill_schedule != "greedy" and self.query_execution in {"auto", "dense-v1"}
            and self.packing != "scalar" and self.workers == 1)
        if not columns_policy:
            requested.pop("input_interface")
        # Explicit greedy reproduces the pre-window execution identities.
        # Streaming/resident do not use the host prefill strategy at all.
        window_policy = self.schedule == "prefill" and self.placement != "resident" and self.prefill_schedule != "greedy"
        if not window_policy:
            requested.pop("prefill_schedule")
        if self.query_execution == "legacy":
            requested.pop("query_execution")
        if self.finite_check_packing == "scalar":
            requested.pop("finite_check_packing")
        extended = (bool(self.node_devices) or self.communication != "ordinary" or self.return_mode != "complete"
                    or self.finite_check_packing != "scalar" or self.query_execution != "legacy")
        if not extended:
            # Historical v1 identities remain byte-for-byte reproducible. These
            # default values introduce no new selection into an old record.
            for key in ("node_devices", "communication", "return_mode"):
                requested.pop(key)
        effective = dict(requested)
        selected_boundary = self.return_mode != "complete" and self.placement != "resident"
        if selected_boundary:
            effective["record_policy"] = "selected-boundary-v2; explicit task observations"
        if columns_policy:
            effective["input_policy"] = "tensor-window-v1; built-in output task and compatible Settle modules"
            effective["position_policy"] = COLUMN_POSITION_POLICY
        if window_policy:
            effective["prefill_schedule"] = "window"
            effective["scheduler"] = "component-window"
        if self.packing == "auto":
            effective["packing"] = "scalar" if self.client == "pytorch" and self.schedule == "streaming" else "packed"
        if self.placement == "resident":
            effective["operator"] = "aten+device-control"
        if self.query_execution == "auto":
            effective["query_policy"] = "kv-block-v1+packed-v2+describe-batch-v2; independent capabilities, dtype/schedule fallback to legacy"
            # A record-query fallback is not a capability decision for the
            # tensor-window interface. Its local kernels resolve independently.
            if not columns_policy and (self.placement == "resident" or self.schedule == "streaming"
                    or self.packing == "scalar" or self.workers != 1 or self.dtype == "float16" or self.peer_devices):
                effective["query_execution"] = "legacy"
        body = dict(schema="tide-weighted-execution-v6" if columns_policy else
                    "tide-weighted-execution-v4" if selected_boundary else
                    "tide-weighted-execution-v3" if window_policy else
                    "tide-weighted-execution-v2" if extended else "tide-weighted-execution-v1",
                    requested=requested, effective=effective,
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
        if type(self.checkpoint_check) is not bool:
            raise ValueError("checkpoint check must be bool")
        body = dict(schema="tide-weighted-measurement-v1", workload_id=workload_id,
                    execution_id=execution_id, **asdict(self))
        return dict(body, measurement_id=fingerprint(body))
