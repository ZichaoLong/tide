"""Boundary packing shared by resident inference and explicit training clients."""
import torch
from .coordinates import window_inputs


def external_window(runtime, batch_size, cut, inputs, stop, sealed_until):
    r = runtime
    if r.spec:
        if stop is not None or sealed_until is not None:
            raise ValueError("Settle advances whole positions; do not supply logical stop/seal")
        if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 3 or inputs.shape[0] != batch_size
                or inputs.shape[2] != r.config.width or inputs.dtype != torch.float32
                or inputs.device not in (torch.device("cpu"), r.device)):
            raise ValueError("Settle resident inputs require matching CPU/NPU FP32 [batch,positions,width]")
        position = cut // r.spec.stride
        external = r.spec.external(inputs, position, encoded=True)
        stop = sealed_until = (position + inputs.shape[1]) * r.spec.stride
    else:
        if stop is None or sealed_until is None:
            raise ValueError("PDG/TimedDAG require explicit stop and sealed_until")
        external = window_inputs(inputs, stop, sealed_until)
    core = r.engine.core
    return [core.External(x.batch, x.port, x.position, x.time, x.value) for x in external], stop, sealed_until


def forward_limits(runtime, *, training=False):
    limits = runtime.engine.module.Limits()
    for name, value in runtime.options.resident_limits.to_dict().items():
        if name == "chunk_policy":
            value = getattr(runtime.engine.module.ChunkPolicy, value)
        setattr(limits, name, value)
    limits.prefill = runtime.options.prefill
    limits.diagnostics = training or runtime.options.trace
    return limits
