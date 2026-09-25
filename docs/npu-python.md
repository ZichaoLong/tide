# Python TorchNPU path

The Python graph implementation now has one device boundary.  Callers resolve
the logical backend through `tidegraph.runtime.resolve_device`, construct a
`Model(..., device=device)`, and create input tensors on that same device.  The
graph, scheduling, state, Full, Aggregate, Read and Next code remains shared;
TorchNPU is imported only by the runtime boundary when `npu` is explicitly
requested.

The current verified scope is intentionally narrow: eager Python FP32 on the
local aarch64 Ascend 910 stack (PyTorch 2.10.0+cpu, torch-npu 2.10.0, CANN
9.0.0).  The smoke covers generic PDG streaming, TimedDAG frontier and
diamond, generic and layered Settle, a Settle chain, one first-order backward
case, and isolated Full/Aggregate VJPs.  It compares the same CPU-created
weights and inputs against the CPU result.  Run it only after loading the
documented module stack and selecting a currently free physical device:

```bash
source /usr/share/Modules/init/bash
module use "$HOME/privatemodules"
module purge
module load zlong_envs anaconda/3
ASCEND_RT_VISIBLE_DEVICES=PHYSICAL_ID \
  python scripts/npu_smoke.py --device npu --with-backward \
  --output-dir artifacts/npu-smoke-UNIQUE
```

The logical index inside the process is `npu:0` after visibility remapping.
The smoke is a correctness qualification, not a throughput claim.  The
TorchNPU stack emits shared-installation owner warnings on this host; they are
retained in the launch log and are not treated as proof of fallback because a
real synchronized operation and CPU comparison also pass.  A supported
operator trace/profiler is still required before claiming that every optimized
path is free of host fallback.

FP64 is not part of the NPU scope.  The local Ascend 910 TorchNPU release
rejects the FP64 matmul used by the graph Full path, so an NPU model must use
FP32 and the runtime fails early when `Model(..., device="npu", dtype=float64)`
is used.  The CPU FP64/FP32 contract remains unchanged.

The native C++ executors remain a separate target.  This repository still has
only CPU LibTorch binaries; a Python `torch_npu` import is not a
`libtorch_npu` SDK.  Native NPU stays unverified until a version-matched
standalone SDK, CMake/ABI and loader closure are built and a live-device C++
smoke passes.
