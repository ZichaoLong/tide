# Python TorchNPU qualification

This is a bounded correctness qualification for the Python graph implementation,
not a performance or whole-stack accelerator claim. It records the local
aarch64 host, the exact source identity and the cases that actually executed.

## Source, command and runtime

The smoke ran from clean commit `781141882259ad66269dc87533bd82a893fb69a5`
on 2026-09-25. Its manifest records `dirty: ""` and is preserved at
`artifacts/npu-smoke-20260925-i/smoke.json`; the portable checkpoint produced by
the run is next to that manifest. Before launch, `npu-smi info` showed no
process on physical NPU 6. The launcher remapped it with
`ASCEND_RT_VISIBLE_DEVICES=6`, so the process used logical `npu:0`.

The exact workload command, after loading the documented account module stack,
was:

```bash
source /usr/share/Modules/init/bash
module use "$HOME/privatemodules"
module purge
module load zlong_envs anaconda/3
ASCEND_RT_VISIBLE_DEVICES=6 \
  python scripts/npu_smoke.py --device npu --with-backward \
  --output-dir artifacts/npu-smoke-20260925-i
```

The resolved manifest is:

| Field | Value |
| --- | --- |
| host | `aarch64`, Ascend 910 (Ascend910_9392) |
| visible device | one logical NPU, `npu:0` (physical 6) |
| framework | PyTorch `2.10.0+cpu`, TorchNPU `2.10.0` |
| CANN/driver | CANN `9.0.0`, driver `25.3.rc1` |
| dtype | FP32; resolution reason `explicit:npu` |
| source | `781141882259ad66269dc87533bd82a893fb69a5` |

The stack printed shared-installation owner warnings and the documented
common-user warning. They did not indicate fallback: real NPU operations,
explicit synchronization and CPU comparisons all completed.

## Passing scope

All ten records in the manifest have `status: "passed"`:

| Area | Cases |
| --- | --- |
| graph forward/parity | PDG streaming; TimedDAG frontier and diamond; generic Settle; layered Settle; Settle chain |
| first-order differentiation | PDG backward; isolated Linear VJP; isolated Aggregate VJP |
| state handoff | AdamW optimizer step, CPU-portable checkpoint save, fresh model/optimizer load onto NPU |

The smoke compared the same CPU-created weights and inputs with the NPU result.
Forward graph values used `atol=4e-4, rtol=4e-3`; backward and isolated VJP
gradients used `atol=3e-3, rtol=3e-2`. Checkpoint inspection confirmed every
serialized tensor was on CPU, while the fresh continuation states and optimizer
state were deliberately moved to the requested NPU. This is portable handoff,
not an exact cross-device resume promise.

## Dtype and coverage limits

FP64 is not part of this NPU support cell. The local Ascend 910 TorchNPU stack
rejects the FP64 matmul used by the graph Full path (`DT_DOUBLE` is not in the
Cube support list), and `Model(..., device="npu", dtype=torch.float64)` rejects
that combination before execution. CPU FP64/FP32 remains the reference
contract. No throughput, distributed/HCCL, all-module, arbitrary-model,
optimized-kernel trace, or host-fallback claim follows from this smoke.

## C++ NPU boundary

The candidate installed by the TorchNPU Python wheel was inspected read-only:

```bash
python /home/zlong/.agents/skills/develop-portable-torch/scripts/inspect_libtorch_npu_sdk.py \
  /home/zlong/anaconda3/lib/python3.11/site-packages/torch_npu --format json
```

It is classified `python-wheel-runtime`, not a standalone SDK. The report found
no CMake package or public header, a dependency on `libtorch_python.so`, 201
undefined CPython API symbols, and no public `init_npu`, `finalize_npu` or
`synchronize` export. The library SHA256 is
`c0df99ef7584c76c38b05d2e76b070424318d6c8f4e623712211d6fa071efb56`; the only
init header shipped by the wheel has SHA256
`f4d646f3121099ee4cc95b796d36803ea31eb760ce36e568beb10f0bbd78558a`.
Those facts block a reproducible C++ LibTorch target. The existing C++ binary
therefore remains CPU-only and the C++ NPU matrix cell remains unsupported until
a version-matched standalone SDK is built, linked and exercised on a real NPU.
