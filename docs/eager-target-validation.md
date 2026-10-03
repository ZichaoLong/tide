# Eager target-machine qualification

CUDA hardware is unavailable on the current NPU server. The commands below are
portable build/validation recipes. A fresh aarch64 CUDA-linked build and187 CPU
checks are [qualified](evidence/eager-cuda-host-20261003.md); no GPU execution is
qualified by that host-only evidence.
Run on a clean exact commit using a matching CUDA-enabled PyTorch/LibTorch,
compiler, driver and toolkit. Keep the independent CPU FP64/FP32 reference in
that environment and retain the target's source/build/test manifests. Host
architecture and vendor evidence do not transfer automatically.

The eager owner/copy/consumer gates accept an explicitly selected CUDA or NPU
backend. An unavailable explicit target fails. The test selection below excludes
CPU-only owner planning from the actual-device gate; it does not silently skip
missing hardware. Two logical accelerators are required. Physical device
selection and any resource lease belong to the site launcher.

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python scripts/build.py --backend cuda --build-dir build/cuda --jobs 2
cmake --install build/cuda --prefix "$PWD/build/installed-cuda"
cmake -S tools/online_bench -B build/online-cuda \
  -DCMAKE_PREFIX_PATH="$PWD/build/installed-cuda" -DBUILD_TESTING=ON \
  -DTorch_DIR="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "share/cmake/Torch")')"
cmake --build build/online-cuda --parallel 2

TIDE_BUILD_DIR="$PWD/build/cuda" \
TIDE_ONLINE_BINARY="$PWD/build/online-cuda/tidegraph-online-bench" \
python -m pytest -q tests/test_eager_capacity.py tests/test_consumer_traffic_bounds.py

TIDE_BUILD_DIR="$PWD/build/cuda" \
TIDE_TRANSFER_BACKEND=cuda TIDE_PAYLOAD_BACKEND=cuda \
python -m pytest -q tests/test_transfer.py tests/test_payload_ownership.py --dtype float32

TIDE_BUILD_DIR="$PWD/build/cuda" TIDE_ONLINE_DEVICE=cuda:0 \
TIDE_ONLINE_BINARY="$PWD/build/online-cuda/tidegraph-online-bench" \
python -m pytest -q tests/test_online_eager_owners.py --dtype float32 \
  -k 'per_owner_constants or actual_two_device or unified_cli_owner'

TIDE_BUILD_DIR="$PWD/build/cuda" TIDE_EAGER_CAPACITY_DEVICE=cuda:0 \
TIDE_ONLINE_BINARY="$PWD/build/online-cuda/tidegraph-online-bench" \
python -m pytest -q tests/test_eager_capacity.py --dtype float32 \
  -k forced_automatic_chunks

build/cuda/tidegraph-payload-ownership-check --device=cuda:0 --dtype=float32

TIDE_BUILD_DIR="$PWD/build/cuda" TIDE_EAGER_PRECISION_DEVICE=cuda:0 \
TIDE_ONLINE_BINARY="$PWD/build/online-cuda/tidegraph-online-bench" \
python -m pytest -q tests/test_online_eager_precision.py --dtype float32
```

Use a detached bounded job for long builds/gates, inspect all test counts and
terminal exit codes, and verify the installed executable's loader closure.
The static capacity test adapter allocates no model and does not certify a
multi-device runtime by itself. Its default build is controlled by `BUILD_TESTING`.

The equivalent NPU commands use the separate matching Python and standalone SDK
builds documented in [accelerators](accelerators.md), with `npu` backend values
and `npu:0`. CPU correctness runs use the CPU core; native accelerator tensors
must use the corresponding accelerator core. The owner gates cover independent
CPU full records, two complete updates, connected windows, physical sample
splitting, missing/zero gradients and actual destination devices.
For the tested standalone TorchNPU2.10/CANN9.0 stack, export
`ACL_OP_INIT_MODE=0` before these NPU commands; see the
[compiler lifecycle requirement](accelerators.md). This initializes the vendor
compiler before autograd workers use it and preserves normal runtime cleanup.

Before scale use, calibrate [eager memory admission](eager-consumer-capacity.md)
on that target with bounded fresh processes. Inspect actual device operators and
copies in a separate profiler run, then perform formal synchronized timing
without reference/profiler work. The FP16 gate covers explicit payload gradients,
FP32 masters/slots and static loss scaling; its CUDA execution remains pending.
These recipes alone do not qualify resident CUDA, target-version combinations
or full-size throughput.
