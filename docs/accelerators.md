# Accelerator library builds and qualification

The same `GraphConfig → GraphRuntime → Session` contract accepts explicit
`cpu`, `cuda:N` and `npu:N`. CPU FP32/FP64 remains the independent reference.
The initial single-device eager FP32 path now also has explicit multi-device
consumer placement and FP16 policies; see [continuous consumers](online-consumers.md)
and [precision](precision.md). The optional [resident NPU backend](resident-training.md)
performs online scheduling and complete sharded training with a separate C++/CANN
implementation. Its current [integration gate](evidence/integrated-npu-consumers-20261004.md)
is for CANN9.0.0; older eager version checks do not qualify resident execution.
Python-owned native resident execution is distinct from independent PyTorch eager
scheduling. Full-size comparisons remain tracked in [STATUS](STATUS.md).
CUDA FP64 requires its own target-machine gate. NPU FP64 and the explicitly FP64 `norm-fp64-v1` Read
profile fail before execution; they are never silently reduced to FP32.
NPU's optional `fiber_pooling=csr` is unsupported by the tested TorchNPU sparse
matrix multiply backend. The public adapter rejects it; explicitly select the
default `event` policy. Standalone C++ callers must also avoid that CSR policy.
Experiments still own data, task heads, losses and training orchestration.

Use separate Python environments and build directories for CPU, CUDA and NPU,
and rebuild for every host architecture/framework/ABI combination. CUDA depends
on the device compute capability, Torch's compiled architectures, driver/runtime
compatibility and the selected CUDA distribution. A successful CPU run of a
CUDA-linked binary certifies its build/loader, not GPU execution. `auto` is a
convenience for applications; qualification requires an explicit device.

## Python and native adapters

Load one compatible site stack before these commands. Package installation and
module selection belong to the site, not the library. CPU import/help does not
load TorchNPU (`TORCH_DEVICE_BACKEND_AUTOLOAD=0` also disables Torch's automatic
third-party plugin loading).

```sh
python scripts/build.py --backend cpu --build-dir build/cpu --jobs 2
python scripts/build.py --backend cuda --build-dir build/cuda --jobs 2
python scripts/build.py --backend npu --npu-runtime python --build-dir build/npu-python --jobs 2
```

The installed-consumer verifier explicitly preserves the selected Torch when
creating a nested virtual environment. Python's `--system-site-packages` alone
inherits the base interpreter's environment; confirm `torch.__version__` and
`torch.__file__` when constructing an experiment environment inside another venv.

Each command is run in its matching environment. CUDA CMake discovery may also
need a matching CUDA toolkit; on a host without a GPU set a target-specific
`TORCH_CUDA_ARCH_LIST` supported by that toolkit. Tide uses standard ATen and
ships no custom CUDA kernel. That setting does not add missing architectures to
the selected prebuilt Torch wheel.

Native model, input, continuation and gradient tensors must agree on device and
dtype. Worker threads inherit the caller's stream and autograd state. The Python
adapter uses TorchNPU registration owned by Python and does not link a second
vendor runtime. Its manifest checks architecture, Torch, TorchNPU, Python, ABI,
backend and binary hash. A CPU build cannot execute accelerator tensors.

## Independent C++ NPU

Use a version-matched **standalone** `libtorch_npu` SDK built from the official
TorchNPU source. A wheel's `libtorch_npu.so` is not a substitute. The standalone
SDK and Python wheel can contain incompatible libraries with the same SONAME;
keep them in separate build and launch environments.

```sh
cmake -S . -B build/npu-sdk -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/libtorch-npu-sdk" \
  -DTIDE_BACKEND=NPU -DTIDE_NPU_RUNTIME=standalone -DTIDE_PYTHON_BINDINGS=OFF
cmake --build build/npu-sdk --parallel 2
ACL_OP_INIT_MODE=0 build/npu-sdk/tidegraph-accelerator-check --device npu:0 --dtype float32 \
  --output-dir artifacts/cpp-npu-001
```

Alternatively `scripts/build.py --backend npu --npu-runtime standalone` uses
the selected interpreter only to discover its matching Torch and orchestrate
CMake. The resulting executable does not link CPython or `libtorch_python`;
the CANN operator compiler may dynamically initialize its own Python runtime.
For the tested standalone TorchNPU2.10/CANN9.0 stack, set `ACL_OP_INIT_MODE=0`
before starting the process. Eager compiler initialization avoids a reproduced
shutdown wait in embedded Python when lazy initialization first occurs on an
autograd worker. It does not disable operators or skip finalization. Keep this
launch setting in environment records; other vendor versions need their own
lifecycle check. The retained failure and same-binary successful full gate are
in [packed-transfer evidence](evidence/eager-packed-transfer-20261003.md). CMake's
`TideGraph::NpuSDK` adapter consumes the SDK's public headers and library variables;
`tide::runtime` performs public initialization, seeding and synchronization.
The exported `tide::tidegraph` core contains device-neutral graph execution.
Standalone applications using the resolver link `tide::runtime`.

`tidegraph-accelerator-check` compares CPU Streaming with independent native
ring/diamond schedules: complete observables, shared owners, gradients, three
AdamW updates and same-device/CPU checkpoint loading. Accelerator runs also
assert nondefault-stream inheritance in workers. `TIDENCK1` still stores named
weights/optimizer slots; it does not acquire graph continuation serialization.
The Python session checkpoint continues to store complete graph continuation.
Cross-vendor portable handoff is numerical equivalence, not bitwise RNG resume.

## Target-machine acceptance

First run the full CPU baseline using a matching CPU build. Then use a new
output directory for each requested backend, implementation, dtype and stack:

```sh
python scripts/qualify_library.py --build-dir build/cpu --reuse-build --output-dir artifacts/cpu-001
python scripts/qualify_accelerator.py --device cuda:0 --implementation python --output-dir artifacts/cuda-python-001
python scripts/qualify_accelerator.py --device cuda:0 --implementation native \
  --native-library build/cuda --output-dir artifacts/cuda-native-001
build/cuda/tidegraph-accelerator-check --device cuda:0 --dtype float32 --output-dir artifacts/cuda-cpp-001
python scripts/check_native_device.py --device cuda:0 --native-library build/cuda --output-dir artifacts/cuda-owners-001
python scripts/profile_accelerator.py --device cuda:0 --implementation native \
  --native-library build/cuda --case worker-streams --output-dir artifacts/cuda-profile-001
```

For NPU replace the device with `npu:0`, use `build/npu-python` for Python native
commands and the separate `build/npu-sdk` for standalone C++. Physical device
visibility is a launcher choice; programs use remapped logical indices.
The native owners check separately covers SGD momentum and AdamW, aliases,
disconnected/zero gradients, optimizer slot placement and restored updates.

The named suite contains 39 Python and 44 native cases, not a Cartesian product:
PDG/TimedDAG/Settle, four specialized schedules, mixed state modules, same-fiber
memory/pooling, LH Full activation/norm profiles, five Aggregate profiles,
selectors, sparse emission/reset, clocks and delayed parallel edges. Native adds
workers and four same-fiber policies. On NPU, 43 native cases execute and the
CSR case separately verifies explicit rejection. Each supported case compares full observables, exact
routes, None gradient connectivity, isolated VJPs, chunks, trace-disabled behavior,
three optimizer steps, fresh-process resume and CPU checkpoint handoff. Fixtures
start on CPU and are saved in the result directory. The clock fixture declares
valid even phases and an eight-tick cut. Default FP32 tolerances are
`atol=1e-6, rtol=1e-5`; overrides are recorded, never silently enlarged.

`--case NAME` is a development subset. Neither a finite suite nor a profile
certifies arbitrary graphs, scales, performance or absence of every possible
fallback. Candidate-only profiles require real accelerator kernel events and
retain host transfer/scalar events. Graph metadata and discrete decisions may
run on the host; comparisons and checkpoint copies are separate from profiling.
Review operator-placement traces and launch diagnostics for the exact stack.

On the tested TorchNPU2.10/CANN9.0 resident path,original-width FP16 construction
exposed a cached `TensorMove` function-handle lookup failure. Explicit process-local
`ACLNN_CACHE_LIMIT=0` completed the same reduced-batch reproduction and passed
[93 affected resident correctness checks](evidence/resident-cann-cache-policy-20261004.md),
including independent CPU oracles and full training/continuation. Record this
setting and preserve it in paired profiling. Shared modules and graph semantics
are unchanged;the default-cache failures remain retained. Under this explicit
policy,the original-B512 [Add](evidence/formal-b512-resident-fp16-add-20261004.md)
and [Attention](evidence/formal-b512-resident-fp16-attention-20261004.md) inference
companions passed on8 NPUs,one process each. Complete-training companions also
passed:Add on8cards,Attention on11cards;separate inference slices are available.
Automatic repetitions are cancelled under the current user contract. The
[selection review](evidence/selection-review-20261009.md) separates dtype/card/cache
effects,records current profiling and retains all failures. Other runtime versions
still need their own evidence.

Installed dependency checks:

```sh
python scripts/library_consumer.py --device npu:0 --skip-cpp \
  --build-dir build/npu-python --output-dir artifacts/installed-npu-python-001
python scripts/check_cpp_consumer.py --device npu:0 \
  --build-dir build/npu-sdk --output-dir artifacts/installed-npu-cpp-001
```

The latter needs a `scripts/build.py` build manifest, verifies loader closure
without Python/stub libraries, and builds the copied application using only
installed headers/targets. CUDA supports the same consumer commands.
`scripts/export_foundation.py --output-dir NEW` exports clean source, hashes and
these recipes for another machine. Rebuild there and retain its manifests and
suite results; x86_64 evidence is separate from aarch64. Current exact support
claims live in `.torch-portability/contract.json` and the cited evidence.

Portable two-device eager owner/copy/actual-consumer and automatic-capacity
commands are in [eager target validation](eager-target-validation.md). Local NPU
evidence remains distinct from CUDA target-machine-pending status.


Additional [standalone TorchNPU2.9 qualification](evidence/standalone-sdk29-20260928.md)
passed on local CANN8.5.0/8.5.1/8.5.2 using one matched SDK/Torch build. It covers
standalone core ring/diamond training/checkpoint, installed CMake consumption,
and the separately scoped historical-topology consumer. The earlier broad
Python/native-adapter matrix remains distinct from these standalone gates.
