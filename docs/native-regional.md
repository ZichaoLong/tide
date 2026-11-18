# Standalone regional scoring experiment

[`native/regional`](../native/regional) implements one complete regional scoring
Attention+FFN subpath in C++17/LibTorch. It is an optional experiment, not a second
complete Tide implementation or a replacement for `src/tide`. Reception, memory,
allocation, graph timing and model continuation are outside this executable.
Its checkpoint resumes the shared regional parameters, AdamW moments and update
counter with the caller's fixed input bundle.

The [fixed CPU record](evidence/native-regional-cpu-20261118.md) covers its finite
two-precision, shared-gradient, worker-concurrency and restored-update checks;
device and performance qualification remain separate. The
[CPU/NPU performance and stream record](evidence/native-regional-performance-20261118.md)
now supplies those finite observations, including retained BF16 failures, noisy
samples and CPU inner-thread controls; it does not establish full-model adoption.

## Math and precision

Inputs are `z[tasks,nodes,B,L,d/2]`, scalar scores, nonnegative key priors and fixed
loss targets. All tasks share the same nine parameter tensors. Q/K/V/O projections,
weighted Attention, RMS normalization, first FFN projection and SiLU, and the
contracted second FFN/readout match `RegionNetwork(backend="math")`. Description
width is a multiple of 32. `reference.py` provides the same Python operator layout;
tests additionally use a literal quotient formula and finite right differences.

Positive and explicit zero priors use the same representation. A visible zero key
in a nonempty row retains its right derivative; an entirely empty row has the
declared zero VJP. The custom weighted softmax supports first-order derivatives.
Masking the final correction for zero-prior queries does not remove those keys'
contribution to the other queries' prior gradient.

Parameters, inputs, parameter gradients, master weights and AdamW moments are
FP32. `--dtype bfloat16` enables BF16 projection/FFN arithmetic with autocast cache
disabled. Q/K/V Attention products, normalization and the final scalar contractions
are FP32. The first FFN bias is part of `linear`; it is not a separate FP32 addition
to a BF16 matrix result. Native guards verify Q/K/V and first-layer result dtypes;
the bias=1.001, zero-weight witness checks the BF16 rounding boundary. Output dtype
and guards do not certify a hardware accumulator or absence of device fallback.

## Build and runtime

CPU and NPU are separate link artifacts. CPU needs only a matched standalone
LibTorch CMake prefix and can run without CANN. The site SDK command below selects
the documented independent Torch 2.12/CANN 9.2/ABI1 stack; public prefixes remain
read-only. Do not mix its backend library with the Python wheel's library.

```bash
source /usr/share/Modules/init/bash
module purge
module load libtorch-npu/2.12.0-cann9.2.0
cmake -S native/regional -B build/regional-cpu \
  -DCMAKE_BUILD_TYPE=Release -DTIDE_NATIVE_BACKEND=CPU
cmake --build build/regional-cpu --parallel 2
ctest --test-dir build/regional-cpu --output-on-failure
cmake -S native/regional -B build/regional-npu \
  -DCMAKE_BUILD_TYPE=Release -DTIDE_NATIVE_BACKEND=NPU
cmake --build build/regional-npu --parallel 2
```

Freeze the implementation before qualification and retain source/library hashes,
the site build manifest and SDK identity. CMake records source revision/dirty state,
compiler, architecture, ABI and selected package prefixes. Each result records
loaded Torch/backend library paths and rejects an accidentally loaded Python
runtime library. Preserve the build log and final executable's `ldd` closure;
link success and loader success are distinct from device execution.

In a separate Python environment, create deterministic CPU fixtures:

```bash
python native/regional/reference.py fixture --output-dir artifacts/region-fixture \
  --width 64 --nodes 16 --batch 2 --positions 32 --tasks 4
```

The versioned tensor format specifies FP32, little endian, rank/shape, complete byte
count and FNV-1a-64 corruption checks. The producer also emits SHA-256 identities.
This is documented data exchange, not a native-memory ABI or a dependency on Python
at runtime. The C++ process independently reads, validates and runs a saved bundle:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  build/regional-cpu/tide-regional --device cpu --dtype bfloat16 --mode train \
  --input-dir artifacts/region-fixture --output-dir artifacts/region-cpp-w2 \
  --workers 2 --inner-threads 1 --warmup 2 --repeats 5
```

`--device` is required; explicit `auto` deliberately selects CPU. Unsupported
devices, invalid arguments, corrupt bundles and existing output paths fail.
`--resume PATH/checkpoint.pt` restores training before performing actual updates;
the output's `progress_before` and `progress_after` expose the transition.
New artifacts use fsynced staging and atomic replacement. A failed executable
returns nonzero; the supervising job must retain its log and terminal state.

## Tasks, streams and timing

`--workers 1|2|4` chooses serial execution or a persistent outer task pool. The
fixture, task count, shared parameter identity and total objective stay fixed.
Every task restores caller TLS and installs grad mode, autocast policy, device
and captured stream guards. All task losses perform backward against the same
owner; AdamW runs once after they finish. This measures concurrent regional tasks,
not concurrent independent optimizers. CPU budgets must coordinate outer workers,
`--inner-threads`, OpenMP and inter-op threads; inter-op is explicitly one.

NPU defaults to the same captured stream for every task. Establish serial device
execution first. Explicit `--streams pool` is a separate finite multistream probe;
it waits for all device work before the shared optimizer and retains any failure.
It does not establish safe integration into the full Tide graph or a speedup.
Device allocation and physical visibility belong to the project launcher and
[execution contract](execution-flows.md); the executable uses logical indices.

Inference, grad-enabled forward and complete regional training have separate
modes. Training samples include zeroing gradients, all forward/backward tasks,
AdamW and final synchronization. Model/fixture loading, pool creation, warmup and
serialization are excluded. Warmup uses a discarded owner; measured training
samples then continue parameters and AdamW with fixed inputs. The first sample
includes the measured owner's first slot creation. Do not label them IID timings.
Results retain all samples, shape, parameter count, threads, affinity and stream IDs.
Process peak RSS covers the process lifetime, including serialization; it is not
device allocated memory or a process-tree sample. Formal device memory/profiling
and contention identity must come from the admitted experiment.

## Verification and limits

```bash
# Python environment, CPU binary linked only to standalone CPU LibTorch:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/check_native_region.py --binary build/regional-cpu/tide-regional \
  --device cpu --output-dir artifacts/region-native-check
python -m pytest -q tests/test_native_region.py
```

The native checker executes two precisions, three modes, 1/2/4 workers, three
training updates, new-process restore and a real fourth update, shared parameter
gradients/moments, zero/empty priors, bias rounding and malformed-input rejection.
It preserves every comparison at `rtol=2e-5, atol=2e-6`. Python tests check the active
region implementation, independent FP64 quotient/VJP and zero-prior right derivative.
These diagnostic timings are not formal performance measurements.

For an admitted NPU check, pass `--device npu:0` and an executable
`--native-launcher` that purges modules, activates the standalone SDK and invokes
its arguments with `exec "$@"`. The Python reference keeps its separately activated
Python environment. Both consume the same fixture. CPU/NPU numerical comparison,
critical-operator placement, serial/pool-stream validity and performance remain
separate checks. The result explicitly says `device_operator_placement=not_profiled`
until an external trace supplies that evidence. CUDA has no native build adapter
in this finite slice; a CPU pass or NPU link is not CUDA or NPU qualification.
