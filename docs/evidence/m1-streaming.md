# CPU streaming qualification, 2026-09-21

Tested clean source: `9c957fdf90c946ee7263c323d6f5153f793f49e3`.
Host: aarch64; Python 3.11.15; PyTorch/LibTorch 2.10.0+cpu,
Torch revision `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`, C++11 ABI enabled.
Compiler: GNU 10.3.1, C++17, Release, two build workers; ATen/BLAS threads 1.

Commands (use the matching Python environment):

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/build.py --jobs 2
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/verify.py --device cpu --dtype both --output-dir artifacts/verify-m1
```

Build exit 0; qualification exit 0: **105 passed**, pytest duration 8.46 seconds.
Native module SHA256:
`c80f50cbd3b03e4b9cd1599773495abbe49f876fa146b296c75e7e0be995b8fb`.
Raw retained evidence: `artifacts/build-20260921-0800/{status.json,task.log}` and
`artifacts/verify-m1/{result.json,tests.log}`. Build service completed normally.

## Scope

- Independent time-major Python oracle vs sparse native event scheduling.
- FP64/FP32, HARD/HST/SOFTP, serial and three node workers, packed/unpacked batch.
- Complete event trace, final state/history, outputs and crossing messages.
- Isolated output/state/pending-root VJPs including parameter gradient presence.
- Empty windows and three declared chunk partitions; gradients cross in-memory cuts.
- AdamW update and value-checkpoint/optimizer round-trip; resumed forward values.
- Analytic delayed self-loop and explicit HST local VJP; seals, graph validation,
  parallel-edge IDs, CSR/CSC, no-grad propagation and source-state preservation.
- A 10,000-node shared-weight graph with a million-tick gap executes only two
  candidate events and stores two node states. This checks sparse scheduling
  counters, not throughput or peak memory performance.

## Limits

Only `ema-ffn-v1` is qualified. General module plug-ins, packed sequence state,
frontier algorithms, topology specializations and LH numerical parity are still
pending. CPU x86_64 is unverified. This suite is not a training-quality or
performance benchmark. CMake's optional missing static Kineto warning did not
prevent link or real forward/backward execution.
