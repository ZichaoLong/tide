# State kernels, selective SSM and SwiGLU, 2026-09-21

Clean source: `50b5e943b4ba183d66d78a30a4e94468c6ed8349`.
Command: `python scripts/qualify.py --output-dir artifacts/m5a-20260921-0900`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release, two build workers, ATen/BLAS threads 1, node workers up to 3.
Build and tests exited 0: **481 passed**, pytest 38.95 seconds.
Retain `artifacts/m5a-20260921-0900/{status.json,task.log,verification/}`.
The manifest records source fingerprint and hashes of all three native binaries.

New coverage in FP64/FP32: named state slots in trace/reset/checkpoint/detach;
diagonal selective SSM with both tanh and SwiGLU Full; EMA with SwiGLU;
Python/native streaming, Python/native frontier and native chain specialization;
all Emit modes; selected clear; input/parameter/initial-state VJPs. Cyclic SSM
streaming checks serial, packed batch and three-worker execution. Independent
SSM recurrence and SwiGLU derivative formulas anchor the local operators.

Checkpoint v3 includes every tensor slot and preserves the prior alias-topology
guard. A standalone C++ client supplies a custom StateKernel absent from the
built-in registry, executes it through packed streaming, and checks its state,
continuation and analytic input VJP. Existing 323 tests remain passing.

SSM is the explicit representative recurrence in `../state-programs.md`, not a
pretrained Mamba-family import. Attention/Linear Attention/DeltaRule remain
pending. SSM+SwiGLU through SettleGraph embedding needs a separate profile test;
current SettleGraph embedding evidence is the earlier EMA profile. Region
selection and source-weighted sum Agg remain fixed declared profiles. No speedup
or quality claim follows from this qualification.
