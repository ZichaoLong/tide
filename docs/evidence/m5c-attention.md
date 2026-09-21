# Event attention and packed state prefill, 2026-09-21

Clean source: `652f2e7a7a09f4dcf6b220cbc058c580cc10c41d`.
Command: `python scripts/qualify.py --output-dir artifacts/m5c-20260921-0939`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release, two build workers, ATen/BLAS threads 1, node workers up to 3.
Build and tests exited 0: **799 passed**, pytest 62.59 seconds.
Retain `artifacts/m5c-20260921-0939/{status.json,task.log,verification/}`.
The manifest records source fingerprint and hashes of all three native binaries.

New FP64/FP32 coverage: event GQA/MQA and multi-head attention, bounded/unbounded
KV caches, observe-all/selected-only/clear, all Emit modes, full trace and
input/parameter/initial-KV VJPs. Python frontier and native streaming/frontier
compare with an independent head-by-head, event-by-event reference. Cyclic and
DAG chunk continuation, serial/unpacked versus parallel/packed, explicit detach,
checkpoint, shared attention parameters and two AdamW updates also pass.

Direct SettleGraph matches its Python/native TimedDAG embedding; independent
Python/native chain/self-loop schedules and a Python SettleGraph chain anchor
are covered. Analytic uniform-attention VJPs and explicit GQA head assignment
anchor local formulas. Invalid cache/config/packed metadata is rejected.

Execution-shape checks establish real joint batching: three length-six samples
form one `[3,2,6,6]` score group (216 elements); lengths 6/4/6 form two groups
with 176 total elements. Idle samples produce no state. Native unpacked state
prefill uses three sequence calls. Final inference caches/read vectors own only
their compact storage. These are shape/allocation checks, not speed benchmarks.

The standalone external C++ StateKernel client additionally checks default
packed fallback and metadata validation. Existing 619 tests remain passing.
See `../attention.md` for exact event/window/clear/position contracts. LH
same-fiber attention, positional embeddings, paged/ring caches, structured Delta
chunks and workload performance are unqualified. No arbitrary model import claim.
