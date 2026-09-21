# Complete content and state extensions, 2026-09-21

Clean source: `f76a90cb752b0feed0e7aeec44b496664e9f1264`.
Command: `python scripts/qualify.py --output-dir artifacts/content-20260921-1250 --jobs 2`.
Environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, GCC 10.3.1,
C++17 Release; two build workers, ATen/BLAS threads 1, node workers up to 3.
Build and tests exited 0: **1758 passed**, pytest 153.52 seconds.
Retain `artifacts/content-20260921-1250/{status.json,task.log,verification/}`.
Both manifests identify clean source; verification records native source/binary
hashes. Unit `tide-foundation-content-20260921-1250` is inactive, MainPID 0.

Complete content carries canonical program-visible source atoms, local slots,
physical scales and Aggregate contributions through state/Read/Full and packed
state metadata. Raw routing fibers are separate. The native view borrows only
for synchronous calls; Python state programs own registered parameters and use
normal sharing/optimizer/checkpoint machinery. Adapter guards reject custom
Python state overrides, mismatched Linear/Delta sharing and Attention policies.

A source-aware custom state, proposal-based Read and custom Full exercise tags,
positions and contributions. With no clear the hand-computed loss is 116,
per-element input VJPs are 13 and 15, and gain VJP is 48. With selected clear the
loss is 104, input VJPs 7 and 15, and gain VJP 42. An independent sample remains
disconnected. Native scalar, packed parallel streaming and frontier agree under
direct external and source-origin boundary encodings. Scalar state batch and
sequence fallbacks are counted. Python additionally checks direct/encoded
SettleGraph, custom parameter registration, AdamW and checkpoint restoration.
All earlier memory, attention, Aggregate/Emit, inference, isolated public-root,
sharing, cursor and fixed-topology checks pass in FP64/FP32.

Development build `artifacts/content-build-20260921-1240/` failed on an old packed
test fixture's fiber-pointer initializer. The ContentView fixture correction
built successfully in `artifacts/content-build-20260921-1248/`; 619 targeted
checks passed before the implementation commit. Both artifacts are retained.

Read still belongs to the state-program interface at this revision. Independent
Read modes and full Next requests are subsequent gates. This qualification does
not establish original-LH parity, arbitrary pretrained models, higher-order AD
or large-workload performance. Local semantic replay remains the first-order
training baseline; inference does not replay.
