# Final CPU qualification of the graph execution foundation

Clean tested source: `81a1b266af49d918aa6e1587e4ed9e0c4d4e5eb5` on graph-execution-foundation.
**7741 tests passed in1235.63s**, CPU FP64/FP32; all17 relocated smoke variants
and36 fresh-process training manifests/payloads passed. This closes the six-stage acceptance version together with the reviewed
[medium](foundation-medium.md) and [large](foundation-large.md) assessments.
All mandatory implementation/correctness cells passed; bounded scale failures
remain failures. No task service or child process remains live.

## Rebuild, execution and identity

Read-only checkout: `qualification/foundation-final-81a1b26`.
Raw project evidence: `artifacts/foundation-final-20260923-a/`:
status.json, pipeline.json, reviewed-audit.json, full-cpu/result.json/tests.log,
retained test-tmp/, relocated-smoke/, standalone-build/ and per-stage logs.
Exact detached command/resources: `artifacts/foundation-final-a-launch.json`.
The retained driver `artifacts/qualify_foundation_final.py` and auditor
`artifacts/review_foundation_final.py` have hashes in the records.

Source was exported without Git metadata to `relocated source with spaces/`.
A fresh102-step Release build and all17 real smoke variants completed in574.46s;
no old development build was used. The full CPU gate used that newly built
adapter. A separate55-step build with TIDE_PYTHON_BINDINGS=OFF took219.36s;
standalone FP32 and FP64 runs and loader checks all exited0. It independently
constructs/encodes Settle, checks a literal recurrence, initial-state/input/shared
parameter VJPs, unused Read, real sequence prefill, continuation, aliases and
malformed-coordinate rejection. ldd resolves every library and contains neither
libpython nor libtorch_python.

The terminal audit compared all489 tracked files with Git archive, exported
hashes and the frozen checkout; all13 primary binaries match their manifest.
The separate standalone binary and all36 fresh-process output/checkpoint hashes
also match. Source and binary identity are therefore checked beyond a commit
label. Primary native adapter SHA256:
`b92c0fa2517704d4e54a856dcdb15f133108a6515ed2b3b70819aacec2618f4c`.
Standalone no-Python binary SHA256:
`88e69f281cb11a422c2d4b8e3aab23b236a267e5eca57a6cf3c082fb412a7cb3`.

Environment: aarch64, Torch/LibTorch2.10.0+cpu, Python3.11.15, GCC10.3.1,
C++11 ABI. TORCH_DEVICE_BACKEND_AUTOLOAD=0; correctness ATen/OpenMP/BLAS1,
build jobs2. Actual process tree/cgroup/maps recorded at build/full CPU/standalone
stages. Dynamic launch half-budget160 CPUs/726.80GiB,26.02GiB disk available;
observed build/full-test RSS samples about1–1.7GiB, not claimed peak measurements.
No qualification build/test overlapped formal performance timing.

Unit tide-foundation-final-20260923-a ran in background.slice/Nice10 with
RuntimeMaxSec7200. Eight pipeline stages exited0; terminal unit inactive/dead,
MainPID0/ExecMainStatus0 and empty process group. No skipped or failed tests.
The standalone core remains usable without the Python testing/client bindings.

## Six implementation classes

All cells below passed on CPU FP32/FP64 for their declared finite profiles.
Local kernels can be shared; topology specializations own independent schedules.
Formula tests provide additional anchors separate from the shared kernels.

| Class | Entry / independent schedule | Main validation |
| --- | --- | --- |
| PDG generic | Python reference.py; C++ stream.cpp/cursor.cpp | positive-delay feedback, CSR/CSC, sparse state owners, ragged input/parallel edges, batch, packed/unpacked and node workers; complete continuation/isolated roots |
| PDG specialized | Python specialized.py; C++ specialized.cpp, self-loop/ring | independent propagation order; trace/state/history/messages/ledger and first-order VJP/cuts against generic execution |
| TimedDAG generic | Python frontier.py; C++ planner/frontier/block plus streaming inclusion | complete fibers, legal state/Full time batches, sequence/fallback counters, node workers and optimization interactions |
| TimedDAG specialized | independent Python/C++ chain/diamond, including shared middle region | independent dependency layers, competition/history, step/block and isolated training roots |
| Settle generic | independent Python settle.py; C++ settle.h/settle.cpp/settle_projection.cpp | native construction/encoding, model owner map, dense window entry, standalone formula/VJP; generic/encoded streaming/frontier parity |
| Settle specialized | Python settle_chain/settle_layered | scalar layer/time/batch loops and output grouping; no generic executor/planner call; encoded/generic values and isolated VJPs |

[Encoding contract](../settle-embedding.md) limits Settle to the adopted
rank-aligned, broadcast-input/summed-output profile. Body IDs, regions, parallel
edges, local ports and source domains are preserved through explicit boundary
maps. Settle event clock is stride*position+rank; TimedDAG-to-PDG inclusion changes
no clocks or identities. Projection restores state/slots/history/messages/ledger
and training roots at complete position cuts; partial-cut pending remains in the
encoded continuation. Body projection is not a general inverse/checkpoint schema.
No proof of arbitrary graph/module equivalence is inferred from finite instances.

## Modules, options and training

The [capability table](../execution-capabilities.md) records graph × schedule ×
module × inference/training × option support and restrictions. Required finite
modules include event Attention/GQA/window, distinct same-fiber Attention,
Linear Attention, ungated delta-rule-v1, Gated Delta, SSM, FFN/SwiGLU/norm,
complete-source Aggregate and HARD/SOFTP/HST Emit. HST is checked against its
explicit surrogate VJP. Identity/route/mask/ledger comparisons are exact;
floating comparisons use the per-dtype contract tolerances.

Frontier/native Settle migration includes legal packed source transport,
batch Next/reset, compact publication, region workers/deferred cleanup and
same-fiber exact/single, event/CSR, KV ownership, event/head and projection
layouts. Nontrivial time batches are counted. Unsupported requests reject;
causal/custom-program fallbacks record reason/count/path. Present numerical zero
sources are retained. Cache/alias/snapshot lifetimes and parameter updates are
covered. Scalar semantic replay remains the packed training correctness baseline;
its separately measured cost does not establish optimized backward.

The small [model adapter](../model-adapter.md) composes explicit positions,
RMSNorm/RoPE/GQA/SwiGLU, mask and cache layout with independent formula/chunk/VJP
checks and invalidation on updates. It is not arbitrary pretrained-model import.

Six-class training tests cover independent output/state/slot/history/pending
roots, initial states, shared/unused owners, None vs connected zero, multiple
SGD/momentum/AdamW updates (epsilon1e-5, decoupled weight decay), detached chunks
and partial windows. The36 fresh processes compare uninterrupted vs
prefix-save/exit/new-process-load/suffix trajectories in both dtypes for SGD
with momentum and AdamW, over single-graph, two-clock application and native
named-value scopes. Owner/alias/group identity and later updates are checked.

Continuation v5, application bundle and native TIDENCK1 retain distinct schemas
and restore scopes. Transactional malformed-input rejection and publication
failures remain covered. Named model/optimizer values do not contain graph
continuation; none of these files promises the complete training controller,
data cursor or framework RNG. Python/native checkpoint file interoperability
is not inferred. Earlier failed/cancelled gates are preserved, not relabeled.

## Performance and boundary of the delivery

The frozen12 medium configurations have108 completed runs with three independent
repeats and all five implemented measurement phases; see
[medium evidence](foundation-medium.md). No grad-forward-only result is called
training throughput. Large limits, partial measurements and unlaunched stages
are separate in the [large report](foundation-large.md), including the corrected
historical narrow owner count. No new tuning candidate was selected; defaults
remain conservative. The original LH C++ exact mapping and comparable-scale
routes retain their separate contracts and existing evidence; LH does not define
Tide training semantics.

One-command target-local build/smoke and actual supported arguments are in
[benchmark entry](../foundation-benchmarks.md). Reproduce the full CPU gate with
scripts/verify.py --device cpu --dtype both --build-dir MATCHING_FRESH_BUILD
--output-dir NEW, using a clean frozen worktree. For the no-Python check, configure
CMake with TIDE_PYTHON_BINDINGS=OFF, build tidegraph-settle-check, and run both
--device=cpu --dtype=float32 and --dtype=float64. Exact argv is retained per stage.

Only the recorded aarch64 CPU FP32/FP64 target is qualified. CUDA, Ascend, x86
execution on another host, arbitrary imported models/LH configurations,
higher-order AD and entire-controller resume remain explicit extensions.
