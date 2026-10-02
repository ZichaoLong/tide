# Shared reverse gather inputs

Implementation `1757b905943bc2cb3485be7c02530adf678d3dab` passed nine clean jobs.
[Audited records](resident-reverse-gathers-20261002.json) pin source, affected
archive members, core/CANN dependencies, binaries, loader closure, tests,
allocator records and profiler CSVs. Audit: `TASK/launchers/reverse_gather_evidence.py`.

Owner packs now share one padded read-only input per reverse program phase:
Full values/gradients, global event/fiber journals and scales, and state-stage
cotangents/connection flags. Each replay refreshes the copy from that execution's
source; no numerical results are cached across windows or programs. Sources stay
unchanged until all readers finish. Per-owner indices, outputs and result flags
remain independent. Allocation follows the first owner's existing preflight;
program-owned tensor handles protect lifetime after the helper is destroyed.

Native FP32/FP16 ordinary and compact-journal training passed128 trajectories,
2,048 windows and512 updates against independent CPU FP32/FP64 references.
Explicit two-to-three-card restore passed another32 trajectories,512 windows and
128 updates. The new direct check covers repeated indices, zero sentinel rows,
bool/FP32/FP16 inputs, changed source values over three replays, independent
outputs, exactly one padded allocation, foreign-program refusal and close cleanup.
Python retention16 and complete-consumer24 tests passed without skips; the latter
cover three graph families and both Python-native and installed LibTorch clients.

A same-two-card-lease old/new calibration used D512/B8/T4/V257 Attention,
128 body nodes, four physical B2 groups, FP32 AdamW and two connected windows
per update. The old consumer is clean3462dae; all consumer source bytes,
workload, requested/effective chunks and memory estimates are identical.

| Logical card | Before peak bytes | After peak bytes | Reduction |
| --- | ---: | ---: | ---: |
| Coordinator0 | 8,396,636,672 | 8,368,268,800 | 28,367,872 bytes / 27.054 MiB |
| Owner1 | 7,520,954,880 | 7,520,954,880 | 0 |

Loss was exactly7.532631874084473 in both processes. All work and retained-storage
counters matched. No estimator, physical budget or safety margin was reduced.
This establishes allocation savings at this fixture, not an original-size
training pass or a speed claim.

A separate two-card FP16 two-update profile recorded53,190 operators:
50,642 AI_VECTOR_CORE,1,825 AI_CORE and723 MIX_AIV, with no observed AI_CPU rows.
The profile includes construction/cleanup and is excluded from throughput claims.
Public ABI, portable core, device kernels and private training layout are unchanged;
five affected host objects and the standalone checker were rebuilt, with verified
unchanged dependencies reused. Python-native and standalone runtimes remain separate.

One development build failed at module activation before its job writer because
an unavailable module name was requested. That failure remains recorded; corrected
and clean builds use the established LibTorch-NPU2.10/CANN9.0 stack. No device
correctness failure is being relabelled. Original-width Add revalidation follows
this qualification; original B512 complete training and full-size comparisons
remain open.
