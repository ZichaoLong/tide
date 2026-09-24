# Current handoff

Updated: 2026-09-24 (Asia/Shanghai). Branch graph-execution-foundation.
**Six-stage finite CPU foundation acceptance COMPLETE. Requested Add comparison COMPLETE.**
No task live jobs or descendants remain. ROADMAP is the sole backlog; semantics
is the current contract. Do not restart old milestones or repeat completed large
experiments without a new question. No push or reference-repository writes.

## Latest requested work: fixed-graph Add comparison

Delivery: [reviewed Add results and objective assessment](evidence/add-scale-comparison.md).
Contract: add-scale-comparison.md; portable CLI: ../tools/cpu_compare/README.md.
Implementation source:0c053ebc7f8b3be03220c54aebf56502f5f0f6f1.
Frozen clean source: /var/tmp/zlong-graph-execution-foundation/qualification/add-scale-20260924.
Raw records: artifacts/add-scale-comparison-20260924-a/ (status.json, pipeline.json,
reviewed-audit.json, analysis.json, per-run manifests/metrics/summaries, export/,
three source/build records, optional process-samples.json).
Exact launch/preflight: artifacts/add-scale-comparison-a-launch.json and
artifacts/add-scale-host-preflight-20260924.json. Retained run/review drivers are
hashed in the records. Portable archive: export/cpu-attention-compare.tar.gz
under that result directory; it now supports Add and explicit grad-forward.

All4 actual native smoke cases and12 wide cases completed/exit0, with3 independent
process repeats per engine/mode. Unit tide-add-scale-comparison-20260924-a is
inactive/dead, MainPID0/Result=success/ExecMainStatus0, with no remaining group PID.
Independent audit checks frozen source against Git archive, exported inventories,
all3 build/binary records,16 validated run records, counts, thread settings,
denominators, memory bounds and exact same-engine checksum sequences.

D2048/B512/V50304, FP32, nominal leaf1/32;9,468,020,899 parameters (8.818 divided
by1024³). Same graph/module scale, independent weights, no backward/optimizer.
Median of3 process-window means (12 steps, warmup4), ms/sample-token:

- LH nograd5.82417; PDG nograd5.59769. Repeat ranges overlap; no consistent win.
- LH grad-forward7.12832; PDG grad-forward108.23288 (15.18x). Scalar semantic
  replay is a priority performance target; Tide training semantics stay intact.

Actual56-core affinity160-215 from half-host budget160; dynamic half-memory752GiB.
Peak RSS across cases <=85.17GiB. LH actual ATen/OpenMP/OpenBLAS56 (interop reports
320, not320 active workers); PDG pools1 plus56 node/head workers in separate phases.
Trackio unavailable/degraded; all local records are complete and validated.

472 directed CPU FP64/FP32 tests passed/113.88s in artifacts/add-compare-dev-20260924-d;
unit inactive/MainPID0/exit0. Retain failures -a (bad pytest path), -b/-c (FP32-derived
FP64 norm descriptor incorrectly checked at FP64 tolerance). The scale comparator
now uses payload precision for derived descriptors; no execution formula or
exact route/identity check changed. No claim of a new7741-test global rerun.

## Completed foundation and its limits

Delivery: evidence/foundation-final.md. Capabilities: execution-capabilities.md.
Six implementation classes, independent schedules, representative modules,
native Settle construction/encoding, applicable optimization migration, isolated
first-order roots, optimizer/alias/None-versus-zero and scoped checkpoints passed.
Qualified source81a1b266af49d918aa6e1587e4ed9e0c4d4e5eb5; clean checkout
qualification/foundation-final-81a1b26. Evidence artifacts/foundation-final-20260923-a:
7741 CPU FP64/FP32 tests/1235.63s,17 relocated smoke variants,36 fresh-process
checkpoint trajectories,489 source files/13 binaries and no-Python native Settle.

Performance evidence: foundation-medium.md (12 configs/108 completed runs) and
foundation-large.md under docs/evidence. Large assessments retain7 complete,
5 timeouts and1 unlaunched stage; these are not target-scale successes. One narrow
PDG supplement actually ran1 worker despite limit32; it cannot establish parallel
scaling limits. Some300s limits included construction/warmup/profile. Do not
repeat expensive old failures without separating these causes and actual work.
Native self-loop/ring specializations also do not expose multiple nodes in their
Aggregate/Full dispatch; worker configuration is not a universal speedup claim.

Native named-value checkpoint TIDENCK1 does not contain graph continuation;
graph/application checkpoints do not restore the whole controller/RNG/data cursor.
The tiny model adapter is not arbitrary pretrained import. CUDA/Ascend/x86 target
qualification and optimized training remain extensions, not current evidence.
Suggested next priorities and resource limits remain in ROADMAP. The latest
Add result makes packed training VJP optimization the strongest next candidate.
No further heavy task is queued or implicitly authorized by this completed audit.

## Re-entry

```
git status --short --branch
git log -6 --oneline
/home/zlong/anaconda3/bin/python scripts/status.py
```

Environment: /home/zlong/anaconda3/bin/python, aarch64, Torch/LibTorch2.10.0+cpu,
GCC10.3.1/Python3.11.15/C++11 ABI; TORCH_DEVICE_BACKEND_AUTOLOAD=0.
Correctness pools1/build2; future experiments retain half-effective CPU/memory,
sequential heavy timing and maximum8 Ascend cards if that extension is requested.
Source/artifacts resolve under /var/tmp/zlong-graph-execution-foundation.
Preserve cited artifacts and failure reproducers; no subagents or push.
