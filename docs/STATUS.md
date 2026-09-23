# Current handoff

Updated: 2026-09-23 (Asia/Shanghai). Branch graph-execution-foundation.
Overall acceptance is active, not complete. ROADMAP is the only backlog.
Stage1 audit/frozen twelve-config suite accepted. S2 native standalone SettleGraph
verified at ac19aad; evidence/native-settle-frontend.md has exact qualification.
S2 independent ring/diamond Python+C++ and Python layered Settle implemented at
70809fd; 554 directed tests passed. Complete CPU gate is still running below.

S3.2 frontier/native Settle and independent specialization option migration now
passes 1098 directed CPU FP64/FP32 tests (182.07s), build/test exit0. Changes:
packed source/event/sequence transport, causal same-time Next/reset batching,
parallel independent sample-region selection, compact events/deferred cleanup,
explicit prefill fallback reasons and unsupported-policy rejection. Scalar
semantic replay remains the training baseline; defaults unchanged. Source was
frozen throughout the gate. Raw source archive/development/build and terminal
audit: artifacts/frontier-options-dev-20260923-a/. Unit
`tide-frontier-options-dev-20260923-a` inactive/dead, MainPID0/exit0; no descendants.
These coherent implementation/tests/docs changes are ready for local commit.

Active job: `tide-foundation-stage2-20260923-a`, source70809fd in clean read-only
/var/tmp/zlong-graph-execution-foundation/qualification/foundation-stage2-70809fd.
MainPID126713 last observed; background.slice/Nice10, build2/ATen-BLAS1;
RuntimeMaxSec3600. Immutable driver artifacts/qualify_foundation_stage2.py runs
scripts/build.py then scripts/verify.py --device cpu --dtype both. Output:
artifacts/foundation-stage2-20260923-a/{status.json,pipeline.json,build/,
full-cpu/result.json,full-cpu/tests.log}; launch foundation-stage2-launch.json.
Do not modify its source/build/driver. No benchmark overlaps. Check terminal
unit, actual exits, process cleanup, source hashes against Git archive and all
binary hashes, then commit reviewed S2 evidence separately.

Next bounded increment after S3.2 commit: add explicitly named ungated DeltaRule
(existing delta is Gated DeltaRule; preserve it), deterministic model-style
layout/norm/explicit-position/RoPE/mask/cache adapter and formula/chunk/VJP tests.
Cross-graph QKV/output physical projection strides need owner/update coverage;
scale initialization already supports them (not a general Native constructor
option). Follow ROADMAP S3.1/S3.3, then clean stage3 qualification.
No unrelated user changes, no push/sub-agents/reference writes. Native value
checkpoint already qualified; no wide LH/PDG tuning is queued.

Re-entry:
```
git status --short --branch
git log -6 --oneline
/home/zlong/anaconda3/bin/python scripts/status.py
```
Python /home/zlong/anaconda3/bin/python; aarch64/Torch2.10.0+cpu/GCC10.3.1.
CPU FP32/FP64 only. TORCH_DEVICE_BACKEND_AUTOLOAD=0, correctness threads1/build2.
Startup resources: artifacts/foundation-audit-20260923/resources.json;320 CPUs,
aggregate default160, eight NUMA nodes; dynamic half-effective-memory ~755GiB.
Recompute before large work. Disk ~27GiB free. Main repository resolves to
/var/tmp/zlong-graph-execution-foundation/repository; build/artifacts are siblings.

Retain artifacts/specialized-linear-fp32-repro-20260923/: original test squared an
already quadratic composite loss. Direct declared-objective gradients pass the
unchanged tolerance (max abs4.77e-7); fixed only test composition. Prior packed
transport full gate6897 passed/891.90s and wide/narrow performance evidence remain
valid within their scope; no repeated stable combined-option gain, defaults off.
Native TIDENCK1 named values, graph continuation v5 and application bundles are
distinct; none restores the entire training controller/RNG/data cursor.

Durable writes use scripts/durable_records.py and readback. Commit tested code,
then freeze isolated source for long qualification and separately commit reviewed
evidence. Never edit inputs read by a live development job or call a live run
passed. STATUS is current handoff, not a session diary.
