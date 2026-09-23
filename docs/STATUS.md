# Current handoff

Updated: 2026-09-23 (Asia/Shanghai). Branch graph-execution-foundation.
Overall six-stage acceptance remains active. ROADMAP is the sole backlog.
S1 audit/frozen12-config suite and S2 native Settle/independent schedules accepted:
docs/evidence/foundation-scope-audit.md, native-settle-frontend.md and foundation-stage2.md.
S2 clean gate7167 passed on70809fd; source/binary/exit audits retained.

S3 implementation4a2c366/1a982a1: cross-graph packed transport/Next, explicit
fallbacks, ungated DeltaRule, projection strides and tiny RMSNorm/RoPE/GQA/SwiGLU
adapter. Directed gates1098 and736 passed. S4 implementation5014c3d plus494e6a9:
six-class multiple updates and36 fresh-process checkpoint trajectories. Original
274-test gate had a Settle logical-cut/position test-helper defect;54 corrected
tests passed. Failure retained in artifacts/settle-training-clock-repro-20260923/.
Do not accept cancelled partial stage34-a (exit143, no descendants).

Accepted combined clean S3/S4 qualification: tide-foundation-stage34-20260923-b,
source494e6a9, qualification/foundation-stage34-494e6a9; read-only reused build
qualification/modules-dev-20260923-a/build. This build's archived dirty source
matches C++ bytes; it was not rebuilt at494e6a9. Output/status/full-cpu result and
logs: artifacts/foundation-stage34-20260923-b/. 7611 passed/1153.34s, exit0; inactive/dead/MainPID0, no descendants.
All source hashes,13 binaries and36 fresh-process payloads audited. Reviewed
evidence: docs/evidence/foundation-stage34.md. Final S6 rebuilds clean source in a new directory.

Uncommitted S5 implementation: foundation_workloads/execute/measure/worker/control/
lifecycle/resources and benchmark_foundation.py; optional C++ work/replay metrics.
Frozen12 workloads and2 large presets; explicit phases, reset warmups, detached
training windows, dynamic half-effective resources, process-group timeout/RSS/
cancellation/reaping, source/build identity and local Trackio records. Scalar
transport weights fixed0.8, deterministic matrices std1/sqrt(D). Python reference
retains traces; native wall timing trace-free, separately disclosed. Detailed
accounting is a separate pass. No formal timing or new tuning candidate started.

Development build/test launching in its own archived dirty snapshot:
tide-foundation-bench-dev-20260923-a; qualification/foundation-bench-dev-20260923-a;
artifacts/foundation-bench-dev-20260923-a/{status.json,task.log,development.json}.
Main-tree edits may continue; never edit either active checkout/build. Directed
runner tests cover exact owners, complete client semantics, actual training
updates/detaches, accounting parity, prefill/fallback and child reaping. Commit
a coherent implementation only after gates pass. Then smoke/record validation,
fixed performance suite sequentially with no heavy overlap, S5 reviewed report,
and clean S6 full CPU/standalone/relocation qualification.

Re-entry:
```
git status --short --branch
git log -6 --oneline
/home/zlong/anaconda3/bin/python scripts/status.py
```
Python /home/zlong/anaconda3/bin/python; aarch64/Torch2.10.0+cpu/GCC10.3.1.
CPU FP32/FP64; TORCH_DEVICE_BACKEND_AUTOLOAD=0; correctness pools1/build2.
Dynamic resources scripts/foundation_resources.py:320 CPUs, aggregate160 at last
check; eight NUMA nodes; recompute half-effective-memory and disk before large jobs.
No NPU/CUDA claim. Disk27GiB free at startup. Source resolves under /var/tmp.
Native value checkpoint is already qualified. Existing LH/PDG wide and narrow
failure evidence remains; no repeated wide tuning. No push, subagents or reference
writes. Use durable_records.py with readback. Preserve all cited failure artifacts.
