# Current handoff

Updated: 2026-09-23 (Asia/Shanghai). Branch graph-execution-foundation.
Overall acceptance remains active; current stage S5. ROADMAP is the only backlog.

Accepted: S1 frozen scope/12 medium configs/2 large presets; S2 standalone native
Settle and independent ring/diamond/layered schedules; S3 modules/prefill/options;
S4 six-class training and fresh-process persistence. Evidence:
- docs/evidence/foundation-scope-audit.md, native-settle-frontend.md, foundation-stage2.md
- docs/evidence/foundation-stage34.md: clean494e6a9,7611 passed/1153.34s,
  470 source files/13 binaries/36 fresh-process payloads audited, terminal exit0.
  Reused immutable archived module build is explicitly identified; S6 rebuilds.
The cancelled stage34-a/helper clock defect is retained in
artifacts/settle-training-clock-repro-20260923/; do not call that attempt passed.

S5 implementation9f0cfff: unified three-family runner, real training phases,
separate work/replay accounting, dynamic budgets/lifetime control, source export.
127 directed tests/55.95s passed in foundation-bench-dev-20260923-b;5 durable
records validated. Earlier build gate a passed524 tests but failed the original
orphan deadline check. Cancellation/proc-scan failure and6-test correction remain
in artifacts/foundation-lifecycle-repro-20260923/. Both development units terminal.

Completed formal medium evaluation: tide-foundation-medium-20260923-a.
Clean frozen source: qualification/foundation-performance-9f0cfff.
Read-only build: qualification/foundation-bench-dev-20260923-a/build.
Outputs: artifacts/foundation-medium-20260923-a/{status.json,task.log,suite/}.
Launch/resource and live process/cgroup/binary inspection records are retained.
All12 configs,3 independent repeats,2 reset warmups, per-process180s, ATen1,
node-worker limit4, RuntimeMaxSec22000. All108 runs completed, zero failures; unit exit0/MainPID0. Audit/report at
artifacts/foundation-medium-20260923-a/{reviewed-audit.json,report.json,report.md}.
No formal timing remains active.

Follow-up implementation ready to commit after130 directed tests/56.42s,
exit0/MainPID0 (foundation-records-dev-20260923-a). Changes:
- reject explicit timeout0; publish completed measured phase before profiling;
  retain effective thread and binary identity before model allocation;
  allow30s bounded group cleanup after a large-process deadline;
  directed regression gate passed in foundation-records-dev-20260923-a;
  artifacts/foundation-records-dev-20260923-a, same immutable matching build.
- README, capability and portability-contract docs; schema validator passed.
  x86 target-machine execution is an optional extension of this acceptance.
No unreviewed user changes, no push/subagents/reference writes.

Next bounded sequence:
1. Audit terminal medium records, source/binary hashes, actual exits and cleanup;
   summarize three repeats and dispersion with scripts/foundation_report.py.
2. Run directed benchmark/CLI tests for the small record follow-up, commit it.
3. Freeze new clean source; evaluate both large presets for timed-dag and settle,
   node workers32/ATen1,300s per stage, stopping a family at its first failure.
   Reuse existing wide PDG/LH and narrow PDG timeout evidence; no repeat search.
4. Review/commit S5 evidence, then freeze final source and run the prepared
   artifacts/qualify_foundation_final.py driver via scripts/job.py. It exports
   to a new path with spaces, rebuilds/smokes, runs full CPU FP64/FP32, and builds
   C++ Settle with TIDE_PYTHON_BINDINGS=OFF plus loader/forward/VJP checks.
5. Audit S6, update final matrix/status, commit evidence and stop this scope.
No new performance bottleneck/candidate has been selected; defaults stay unchanged.

Re-entry commands:
```
git status --short --branch
git log -6 --oneline
/home/zlong/anaconda3/bin/python scripts/status.py
```
Python /home/zlong/anaconda3/bin/python; aarch64/Torch2.10.0+cpu/GCC10.3.1,
C++11 ABI. CPU FP32/FP64 only, TORCH_DEVICE_BACKEND_AUTOLOAD=0. Correctness
pools1/build2. Recompute scripts/foundation_resources.py before large work:
last320 CPUs/eight NUMA nodes, aggregate half-budget160; memory is dynamically
half-effective, never a fixed256GiB cap. Disk27GiB at last check. Source resolves
under /var/tmp/zlong-graph-execution-foundation/repository. Preserve cited builds,
raw records and failures. Use durable_records.py and readback for all handoffs.
