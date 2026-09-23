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
Medium evidence: docs/evidence/foundation-medium.md.
Medium audit complete: all108 records validate; repeat logical/operator counts
identical within each variant. Audit tuple/list readback failure was fixed and
retained in its audit-readback-repro directory.

Follow-up implementation committed as3d2622a after130 directed tests/56.42s,
exit0/MainPID0 (foundation-records-dev-20260923-a). Changes:
- reject explicit timeout0; publish completed measured phase before profiling;
  retain effective thread and binary identity before model allocation;
  allow30s bounded group cleanup after a large-process deadline;
  directed regression gate passed in foundation-records-dev-20260923-a;
  artifacts/foundation-records-dev-20260923-a, same immutable matching build.
- README, capability and portability-contract docs; schema validator passed.
  x86 target-machine execution is an optional extension of this acceptance.
No unreviewed user changes, no push/subagents/reference writes.

Active bounded large assessment: tide-foundation-large-20260923-a,
clean source3d2622a, qualification/foundation-large-3d2622a; same immutable native
build. Output artifacts/foundation-large-20260923-a/{status.json,task.log,suite/}.
Exact preflight/budget/command: artifacts/foundation-large-a-launch.json.
Wide D2048/B512 target384 body nodes/17,722,251,712 parameters; narrow D128/B512
target46,912 body nodes/8,496,773,056 parameters. DAG and Settle run sequentially;
node workers32, ATen/BLAS1,2 reset warmups,1 bounded repeat,300s per stage plus
up to30s cleanup. RuntimeMaxSec4000. Dynamic half-budget160 CPUs/~740GiB at launch;
read launch record for exact current memory. No heavy task may overlap.
No new tuning candidate selected. Existing PDG wide/narrow evidence reused.

Next: inspect each terminal run and cleanup; retain all failures/skipped larger
stages. Audit with artifacts/review_foundation_suite.py and summarize. Review/
commit S5 evidence, freeze final clean source, then launch the prepared
artifacts/qualify_foundation_final.py through scripts/job.py. It exports/rebuilds/
smokes in a new path with spaces, runs full CPU FP64/FP32 and builds C++ Settle
with TIDE_PYTHON_BINDINGS=OFF plus loader/forward/VJP checks. Audit S6, update final
matrix/status, commit evidence and stop this acceptance scope.

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
