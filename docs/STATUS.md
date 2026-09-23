# Current handoff

Updated: 2026-09-23 (Asia/Shanghai). Branch graph-execution-foundation.
Acceptance active: S1–S4 accepted; S5 performance assessed; S5 relocation/S6 next.
ROADMAP is the only backlog. No task job is live.

Accepted implementation and correctness:
- S1 frozen12 medium configs/two large presets; scope-audit.md.
- S2 standalone native Settle frontend and independent ring/diamond/layered;
  native-settle-frontend.md and foundation-stage2.md.
- S3/S4:7611 CPU FP64/FP32 tests at clean494e6a9, complete six-class training,
  modules/options/prefill and36 fresh-process checkpoint trajectories;
  docs/evidence/foundation-stage34.md. Reused build explicitly identified.
- Runner implementation9f0cfff and recording fixes3d2622a:130 directed tests,
  real phase timings, work/replay accounting, resource/lifetime control/export.

S5 reviewed evidence:
- docs/evidence/foundation-medium.md:12 configs/108 runs,3 independent repeats,
  all completed. Artifacts foundation-medium-20260923-a; unit exit0/MainPID0.
- docs/evidence/foundation-large.md: two DAG/Settle presets,5 complete stages,
  4 timeouts and1 unlaunched stage. All9 records validate; all groups reaped.
  Clean3d2622a; artifacts/foundation-large-20260923-a contains reviewed-audit.json,
  report.json, suite and terminal-inspection.json. Unit inactive/exit0/MainPID0.
  Wrapper evaluation success does not certify failed or unrun target scales.
- Existing PDG/LH wide and narrow failure reused, no new tuning candidate/default.
  No heavy task overlapped performance timing.

Next bounded increment: commit this reviewed S5 evidence, freeze that clean
commit in a new read-only worktree, and run artifacts/qualify_foundation_final.py
through scripts/job.py in unit tide-foundation-final-20260923-a. Use new output
artifacts/foundation-final-20260923-a and fresh build directories; do not reuse
the development build for S6. Driver exports to a new path with spaces, builds/
smokes all17 variants, runs full CPU FP32/FP64, then separately builds native
Settle with TIDE_PYTHON_BINDINGS=OFF and verifies both dtypes/VJP/loader closure.
Audit terminal results with artifacts/review_foundation_final.py OUTPUT UNIT.
Update final matrix/status/evidence, commit, verify no live jobs, then stop.

Working changes at this handoff: reviewed large evidence + ROADMAP/STATUS only.
Prepared audit/qualification/inspection drivers are retained under ignored
artifacts. Audit helper failures remain under the medium/large audit-repro paths;
original experiment failures and cancelled stage34-a remain unchanged.

Re-entry:
```
git status --short --branch
git log -6 --oneline
/home/zlong/anaconda3/bin/python scripts/status.py
```
Environment: /home/zlong/anaconda3/bin/python, aarch64, Torch/LibTorch2.10.0+cpu,
GCC10.3.1/Python3.11.15/C++11 ABI. CPU FP32/FP64 only, no CUDA/NPU/x86 runtime
claim. TORCH_DEVICE_BACKEND_AUTOLOAD=0, correctness pools1/build2. Refresh dynamic
half-effective resources and disk before launch; prior budget160 CPUs/~743GiB,
disk27GiB free. Source/artifacts resolve under /var/tmp/zlong-graph-execution-foundation.
Keep all cited source/build/results and failure reproductions. No push, no
subagents, no writes to read-only references. Use durable_records.py for records.
