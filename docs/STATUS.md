# Current handoff

Updated: 2026-09-24 (Asia/Shanghai). Branch graph-execution-foundation.
**Six-stage graph execution foundation acceptance COMPLETE.**
All mandatory implementation/correctness units passed for the declared finite
CPU FP32/FP64 profiles; fixed performance evaluation is closed with honest limits.
ROADMAP is the sole backlog; only explicitly optional extensions remain.

New user-requested Add comparison is active (2026-09-24), separate from the
completed six-stage acceptance. Scope: docs/add-scale-comparison.md. Add scale
fixture, portable Add/grad-forward/outer-timer options and derived-descriptor
precision validation passed472 directed CPU FP32/FP64 tests/113.88s.
Gate: artifacts/add-compare-dev-20260924-d, unit inactive/MainPID0/exit0.
Failures -a (bad pytest path), -b and -c (FP32-derived FP64 norm tolerance) remain
retained; no runtime formula changed. No reference sources changed.
Implementation is ready for a coherent commit and frozen worktree. Next: launch
artifacts/run_add_comparison.py against that clean source with recorded host
resources,4 real native smoke cases and12 D2048/B512 Add runs (3 repeats per
engine/mode). Planned unit/output: tide-add-scale-comparison-20260924-a /
artifacts/add-scale-comparison-20260924-a. Main source may change only after
freezing the input checkout; never edit its files or experiment driver while live.
After terminal inspection, audit identities/metrics, write evidence and commit it.
Do not claim this Add extension qualified from the preceding7741-test gate.

Delivery entry: docs/evidence/foundation-final.md.

- Six implementation classes/encoding/independent anchors: final report + ROADMAP.
- Module, prefill, option, fallback and training scope: execution-capabilities.md.
- Performance: evidence/foundation-medium.md and evidence/foundation-large.md.
- One-command build/smoke and actual CLI: foundation-benchmarks.md.
- Semantic contract: semantics.md and settle-embedding.md.

Qualified clean source:81a1b266af49d918aa6e1587e4ed9e0c4d4e5eb5.
Read-only checkout: qualification/foundation-final-81a1b26.
Raw evidence: artifacts/foundation-final-20260923-a/reviewed-audit.json,
pipeline.json, full-cpu/{result.json,tests.log,test-tmp}, relocated-smoke/,
relocated source with spaces/build/, standalone-build/ and stage logs.
7741 CPU FP32/FP64 tests passed/1235.63s;489 source files/13 binaries and36 fresh
process checkpoint manifests/payloads audited. Fresh relocated rebuild and17
smoke variants passed. Separate TIDE_PYTHON_BINDINGS=OFF build, native Settle
FP32/FP64 formula/VJP/prefill/continuation and no-Python loader checks passed.
The evidence commit containing this handoff changes docs/support metadata only;
implementation/tests remain identical to the qualified source. Exact final local
commit and clean-status audit: artifacts/foundation-final-20260923-a/delivery.json.
No unreviewed user changes; no uncommitted changes after that evidence commit.

Performance closure:

- 12 frozen medium configs/108 completed runs,3 independent repeats each.
- New large assessments:12 launched stages,7 complete/5 timeouts, plus1 unlaunched
 larger Settle stage. Per-run failures remain failures despite evaluated wrappers.
- PDG/LH wide17.27B retained. Historical narrow is actually16.608B/115713 physical
 nodes; exact-count audit retained. The frozen8.497B PDG supplement completed
 N256/N1024, then constructed46912 nodes before timing out in its second warmup.
- Its fixed native-stream-packed variant is serial (actual1, worker limit32),
 stride1/cut6 retains pending tail; input throughput is not completed-output
 throughput. It does not establish parallel narrow performance limits.
- No target-scale success claim for timed-out/unlaunched targets. No new tuning
 candidates or defaults; no heavy task overlapped formal timing.

At the completed foundation delivery there were no task live jobs or descendants. Reviewed terminal units, all MainPID0/exit0:

- tide-foundation-final-20260923-a (all qualification stages passed)
- tide-foundation-medium-20260923-a (all108 runs completed)
- tide-foundation-large-20260923-a (DAG/Settle bounded assessment)
- tide-foundation-pdg-narrow-20260923-a (PDG bounded supplement)
Per-case timeout exits-15 and cleanup records are retained. Previous failed/
cancelled development gates and audit-helper reproductions are not relabeled.

The six-stage acceptance scope remains closed. The new Add comparison above is
explicitly requested. Further model/backend/performance work needs a new request;
ROADMAP extensions include (CUDA/NPU/x86 target runtime, arbitrary models/LH configs,
higher-order AD or whole training-controller/RNG/data-cursor resume).

Re-entry verification:

```
git status --short --branch
git log -6 --oneline
/home/zlong/anaconda3/bin/python scripts/status.py
```
Environment: /home/zlong/anaconda3/bin/python; aarch64, Torch/LibTorch2.10.0+cpu,
GCC10.3.1/Python3.11.15/C++11 ABI; TORCH_DEVICE_BACKEND_AUTOLOAD=0. Correctness
pools1/build2. Resource budgets were dynamically half-effective (160 CPUs,
719–743GiB for the new large runs), with actual RSS/thread/cgroup records.
Source/artifacts resolve under /var/tmp/zlong-graph-execution-foundation.
Retain all cited sources/builds/results/failure reproducers. No push, subagents,
or writes to read-only references were performed. Durable records are local truth.
