# Current handoff

Updated: 2026-09-23 (Asia/Shanghai). Branch graph-execution-foundation.
Overall graph execution foundation acceptance version is active, not complete.
Stage1 scope/matrix audit and twelve-config performance freeze are accepted;
[ROADMAP](ROADMAP.md) is the only backlog/stage index. [Scope evidence](evidence/foundation-scope-audit.md)
and [capabilities](execution-capabilities.md) distinguish existing verified
kernels from required gaps. Previous PDG/LH tuning is closed; defaults remain.

S2.1 native SettleGraph is verified for the rank-aligned broadcast-input/sum-output
profile. Implementation commit ac19aad; [reviewed evidence](evidence/native-settle-frontend.md).
Directed gate230 passed/23.26s. Independent clean CMake build with
TIDE_PYTHON_BINDINGS=OFF and standalone FP64/FP32 passed; no libpython/torch_python
runtime dependency.456 source hashes match Git archive. Both units are terminal
inactive/dead, MainPID0/exit0; no active durable job.

Current uncommitted S2.2 work: native/Python ring and diamond (including paired
middle region) specialization; independent Python layered SettleGraph; frame
formulas split into specialized_step.py; directed tests in
 tests/test_specialized_topologies.py. Native changes still require rebuild/test.
An interactive Python directed check failed1/90 on an over-scaled composite
FP32 Linear VJP: test helper squared an already quadratic scalar objective.
Reproducer retained at artifacts/specialized-linear-fp32-repro-20260923/.
Inspect objective-diagnostic.json and failure.log; direct declared composite
objective VJPs pass unchanged tolerances. Correct the test loss composition,
then run durable directed gate including old specializations and native frontend.
No execution-code fix or tolerance relaxation has been made for this probe.

S3 audit refinement: existing `delta` is explicitly Gated DeltaRule (learned decay
and beta); plain ungated DeltaRule is a required missing profile. Preserve the
existing formula/name and add a separately named profile in S3.1.
No push/sub-agents. Reference directories remain read-only. Native checkpoint
is already qualified; no new wide performance run is queued.

Commands before continuing (also after compaction):
```bash
git status --short --branch
git log -6 --oneline
/home/zlong/anaconda3/bin/python scripts/status.py
```
Use `scripts/develop.py` under `scripts/job.py`/detached background.slice/Nice10
for directed native build/tests. Freeze all job input source until terminal, or
use an isolated snapshot. Coherent implementation commit precedes long clean
qualification; reviewed evidence has a separate commit. Never call a live run
passed. Durable text/JSON writes use scripts/durable_records.py with readback.

CPU aarch64, Python3.11.15/Torch2.10.0+cpu/GCC10.3.1, CPU FP64/FP32 only.
Python: /home/zlong/anaconda3/bin/python. TORCH_DEVICE_BACKEND_AUTOLOAD=0,
correctness ATen/OpenMP/BLAS1/build2. Startup resource record:
`artifacts/foundation-audit-20260923/resources.json`:320 available CPUs, default
aggregate160; dynamic half-effective-memory budget ~755GiB. Refresh before large
jobs; record actual thread pools/RSS/cgroup. Eight NUMA nodes. ~27GiB disk free.
Repository symlink resolves to /var/tmp/zlong-graph-execution-foundation/repository;
build/artifacts are siblings. Retain cited artifacts and failed reproducers.

Previous terminal job tide-packed-transport-20260923-143526: inactive/dead,
MainPID0, exit0; full gate6897 passed/891.90s. Detailed source, artifacts, binaries,
portable kit and performance limits live in [reviewed evidence](evidence/packed-transport.md).
Raw artifacts retained under artifacts/packed-transport-20260923-143526/.
No repeated stable speedup for combined options; experimental defaults stay off.
Native TIDENCK1 value/optimizer checkpoint, graph continuation v5 and two-clock
application bundle are distinct scopes; none promises full controller/RNG resume.

S2.1 artifact directories: native-settle-dev-20260923-a and
native-settle-standalone-20260923-a under artifacts/. Standalone snapshot:
/var/tmp/zlong-graph-execution-foundation/qualification/native-settle-ac19aad.
Its source/build/driver stay immutable. The reviewed evidence owns exact paths,
commands, binary identity and terminal audit. Full stage2 gate remains after S2.2.
