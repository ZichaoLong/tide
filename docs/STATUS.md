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

S2.2 is implemented and directed-tested: Python/C++ multi-node ring and diamond
(singleton or shared middle region); Python fully connected layered SettleGraph
with independent scalar frame/layer schedule. Delays/ragged input, empty selection,
all state slots, tensor history, initial-state VJPs, shared owners, isolated roots,
chunk cuts and existing chain/self-loop regression passed. Durable gate554 passed
in61.54s; build exited0. Unit tide-specialized-topologies-dev-20260923-a is
inactive/dead, MainPID0/exit0. Source archive, tree hash, tests/build manifests,
terminal-inspection.json and logs: artifacts/specialized-topologies-dev-20260923-a/.
Current edits are coherent S2.2 implementation/tests/contracts ready to commit.
No active job and no unrelated user edits.

Original failed Python Linear test is retained at
artifacts/specialized-linear-fp32-repro-20260923/. It squared the already quadratic
composite objective again. Declared direct-objective VJPs pass unchanged tolerances
(max abs4.77e-7); only test loss composition changed. No execution workaround.

Next bounded step: commit S2.2, freeze a clean read-only worktree and run complete
CPU FP64/FP32 regression through scripts/build.py and scripts/verify.py. Then
commit reviewed stage2 evidence separately. While that isolated gate runs, S3
can progress on the main worktree. Priority: legal frontier packed source/event/
sequence transport, batch Next/reset and explicit fallback/capability counters;
then ungated DeltaRule and the small norm/position/RoPE/mask/cache adapter.

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
