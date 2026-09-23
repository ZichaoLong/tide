# Current handoff

Updated: 2026-09-23 (Asia/Shanghai). Branch graph-execution-foundation.
Overall graph execution foundation acceptance version is active, not complete.
Stage1 scope/matrix audit and twelve-config performance freeze are accepted;
[ROADMAP](ROADMAP.md) is the only backlog/stage index. [Scope evidence](evidence/foundation-scope-audit.md)
and [capabilities](execution-capabilities.md) distinguish existing verified
kernels from required gaps. Previous PDG/LH tuning is closed; defaults remain.

S2.1 native SettleGraph is implemented: independent C++ graph/rank validation,
encoded boundary construction, owner-preserving model mapping, dense-window
entry and complete-boundary projection. The binding is an adapter; standalone
C++ requires no Python. Directed development gate passed230 CPU FP64/FP32
checks in23.26s, including standalone literal-formula/VJP and complete native vs
Python traces, slots, ownership and chunk cuts. This is not the final full gate.

Terminal unit tide-native-settle-dev-20260923-a: inactive/dead, MainPID0,
ExecMainStatus0/Result=success; status.json and development.json both passed.
Source is stage1 e3ffad0 plus archived implementation (tree SHA256 in records).
Artifacts: artifacts/native-settle-dev-20260923-a/{source.tar.gz,status.json,
development.json,task.log,live-inspection.json,terminal-inspection.json}.
No active job. Current uncommitted changes are this coherent S2.1 code/tests/docs,
ready for local implementation commit. No user changes were overwritten.

Next: commit implementation, create a detached clean read-only worktree, build
`tidegraph-settle-check` with TIDE_PYTHON_BINDINGS=OFF in a separate build directory,
run FP64/FP32 and audit binary dependencies; commit reviewed evidence separately.
Then S2.2 independent multi-node ring, diamond with region selection, and Python
layered SettleGraph. Keep full CPU gate for the stage boundary. No new wide runs.
No push/sub-agents. All reference repositories remain read-only.

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
