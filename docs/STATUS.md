# Current handoff

Updated: 2026-09-23 (Asia/Shanghai). Branch graph-execution-foundation.
Overall graph execution foundation acceptance version is active, not complete.
Stage1 scope/matrix audit and twelve-config performance freeze are accepted;
[ROADMAP](ROADMAP.md) is the only backlog/stage index. [Scope evidence](evidence/foundation-scope-audit.md)
and [capabilities](execution-capabilities.md) distinguish existing verified
kernels from required gaps. Previous PDG/LH tuning is closed; defaults remain.

Next bounded increment S2.1: implement standalone C++ SettleGraph construction,
encoding, model alias mapping, input clocks and complete-boundary projection;
reuse native frontier. Add standalone FP64/FP32 forward/VJP and directed parity,
malformed spec/cut and ownership tests. Then S2.2 independent ring/diamond/layered.
No native checkpoint reimplementation and no new wide tuning are queued.

No active job. No pre-existing uncommitted work. Current edits are the stage1
ROADMAP/capability/evidence/config freeze and this handoff, ready for review and
local commit. Check git status/log on re-entry for exact current source. Original
implementation at entry is recorded in the scope evidence. Do not push or use
sub-agents. LH/fractal-latcarf/ObsidianVault remain read-only.

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
