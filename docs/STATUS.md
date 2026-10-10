# Current handoff

Updated 2026-10-10 (Asia/Shanghai). The new G1–G5 stage is active. Continue all
locally achievable implementation, correctness, finite selection evidence and
CUDA target handoff; commits do not pause the task. No subagents. Authorization
and execution contracts: [execution-flows §11](execution-flows.md); sole backlog:
[ROADMAP G1–G5](ROADMAP.md). Selector/new upstream semantics remain deferred.

## Source and next actions

Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`, branch
`graph-execution-foundation`. Initial clean HEAD was c0ce4ee5. Stage contract
05af8f8 and State/Read implementation132cada7b21081d7f611326e1e0971e057201b4c
are committed/pushed. Current uncommitted work: G1 metric/test/qualification
entry corrections, G2 component profiler, and G3 CUDA resident backend.
Keep coherent commits and use clean immutable qualification after implementation.
Do not edit a snapshot while its job runs. `TASK=/mi/data2T/zlong/tide-execution-flows`.

1. Inspect the full G1 CPU gate and directed correction gate below. Close all
   failures; commit G1 correction separately from unfinished CUDA code.
2. Finish CUDA standalone/Python/consumer compile and backend-neutral public
   target tests. Rebuild NPU shared backend after abstraction changes.
3. Profile NPU mixed/resident only with a current safe device allocation and
   fresh bounded budget; do not automatically retry the timed-out lease. Finish
   related CPU/NPU qualification and only justified selection measurements.
4. Commit evidence separately; update support/selection and exact CUDA target
   commands. Real GPU correctness/residency/multicard/performance remain pending.

## Current jobs and development evidence

All paths below are `TASK/{sources,builds,runs,plans}/NAME` unless specified.
Inspect with `systemctl --user show tide-NAME.service -p ActiveState -p Result
-p MainPID -p ControlGroup`; read run `status.json`, `task.log` and gate report.
Current services use background.slice, KillMode=control-group, Nice10, fixed
read-only sources, OMP/BLAS1 and bounded resources. Stop commands, if newly
requested: `systemctl --user stop tide-NAME.service` (current jobs only).

- `g1-cpu-qualification01`: clean132cada full CPU FP64/FP32 gate still running,
  last observed81%.90min,8 build jobs,CPUQuota1000%,16GiB,192 tasks. Command
  `scripts/job.py --output-dir RUN -- python scripts/qualify.py --output-dir RUN
  --jobs 8`. `verification/result.json` and `verification/tests.log` own result.
  Known failures: old State/Read replay/clock counter assertions, omitted online
  consumer build, and lean process tests inheriting Torch parent's ru_maxrss.
  These are being corrected; this gate is not passed.
- `g1-regression-dev04`: terminal exit0/MainPID0,173 tests passed. Matching core
  build and online consumer (`builds/g1-online-dev04`) validated metric/clock and
  qualification plumbing. Earlier State/Read development passed193,1538 and324
  checks; each scoped report retained. G1 custom/nondefault replay remains.
- `g1-regression-dev05`: terminal exit0/MainPID0,36 tests passed. Frozen directed test correction. Reuses dev04
  core by source/binary digest; runs grad-profile and measurement-lane tests.
  The latter now run from lean supervisors, preserving64MiB lane budgets and
  terminal wait4 checks. Reproducer: Torch-loaded parent/exec child both inherit
  ~232MiB ru_maxrss; lean child ~11MiB. Budget15min,CPUQuota400%,4GiB,128 tasks.
- `g3-cuda-dev04`: running frozen G3 build, reuses matching132cada CUDA core
  `builds/g3-cuda-core-dev01`. Plan `plans/g3-cuda-dev04.sh`; build
  `builds/g3-cuda-resident-dev04`.60min,4 jobs,CPUQuota800%,16GiB,192 tasks.
  dev03 compiled all102 shared semantic CUDA kernels and failed later on an old
  state reverse boundary test missing the reusable gather arguments; fixed here.
  dev01/dev02 compile failures retained. CPU source contracts dev02 compiled all
  102 kernels and passed1 CTest; no actual CUDA device claim.
- `g1-npu-build01`: clean132cada Python and standalone cores plus scale consumer
  built,5 CTests passed; terminal exit0/MainPID0/empty cgroup. Builds
  `g1-npu-python01`, `g1-npu-standalone01`, `g1-npu-scale01`.
- `g1-npu-qualification01`: terminal queue timeout after120.53s, no free device;
  **no candidate device code executed**. Planned64 module and8 training checks
  did not run. `runs/g1-npu-qualification01/queue.json` preserves receipt.
  Do not call it a correctness failure or silently repeat the unchanged job.
- `g2-component-smoke02`:6 small CPU processes (Read/Add/Attention × replay/batch),
  rows2,width8,parity+trace passed. Smoke timings are not selection evidence.
  New profiling script still needs device-kernel trace acceptance. G2/G5 actual
  medium/resident/selection experiments remain unexecuted.

## Environment and evidence limits

Public NPU module `libtorch-npu/2.10.0-cann9.0.0`, Python under
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311`. Standalone SDK
is separate from wheel runtime. This explicit user stack supersedes old guide
paths. Private CUDA module `~/privatemodules/torch-cuda/2.10.0-cu128`, toolkit
12.8.1 under `/mi/data2T/zlong/gpu-toolchains`; explicit80/90/100 architectures.
No GPU here; drivers unchanged. Last disk check root64GiB/data84GiB available;
external use fluctuates, recheck before large writes. Last inventory all16 NPUs
had external processes; no process was touched. New experiments use existing
project records, Trackio off, heavy timing serial, no automatic retries.

CUDA work shares graph/kernel semantic sources with independent CUDA conditional
control and P2P backend. Initial semantic adapter is scalar per logical worker;
compilation is not proof of correctness, residency, fusion or performance.
State/Read batch graph composition likewise is not a fused-kernel claim.

## Preserved previous stage and protected history

F1–F7 earlier qualification/selection remain valid only for cited old sources.
Old matrix61/120 bound FP32 timings,59 missing cases and later failed follow-ups
are not a new execution queue. [Terminal record](evidence/selection-terminal-20261010.md),
[selection advice](evidence/selection-review-20261009.md). CPU78e9df6 full gate,
exact-comparator/eager/resident e69b3bd and old CUDA host-build results are not
qualification for current modifications.

Preserve original strict near-tie witness: CPU246 one FP32 ULP ahead; resident
245/246 tie selects245; proposal error7.7039e-6,events2325/2327.
[Witness](evidence/original-add-route-witness-20261004.md). Do not weaken discrete
comparison. Full/Aggregate batched VJPs and explicit resident device VJPs were
already enabled; retain independent candidates and Python/native/standalone IDs.

**Never resume, stop, signal or clean historical-cpu-attention-01 / worker2686919.**
The malformed historical build-reverse-gather-python-dev01 status makes
scripts/status.py exit1; preserve it, not a current task blocker. No project
writes to ObsidianVault; lh/fractal-latcarf and other references stay read-only.
