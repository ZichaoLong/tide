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
and correction8a4d60f plus Norm VJP rounding fixa02c18c are committed/pushed. G3 CUDA resident implementation committed/pushed as bda25de.
G2 component profiler and accurate State/Read policy metadata are committed/pushed
as d1df03c after6 passed CPU entry smokes; no actual selection timing yet.
Target-gate follow-up ecafdf4 is committed/pushed: automatic capacity tests,
backend-neutral owner devices, and an explicit existing strict/conditioned
full-training option. Host placement/CLI/syntax checks passed; no new device claim.
Pending entry-only correction: complete peer qualification explicitly requires
remapped logical0 (the existing peer checker uses0/1); public runtime indices
remain general. No component/runtime source changes.
G1 metric/gate fixes and exact Norm VJP rounding are committed.
Keep coherent commits and use clean immutable qualification after implementation.
Do not edit a snapshot while its job runs. `TASK=/mi/data2T/zlong/tide-execution-flows`.

1. Inspect clean a02c18c full G1 CPU qualification03. Corrections passed directed
   gates and are committed. Preserve failed01 and cancelled02 records.
2. Inspect the running clean bda25de CUDA/NPU builds, including Python/standalone
   plugins and installed consumers. Review the small target-gate follow-up. Complete the
   source-related CPU host gate and retain actual GPU execution as target-pending.
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

- `g1-cpu-qualification01`: clean132cada full CPU gate terminal failed:
  73 failed,9451 passed,723 skipped in2136.79s.69 failures were accounting,
  omitted online build and inherited RSS test harness issues fixed in8a4d60f.
  Four additional norm32/FP64 proposal VJP failures are real rounding regressions:
  x*(dy/norm) differs from reference dy*(x/norm). Original tolerances preserved.
  `verification/tests.log` and `verification/result.json` retain all witnesses.
- `g1-cpu-qualification02`: clean8a4d60f full gate cancelled (exit143,MainPID0,
  empty cgroup) after the same unresolved Norm VJP defect was diagnosed in gate01.
  No pass claim. Its fixed source remains in use by the separate NPU build02.
- `g1-norm-dev06`: terminal exit0,378 tests passed. Read VJP order fix in Python/C++, exact FP32-to-FP64
  regression witness, full norm/read/isolated-gradient/State directed tests.
  Frozen source/build/run of same name, plan `plans/g1-norm-dev06.sh`.
  45min,8 build workers,CPUQuota1000%,16GiB,192 tasks. The numerical fix is
  committed/pushed as a02c18c; both prior full gate records remain unchanged.
- `g1-cpu-qualification03`: running clean committed a02c18c source, full CPU
  FP64/FP32 plus complex and installed-consumer gates. Dedicated source/run;
  exact matching immutable `builds/g1-norm-dev06` is reused by digest through
  `qualify_library.py --reuse-build`. Plan `plans/g1-cpu-qualification03.sh`,
  results `runs/g1-cpu-qualification03/gate`,90min,8 CPUs,16GiB,192 tasks.
- `g3-cuda-clean01` / `g3-npu-clean01`: running clean bda25de qualification
  builds from `sources/g3-backends-clean01`, plans of matching names. Each
  builds fresh core, standalone/Python resident and installed combined consumers;
  CUDA adds all102 CPU source contracts and directed host regressions.
  Builds have `g3-{cuda,npu}-*-clean01` names.90min each,4 workers,8 CPUs,16GiB,
  256 tasks. Build/CPU evidence only; no GPU operation and no NPU qualification
  workload is launched here. Inspect every terminal stage before claiming pass.
- `g1-regression-dev04`: terminal exit0/MainPID0,173 tests passed. Matching core
  build and online consumer (`builds/g1-online-dev04`) validated metric/clock and
  qualification plumbing. Earlier State/Read development passed193,1538 and324
  checks; each scoped report retained. G1 custom/nondefault replay remains.
- `g1-regression-dev05`: terminal exit0/MainPID0,36 tests passed. Frozen directed test correction. Reuses dev04
  core by source/binary digest; runs grad-profile and measurement-lane tests.
  The latter now run from lean supervisors, preserving64MiB lane budgets and
  terminal wait4 checks. Reproducer: Torch-loaded parent/exec child both inherit
  ~232MiB ru_maxrss; lean child ~11MiB. Budget15min,CPUQuota400%,4GiB,128 tasks.
- `g3-cuda-dev04`: terminal exit0, complete standalone CUDA build and7 host
  CTests passed. Frozen G3 build, reuses matching132cada CUDA core
  `builds/g3-cuda-core-dev01`. Plan `plans/g3-cuda-dev04.sh`; build
  `builds/g3-cuda-resident-dev04`.60min,4 jobs,CPUQuota800%,16GiB,192 tasks.
  dev03 compiled all102 shared semantic CUDA kernels and failed later on an old
  state reverse boundary test missing the reusable gather arguments; fixed here.
  dev01/dev02 compile failures retained. CPU source contracts dev02 compiled all
  102 kernels and passed1 CTest; no actual CUDA device claim.
- `g3-public-cpu-dev01`: failed before collection due to a nonexistent test
  path in its launch plan; no source defect diagnosed. `g3-public-cpu-dev02`
  uses the same frozen dev01 source with corrected existing test paths.
  Terminal exit0:140 passed/84 optional-device skips. Backend-neutral public configuration/consumer
  CPU regression, existing matching dev04 CPU core, frozen scoped source.
  20min,4 CPUs,6GiB,128 tasks; optional real device cases skip and are not
  target qualification. The new complete target entry requires3 real devices
  and fails if any selected public case skips.
- `g1-npu-build02`: terminal exit0/MainPID0/empty cgroup,5 CTests passed; three core/scale builds from clean8a4d60f
  source `sources/g1-cpu-qualification02`, plan `plans/g1-npu-build02.sh`.
  45min,4 build workers,CPUQuota600%,12GiB,192 tasks; no device execution.
- `g1-npu-build01`: clean132cada Python and standalone cores plus scale consumer
  built,5 CTests passed; terminal exit0/MainPID0/empty cgroup. Builds
  `g1-npu-python01`, `g1-npu-standalone01`, `g1-npu-scale01`.
- `g1-npu-qualification01`: terminal queue timeout after120.53s, no free device;
  **no candidate device code executed**. Planned64 module and8 training checks
  did not run. `runs/g1-npu-qualification01/queue.json` preserves receipt.
  Do not call it a correctness failure. After observed external process exit on
  physical13, a deliberate first-execution gate `g1-npu-qualification02` is
  ended after35.50s without allocation on the same clean132cada source/builds.
  No device execution; receipt preserved. Further unchanged queue retries are
  not planned. A subsequent read-only inventory again showed13 free, so brief
  external availability does not establish a stable qualification window.
- `g2-component-smoke03`: terminal exit0;6 independent tiny CPU entry smokes passed from
  a new frozen source, now using norm32 Read and mandatory trace validation.
  5min,2 CPUs,4GiB,128 tasks; these timings are not selection evidence.
- `g2-component-smoke02`:6 small CPU processes (Read/Add/Attention × replay/batch),
  rows2,width8,parity+trace passed. Smoke timings are not selection evidence.
  Latest script requires actual accelerator kernel trace events, records stage
  and kernel durations/counts/digests, and rejects fallback. G2/G5 actual
  medium/resident/selection experiments remain unexecuted.

## Prepared current-source NPU correctness (not launched)

`plans/g4-npu-modules01.{sh,py}` runs the complete named eager module suite,
Python/native × FP32/FP16, from clean ecafdf4 with byte-verified bda25de cores
(`g3-npu-core-{python,standalone}-clean01`; identical current core hash).
It covers all built-in State/fiber profiles and Read modes, retains explicit
unsupported CSR rejection, and adds the standalone FP32 module checker.
Current-source resident/multi-card and complete standalone training remain
separate gates. One NPU; fresh60s maximum allocation wait/2700s execution,
50min service,CPUQuota600%,MemoryMax12GiB,TasksMax192. Launch after a fresh
inventory supports an allocation opportunity; no unchanged queue retry.

## Fresh finite performance tranche (prepared, not launched)

`plans/g2-cpu-assessment01.{sh,py}` declares10 serial fresh-process cases:
Read norm32/Add/Attention replay versus batched components (rows32,width128),
then standalone CPU LibTorch Add/Attention continued training before/after G1.
Components: clean ecafdf4. Complete consumers: clean c0ce4ee baseline and a02c18c
candidate; exact matching old/new core and installed client hashes are checked.
Medium packets in `TASK/packets/g2-medium-{add,attention}01`: D128,B32,T4,V257,
128 body nodes/544 edges;8,995,632 /17,384,240 parameters. TimedDAG prefill,
SGD/Add or AdamW/Attention, two connected windows, one warmup plus one measured
step; four native workers/one ATen thread; physicalB32, explicit aggressive
operator chunk policy and6GiB static memory admission. No timed CPU oracle.

Fresh budget:120s/case,8GiB per supervised process group,2GiB output cap,
30min service,8-CPU affinity,CPUQuota800%,MemoryMax12GiB,TasksMax192. Stop on
first failure/bound; no repeats or timeout increases. Use existing durable
records/Trackio off. Launch only after the three current build/qualification
jobs terminate and CPU qualification03 passes, avoiding internal timing
contention. Single observations with external load, not stable distribution or
full-size selection. An NPU tranche will be declared separately after allocation
and correctness; no old queue or full matrix is reopened.

## Environment and evidence limits

Public NPU module `libtorch-npu/2.10.0-cann9.0.0`, Python under
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311`. Standalone SDK
is separate from wheel runtime. This explicit user stack supersedes old guide
paths. Private CUDA module `~/privatemodules/torch-cuda/2.10.0-cu128`, toolkit
12.8.1 under `/mi/data2T/zlong/gpu-toolchains`; explicit80/90/100 architectures.
No GPU here; drivers unchanged. Last disk check root64GiB/data80GiB available;
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
