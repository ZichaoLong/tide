# Current handoff

Updated 2026-10-10 (Asia/Shanghai). G1–G5 is active. Continue all locally achievable
implementation, correctness, finite selection evidence and CUDA target handoff.
Commits do not pause work; no subagents. Execution contract: [execution-flows §11](execution-flows.md).
Sole backlog: [ROADMAP G1–G5](ROADMAP.md). Selector/new upstream semantics stay deferred.

## Source and next actions

Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`, branch
`graph-execution-foundation`. Initial clean HEAD c0ce4ee5. Committed/pushed G1
State/Read implementation132cada, accounting/gate fixes8a4d60f, exact Norm VJP
rounding fixa02c18c; G3 CUDA/shared boundary bda25de; component profiler d1df03c;
target capacity/control-policy follow-up ecafdf4 and logical0 peer gate5018133.
Current pending entry hardening checks the compiled consumer source inventory,
not only matching core/resident libraries;6 directed CPU tests and all four
existing CUDA/NPU installed/online source audits pass. Runtime sources unchanged.

`TASK=/mi/data2T/zlong/tide-execution-flows`; paths below use this root.
Sources are frozen while jobs read them. Core source digest remains
`dbc370539ed261846524c427b7b9a37a38f6857ba6a6ccb78ad365168140a150`.

1. Finish inspecting CPU qualification03 and NPU clean build01 below.
2. Once those terminate and CPU qualification passes, launch the declared
   `plans/g2-cpu-assessment01.{sh,py}` serial tranche. No old matrix resumes.
3. Run `python $TASK/plans/audit-g1-g3-clean01.py` only after the three named
   jobs pass. Review its output and commit evidence separately from source.
4. With a genuinely available allocation, run prepared current-source NPU
   modules, related resident/complete-training regression and finite profiling.
   Last inventory: all16 devices have external processes; no unchanged lease retry.
5. Review performance and limits, reconcile support/selection, retain CUDA
   correctness/residency/multicard/performance as real-target pending.

## Live jobs

Inspect `systemctl --user show tide-NAME.service -p ActiveState -p Result
-p MainPID -p ControlGroup`, then `runs/NAME/status.json` and `task.log`.
All current jobs use background.slice, KillMode=control-group, Nice10, immutable
sources, bounded resources and OMP/BLAS1. Never call a live job passed.

- `g1-cpu-qualification03`: live clean a02c18c,
  `sources/g1-cpu-qualification03`, matching reused `builds/g1-norm-dev06`.
  Full CPU FP64/FP32 main suite **9595 passed/654 optional skips,2109.16s**;
  complex topology and installed consumer stages are still running. Results
  `runs/g1-cpu-qualification03/gate`; plan of same name.90min,8CPUs,16GiB,192tasks.
- `g3-npu-clean01`: live clean bda25de from `sources/g3-backends-clean01`.
  Python/standalone cores, resident libraries and installed/online consumers
  built; scale consumer is still building. Plan of same name, builds
  `g3-npu-*-clean01`.90min,4workers,8CPUs,16GiB,256tasks. Build-only; no current
  candidate NPU workload has executed.

## Terminal evidence awaiting reviewed report

- `g3-cuda-clean01`: exit0/MainPID0/empty cgroup. Clean bda25de, private CUDA
  toolkit12.8.1, architectures80/90/100. Core, standalone/Python resident and
  installed/online consumers built;66 standalone manifest binaries,2 Python
  libraries;7 host CTests,102 CPU semantic sources/1CTest and229 CPU pytest
  passed/84 optional hardware skips. `runs/g3-cuda-clean01/host.xml` and
  `builds/g3-cuda-*-clean01`. No GPU executed.
- `g3-cuda-unavailable01`: original wrapper remains **failed exit1**, terminal
  empty cgroup. Clean5018133 full target entry correctly refused missing CUDA
  at its only stage, preflight. Wrapper expected an obsolete error substring.
  No rerun: read-only `runs/g3-cuda-refusal-audit01.json` accepts the retained
  refusal and hashes the failure artifacts. No device correctness claim.
- G1 development gates passed193/1538/324/173/36/378 cases at their scoped
  sources; latest `g1-norm-dev06` covers exact corrected FP32 norm VJP order
  `dy*(x/norm)` before FP64 conversion. No tolerances relaxed.
- Full `g1-cpu-qualification01` failed73/passed9451/skipped723 on132cada:
  69 plumbing/accounting/RSS issues fixed8a4d60f;4 real norm rounding failures
  fixeda02c18c. Qualification02 was cancelled after diagnosis (exit143,empty
  cgroup). Preserve both. Prior NPU builds01/02 passed but predate the norm fix.
- `g1-npu-qualification01/02` ended without allocation after120.53s/35.50s;
  no candidate device code ran. Preserve queue receipts, no automatic retries.
- CUDA development builds/CPU host checks and tiny component smokes are scoped
  development evidence. Earlier failed builds/launch wrappers remain retained;
  no development smoke is full device qualification or selection evidence.

## Prepared NPU correctness; not launched

`plans/g4-npu-modules01.{sh,py}`: clean ecafdf4 source
`sources/g2-components-ecafdf4`, byte-verified current bda25de Python/standalone
cores. Complete named Python/native eager module suite FP32/FP16, independent
CPU references, all State/fiber/Read profiles, observables/VJP/None, updates and
checkpoints, explicit unsupported CSR rejection, operator doctor and standalone
FP32 checker. No resident/multicard/performance claim. OneNPU; allocation60s,
execution2700s,50min service,6CPUs,12GiB,192tasks. Launch only with a fresh
allocation opportunity. Resource-information question pending; independent work
continues. Related resident/complete-training gates remain separate.

## Declared fresh CPU performance tranche; not launched

`plans/g2-cpu-assessment01.{sh,py}` declares10 serial fresh processes:
Read norm32/Add/Attention replay/batched PyTorch components, rows32,width128;
then standalone CPU LibTorch Add/Attention continued training before/after G1.
Components clean ecafdf4; complete baseline clean c0ce4ee and candidate a02c18c,
with byte-checked core/consumer/binary/loader. Packets
`packets/g2-medium-{add,attention}01`: D128,B32,T4,V257,128body nodes/544edges,
8,995,632 /17,384,240 parameters. TimedDAG prefill, SGD/Add or AdamW/Attention,
two connected windows,1warmup+1measured step,ATen1/native workers4. PhysicalB32,
explicit aggressive chunking,6GiB admission. No timed oracle; separate profiling.

Budget120s/case,8GiB process group,2GiB outputs;30min service,8CPU affinity/quota,
12GiB,192tasks. First failure/bound stops; no repeat or timeout increase.
Wrapper requires all three current clean jobs terminal and CPU qualification
passed. Existing records,Trackio off. Single observations under external load,
medium scope only. NPU profiling gets a separate fresh budget after correctness.

## Environment, limits and protected history

NPU module `libtorch-npu/2.10.0-cann9.0.0` supplies the SDK/CANN stack; Python
is explicitly `/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
Python wheel and standalone runtime owners remain separate. CUDA private module
`~/privatemodules/torch-cuda/2.10.0-cu128`, tools under
`/mi/data2T/zlong/gpu-toolchains`. No GPU; no public drivers changed. Last storage
check root64GiB/data101GiB free; recheck before large writes. All16 NPUs currently
have external processes; none touched. For resident NPU retain the recorded
explicit `ACLNN_CACHE_LIMIT=0`/`ACL_OP_INIT_MODE=0` configuration.

State/Read batching is composed VJP work, not proven fused kernels. Custom State/
Read and nondefault native fiber policies retain replay. Structural groups,
causal depth, containers and per-row views/clones remain batching costs. Resident
already uses device VJPs; CUDA's initial one-thread semantic adapter, serial
reductions/scatter and uncalibrated GPU memory are known limitations.

Old F1–F7 qualification and performance remain tied to their cited sources;
61/120 old timings and missing cells are not a new queue. [Terminal record](evidence/selection-terminal-20261010.md).
Preserve the [original strict near-tie failure](evidence/original-add-route-witness-20261004.md):
CPU246 oneFP32ULP ahead; resident245/246 tie selects245; proposal error7.7039e-6,
events2325/2327. No relaxed discrete comparison or universal strict-equivalence claim.

**Never resume, stop, signal or clean historical-cpu-attention-01 / worker2686919.**
Malformed historical build-reverse-gather-python-dev01 can make status.py exit1;
preserve it. No project writes in ObsidianVault; all reference repositories read-only.
