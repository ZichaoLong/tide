# Current handoff

Updated 2026-10-02. **ACTIVE; continue autonomously.** User authorized continued
implementation, commits and pushes. No pause instruction; no subagents.
Repository /home/zlong/llm/graph-execution-foundation resolves to
/var/tmp/zlong-graph-execution-foundation/repository, branch graph-execution-foundation.
Implementation48e44b0 committed/pushed and fully qualified; reviewed evidence is
ready for its separate commit. No uncommitted production code.
Reference repositories and ObsidianVault remain read-only.
[execution-flows.md](execution-flows.md) is authoritative; [ROADMAP F1–F7](ROADMAP.md)
is the sole backlog. Overall goal incomplete.

## Contract

Candidates independently consume common inputs/parameters/initial state; never
reference events/routes/results/gradients. General online greedy covers legal
family topology/input including positive-delay PDG feedback. Preserve int64,
stable order, duplicate edges, missing/zero messages, None/zero gradients,
complete continuation and explicit differentiation boundaries. Performance:
PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch; CPU/NPU × streaming/prefill ×
inference/complete training. Five presets plus fine switches. FP32 main, FP16
separate. Python resident is a native C++/CANN client. No convergence requirement.
Implementation commit → immutable affected qualification → separate evidence
commit; push each. Contract outranks run-ml-experiments; reuse minimal records.
Do not rerun unchanged8,954 CPU checks. Own heavy timings are serial.

## Latest qualified increments; do not repeat

- Training diagnostics:ca26b47/af263b5, public51/four component cells.
- Eager sample chunks:e6cc52b/e05fe80, CPU60/NPU38.
- Device gradient accumulation:830904b/a3c8d4b, Python21/native384windows.
- Device continuation switching:c96ebcd/d2a425d, Python30/native768windows.
- Resident sample slicing:75543a7/8d97736, CPU15/NPU45; actual representative
  Attention whole→B2 allocator2.244→1.840GiB(-18.0%),0AiCPU profile.
- Optional compact snapshots:f8cc052/ca54c4a, Python18/four native cells;
  saved tensor storage144,506,048→2,755,300bytes,0AiCPU profile.
- **Per-device total live context pool48e44b0**, clean source
  TASK/sources/context-pool-clean01. All eight jobs PASSED:
  build-context-pool-{standalone,python,consumer}-clean01,
  context-pool-{cpu,npu,components,profile,memory}-clean01.
  CPU17,NPU64,no skips; four dense/compact FP32/FP16 native cells,
  32trajectories/768windows/96updates. Independent actual two-card consumer trace
  46,432Vector/1,797AI_CORE/607MIX_AIV,0AiCPU. Same AttentionD128/B8/T4/V257,
  fourB2 slices,oneFP32AdamW/two connected windows: dense allocator1,975,361,024→
  compact1,868,793,344bytes(-5.39%); saved state144,506,048→2,755,300bytes.
  Same loss5.612767696380615/3145events/64outputs/cut,within estimates.
  Audit launchers/context_pool_evidence.py passed.
  [Evidence](evidence/resident-context-pool-20261002.md).

Pool flag --resident-context-bytes0 keeps dense. Positive enables compact
snapshots, caps all simultaneously saved tensor/index bytes on EACH logical NPU,
and separately reserves packing workspace. New handle.device_bytes and optional
snapshot_device(...,device_budgets={index:bytes}) check all cards before payload
copies. Empty/omitted mapping keeps total-only API; older C++ overloads preserved.
Offline plan_execution_flow.py now accepts sample and pool controls too.
Dynamic nonzero extents synchronize at explicit detached snapshot boundaries;
no payload/index/length-vector export, no per-event host scheduling. Samples
retain logical batch/global IDs/loss denominator/one shared update. FP32 and
FP16 numerical policies unchanged. Live KV and retained tapes remain dense.
This is memory calibration, not new-source throughput or original-wide evidence.

## Finite representative matrix — five children remain

Old matrix-remaining01 CANCELLED at completed-child boundary; all24 completed
children retained. Do not resume it or modify retained measurements.
Current parent tide-execution-flows-matrix-remaining02.service started04:24UTC;
TASK/runs/matrix-remaining02/{status,queue,sequence}.json and task.log.
Frozen80dae6e14d41614d0cdb1056bb39b57ca10d07ed at TASK/sources/fiber-append-clean01.
Lease physical1→npu:0,queue cap120s,parent5400s deadline05:54UTC,child900s.
Helper launchers/remaining_family_matrix_resume.py schedules exactly10 missing
children, stops on first failure, no completed-cell reruns. Five now passed:
TimedDAG/Python streaming-confirm02/03 and Settle/Python prefill-screen01,
confirm01/02. RecipePID124125/start1850121793 resumed05:17UTC after pool gates;
boundary-hold-context-pool.json and old compact hold are both resumed.
Current child: Settle/Python/prefill-confirm03. Then streaming screen01 and
confirm01–03; expected to finish within parent deadline if no further long holds.
Do not run heavy builds/gates during measured children. If a hold is necessary,
STOP only the exact recipe, let current measured child finish naturally, then
work. Check PID/start identity before CONT. Deadline is not extended by STOP.

Each submatrix20pilot+3×12confirmation processes,1continued warmup+3measured steps,
2windows/64tokens/step,FP32. LibTorchCPU16packed/mixed4packed; Python default host
policy; ATen/BLAS1. Eight of ten required family/client/schedule submatrices are
already committed, including five inca54c4a. Only Settle/Python both schedules
remain. After terminal children, use launchers/family_matrix_evidence.py with
--prefix matrix-settle-python-prefill --prefix matrix-settle-python-streaming
--output NEW; review and commit remaining matrix evidence. No measurement reruns.

## Next work and environment

Finish evidence commit/push for48e44b0, then prioritize staged scale execution
and retained/reverse memory. Original-wide inference now has feasible static
plans with explicit compact pools; validate larger shapes before original-wide.
Prepare only a bounded staged job and avoid interfering with the running matrix.
Training still needs reductions/calibration: padded retained journals and reverse
arenas plus parameter/gradient copies dominate, beyond saved contexts. Investigate
valid-prefix journal retention (different windows may have different extents),
parameter-bank lifetime sharing, and complete per-card admission. Never loosen
estimates without an allocation/lifetime derivation and independent validation.
Automatic sample admission, eager mixed multi-card placement and actual wide
execution/comparisons remain open. F7 final migration/evidence audit pending.

TASK=/mi/data2T/zlong/tide-execution-flows. Public module
libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized /opt stack supersedes dated personal guide. Preserve module
PYTHONPATH,prepend frozen source/python. TASK_QUEUE_ENABLE=0,
TORCH_DEVICE_BACKEND_AUTOLOAD=0.16logical64GiB Ascend910_9392;lease/remap only.
freeze_run.py uses detached background.slice/Nice10,2buildworkers,bounded tasks.
Last disk free:data200GiB/root12GiB. Current timing lock TASK/online-measurement.lock.
Atomic handoff writes use durable_records.replace_text.
Qualified libraries TASK/builds/context-pool-{standalone,python}-clean01.
Qualified installed consumer TASK/builds/context-pool-consumer-clean01/consumer/tidegraph-online-bench.
Core builds placement-{cpu,npu,npu-python}-clean01; eager binaries
sample-chunks-{cpu,npu}-clean01/consumer/tidegraph-online-bench. Distinct runtime
owners; source/header/options-checked object reuse only. New builds passed loader
closure. Do not modify any frozen source or active helper.

Original wide packets TASK/inputs/fullsize-{add,attention}01/workload.json:
480body/2208edges,D2048/B512/T12/V50304,9,468,053,696 or17,521,117,376params,
clear=true/stride17. Region(budget) defaultsobserve_all=true: unselected nodes
retain KV; clear applies only to selected nodes. Do not change logicalB512 or
lose KV and claim original-wide passed. Static estimate refusal is not physical
impossibility or completion. CUDA real hardware and other CANN tuples target-pending.

Historical historical-cpu-attention-01 intentionally remains SIGSTOP and holds old
timing.lock; never resume/kill/clean it. Historical1.6438× means faster throughput,
not current online-flow evidence. Prior restricted work remains on pushed
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
