# Current handoff

Updated 2026-10-02. **ACTIVE; continue autonomously.** User authorized continued
implementation, commits and pushes. No pause instruction; no subagents.
Repository /home/zlong/llm/graph-execution-foundation resolves to
/var/tmp/zlong-graph-execution-foundation/repository, branch graph-execution-foundation.
Latest implementation committed/pushed75543a7; its evidence is ready to commit.
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

## Latest completed increments; do not repeat

- Training VJP records separated from optional diagnostics:ca26b47/af263b5;
  public51, four component cells,2,352windows/588updates.
- Eager CPU/mixed physical sample chunks:e6cc52b/e05fe80; CPU60/NPU38.
- Resident device gradient accumulation:830904b/a3c8d4b;
  C++384windows/48updates,Python21, independent trace0AiCPU.
- Device continuation switching:c96ebcd/d2a425d; C++768windows/96updates,
  Python30, independent trace0AiCPU. Same-live-owner snapshots preserve complete
  NPU numerical continuation; parameters/optimizer/accumulator shared, no tape.
  Detached boundaries only; snapshots remain dense.
- Resident consumer sample slicing:75543a7. Five clean jobs PASSED at
  TASK/sources/resident-samples-clean01: build-resident-samples-consumer-clean01,
  resident-samples-{cpu,npu,profile,memory}-clean01. CPU15/NPU45,no skips.
  Independent two-device trace46,281Vector/1,797AI_CORE/586MIX_AIV,0AiCPU.
  Same representative Attention D128/B8/T4/V257 one-update/two-window peak:
  whole2.244GiB → chunk2 1.840GiB(-18.0%); loss/events/outputs64/cut agree,
  allocator within estimates. Memory calibration only, not throughput/wide.
  Audit launchers/resident_samples_evidence.py passed. [Evidence](evidence/resident-sample-chunks-20261002.md).

Sample slicing preserves logical batch/global sample IDs/loss denominator and
one shared graph+embedding/head update. Connected windows stay in one backward;
independent slices accumulate at detached boundaries. Absent tail capacity has
no events. Capacity charges dense saved contexts and simultaneous accumulators.
Explicit maximum only, no automatic sample-size choice.
FP32 elementwise1e-6/1e-5 unchanged; FP16 independent CPU norm policy .002+.02*ref
infinity-norm plus original whole-vs-sliced FP16 elementwise .002/.02. Initial
8 near-zero FP16 threshold failures reproduced on whole FP16; production code
unchanged during policy correction. Retained failures/skips are not qualification.

## Finite representative matrix — running only missing children

Old parent tide-execution-flows-matrix-remaining01.service was CANCELLED at a
completed-child boundary, before its04:28UTC deadline. All24 completed child
results preserved; no measurement interrupted. The last was TimedDAG/Python/
streaming-confirm01. resumption.json/sequence.json retain reason and successor.
Do not resume this old recipe or modify its retained results.

New parent tide-execution-flows-matrix-remaining02.service started04:24UTC;
TASK/runs/matrix-remaining02/{status,queue,sequence}.json and task.log.
Frozen80dae6e14d41614d0cdb1056bb39b57ca10d07ed at TASK/sources/fiber-append-clean01.
Single lease physical1→npu:0, queue cap120s, parent timeout5400s, child900s.
Helper launchers/remaining_family_matrix_resume.py schedules exactly10 missing
children: TimedDAG/Python/streaming-confirm02/03, Settle/Python/prefill and streaming
screen01 +confirm01–03. Stops on first failure; do not edit live helper/source.
Current first child streaming-confirm02; inspect sequence for latest status.
No completed-cell reruns. Heavy builds/gates must wait for a measured-child boundary.

Each submatrix20pilot+3×12confirmation processes,1continued warmup+3measured steps,
2windows/64tokens/step,FP32. LibTorchCPU16packed/mixed4packed; Python default host
policy; ATen/BLAS1. Audits use launchers/family_matrix_evidence.py.
Already reported: TimedDAG/LibTorch/prefill8695804, PDG/LibTorch bothaf263b5.
Audited, reports pending: TimedDAG/LibTorch/streaming and Settle/LibTorch both
(TASK/libtorch-matrix-extra-audit.json), TimedDAG/Python/prefill
(TASK/python-prefill-matrix-audit.json). Preserve measurement source80dae6e.

## Next work and environment

First commit/push current sample-slicing evidence, then reduce dense saved/live
KV and retained/reverse/journal costs with general device packing, explicit
capacity admission and small independent semantic gates. Mixed multi-device
parameter/payload placement and automatic eager admission remain. Actual original
wide execution, bounded full-size comparisons and F7 are pending. CUDA real
hardware and other CANN tuples remain target-machine work.

TASK=/mi/data2T/zlong/tide-execution-flows. Public module
libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized /opt stack supersedes dated personal guide. Preserve module
PYTHONPATH, prepend frozen source/python. TASK_QUEUE_ENABLE=0,
TORCH_DEVICE_BACKEND_AUTOLOAD=0.16logical64GiB Ascend910_9392; lease/remap only.
freeze_run.py uses detached background.slice/Nice10,2buildworkers,bounded tasks.
Last disk free:data201GiB/root12GiB. Current timing lock TASK/online-measurement.lock.
Atomic handoff writes use durable_records.replace_text.

Qualified eager binaries TASK/builds/sample-chunks-{cpu,npu}-clean01/consumer/tidegraph-online-bench.
Resident libraries TASK/builds/contexts-{standalone,python}-clean01.
Latest consumer TASK/builds/resident-samples-consumer-clean01/consumer/tidegraph-online-bench.
Core builds placement-{cpu,npu,npu-python}-clean01. Distinct standalone/Python
runtime owners; source/header/options-checked reuse only.

Original wide packets TASK/inputs/fullsize-{add,attention}01/workload.json:
480body nodes/2208edges,D2048/B512/T12/V50304,9,468,053,696 or17,521,117,376params.
Do not reduce logicalB512 and claim wide passed. Region(budget) defaults
observe_all=true: unselected nodes retain KV; clear only clears selected nodes.
Read-only dense estimate still exceeds12-card Attention memory; estimate refusal
is not physical impossibility or completion. Need implement/measure general savings.

Historical CPU job historical-cpu-attention-01 deliberately remains SIGSTOP,
holds old timing.lock; never resume/kill/clean it. Historical1.6438× means faster
throughput, not current online-flow evidence. Prior restricted work preserved on
pushed archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
