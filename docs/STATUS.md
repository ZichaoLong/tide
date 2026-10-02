# Current handoff

Updated 2026-10-02. **ACTIVE; continue autonomously.** User authorized continued
implementation, commits and pushes. No pause instruction and no subagents.
Reference repositories and ObsidianVault are read-only. Repository
/home/zlong/llm/graph-execution-foundation resolves to
/var/tmp/zlong-graph-execution-foundation/repository, branch graph-execution-foundation.
Implementatione6cc52b is committed/pushed; sample-chunk evidence is being committed.
[execution-flows.md](execution-flows.md) is authoritative; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog. Overall goal is incomplete.

## Contract

Each candidate independently consumes common inputs/parameters/initial state;
never reference events/routes/results/gradients. Online greedy accepts legal
arbitrary topology/input, including positive-delay PDG feedback, with natural
streaming fallback. Preserve int64, stable order, duplicate edges, missing/zero
messages, None/zero gradients and full continuation. Performance: PDG LibTorch;
TimedDAG/Settle LibTorch+PyTorch; CPU/NPU × streaming/prefill × inference/complete
training. Five presets plus fine switches; FP32 main, FP16 separate. Python
resident is a native C++/CANN client. Training means loss/VJP/optimizer/continuation
and throughput, not downstream convergence. Contract outranks run-ml-experiments;
use existing minimal records. Implementation commit → immutable affected
qualification → separate evidence commit; push each. Do not rerun unchanged8,954
CPU checks or edit running jobs' frozen source/helpers. Own heavy timing is serial.

## Newly completed; do not repeat

1. Training VJP records separated from optional diagnostic exports, implementation
   ca26b472277eb9282509733aecaffe2b1893cc3c; evidenceaf263b5 is pushed.
   Source TASK/sources/training-records-clean01. Six qualification jobs PASSED:
   build-training-records-{standalone,python,consumer}-clean01 and
   training-records-{components,public,allocator}-clean01. Public51,no skips;
   four component cells cover83FP16 single-device and32sharded trajectories per
   dtype,2,352windows/588updates. Required VJP journals remain; public ABI and
   kernels unchanged. Same representative Attention allocator peak decreases
   2,426,168,320→2,409,123,840bytes (16.26MiB,0.70%); exact loss/work/output/cut.
   No throughput claim; conservative admission envelope unchanged.
   [Evidence](evidence/resident-training-records-20261002.md).
   Initial observer-dev02 failed because its CPU oracle disabled trace; corrected
   dev03 and clean gate pass. Failure retained, no numerical tolerance changed.
2. Physical sample chunks for eager Python/native/LibTorch CPU/mixed consumers,
   implementatione6cc52b07d33e5be33054f43bc938dbe1f46a0da is pushed.
   Source TASK/sources/sample-chunks-clean01. Five clean jobs PASSED:
   build-sample-chunks-{cpu,npu}-clean01, sample-chunks-{cpu,npu,memory}-clean01.
   CPU60/NPU38,no skips; development same totals passed. Global sample IDs and
   whole-batch denominator, per-slice continuation, accumulated gradients/None
   flags, one finite agreement/optimizer update; B5/chunk2 tail coverage.
   --sample-chunk-rows0 preserves whole batch. Explicit maximum only; automatic
   eager admission and resident sample slicing are not implemented. Resident
   rejects nonzero requests. Core/resident ABI/CANN libraries unchanged.
   Four separate LibTorch memory observations, D128/B16/T8/V257 Attention,
   one two-window FP32 AdamW step, CPU16packed/mixed-a4packed,ATen/BLAS1:
   CPU whole/chunk4 RSS1,043,226,624→632,844,288bytes (-39.3%);
   mixed host2,135,420,928→1,748,725,760 (-18.1%);
   mixed NPU allocator292,252,160bytes unchanged. Loss/work/output/cut checks pass.
   One process per configuration: not throughput recommendation or full-size.
   [Evidence](evidence/consumer-sample-chunks-20261002.md).
   Audit launchers/sample_chunk_evidence.py passed; no remeasurement needed.

Earlier clean evidence: complete capacity0b1a5aa, host controls2222d9d,
append-only fiber KV80dae6e/evidence9be8e52. Append staging saved128.001MiB on D128;
see [fiber append](evidence/resident-fiber-append-20261002.md). Immutable parameter
projection retention, canonical streaming/publication, optimizer recomputation,
FP16/sharded training/checkpoint all remain indexed in ROADMAP. They do not imply
original wide size is solved.

## Active finite representative matrix

TASK=/mi/data2T/zlong/tide-execution-flows.
Fixed source TASK/sources/fiber-append-clean01 at clean80dae6e14d41614d0cdb1056bb39b57ca10d07ed.
Measurements predate both new implementations; claims stay tied to that source.
Parent unit tide-execution-flows-matrix-remaining01.service, started01:58:22UTC,
total9000s, child900s, queue120s. Parent leases physical3 as logicalnpu:0.
Records TASK/runs/matrix-remaining01/{status.json,task.log,queue.json,sequence.json}.
Only recipe PID3771316 was held at finished-child boundaries to isolate builds
and gates. Both holds are now RESUMED; /proc start identity1849248082.
boundary-hold.json and boundary-hold-sample-chunks.json record transitions.
No measured sample was interrupted/discarded. Parent timeout continues, so do not
leave a future hold unattended. Do not signal wrappers or historical CPU work.

Helper launchers/remaining_family_matrix.py is a finite34-child recipe. Do not
edit it or family_matrix_screen.py while live. Each missing submatrix has20pilot
processes (five presets×four groups), then3×12confirmations (CPU,screened mixed,
resident). Each confirmation:1continued warmup+3measured steps,2windows/64tokens
per step,FP32,rotated order,own heavy timings serial. LibTorchCPU16packed,
mixed4packed; independent Python default host policy;ATen/BLAS1. Python resident
uses the native client. One-step mixed selection is only a pilot, not optimality.
Every run has source/binary/workload/log receipts and loss/event/output/cut checks.

Completed submatrices:
- TimedDAG/LibTorch/prefill already qualified by host-policy evidence8695804;
  do not rerun. Resident/CPU throughput Add inference.833×/training1.549×,
  Attention inference.855×/training1.126×. Prior traces showed no AiCPU;
  mixed dominated by host launches/sync. CPU wins short-process totals.
- Both PDG/LibTorch schedules:40pilot+72confirmation processes;
  [PDG evidence](evidence/representative-pdg-matrix-20261002.md), committedaf263b5.
  Prefill resident/CPU Add inference.832×/training1.513×, Attention.914×/1.160×.
  Streaming.333×/.877× and.486×/.855×. Resident26prefill vs80streaming stages/step.
  CPU wins all short-process totals. These remain representative, not wide.
- TimedDAG/LibTorch/streaming screen+all3confirmations PASSED, not yet reported.
- Settle/LibTorch/prefill screen+confirm01 PASSED. Read sequence.json for later
  progress; remaining Settle LibTorch and TimedDAG/Settle Python modes continue.

Audit completed groups with launchers/family_matrix_evidence.py --prefix NAME
(repeatable) --output PATH. It already passed the two PDG groups; Python manifest
assertions have not yet been exercised. Do not repeat finished cells. At final
terminal, audit remaining submatrices, write one bounded reviewed comparison and
commit/push evidence. Diagnose a child failure before any further run.

## Source/build environment

Current eager consumers: TASK/builds/sample-chunks-{cpu,npu}-clean01/consumer/tidegraph-online-bench.
NPU resident backends: TASK/builds/training-records-{standalone,python}-clean01.
Core builds: placement-{cpu,npu,npu-python}-clean01. Standalone/Python-owned
runtimes stay separate. Reuse helper build_sample_consumer.py checks exact
consumer sources/options before copying CPU objects and freshly linking;
build_capacity_client.py does source/header/options-checked installed NPU client
reuse. These are not full core/kernel rebuilds.
Public module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized /opt stack supersedes dated personal guide. Default shell Python
cannot run Torch tests. TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0;
retain module PYTHONPATH,prepend snapshot/python. Ascend910_9392,16logical64GiB
chips, only leased/remapped devices. freeze_run.py: background.slice/Nice10,
2build workers, finite bounds, runtime env -C RUN. Last data212GiB/root14GiB free.
Atomic handoff writes use durable_records.replace_text. Timing lock is
TASK/online-measurement.lock, distinct from historical timing.lock.

## Real remaining scale work and next priority

Keep implementation moving while the finite matrix runs; heavy new builds/gates
wait for a known measurement boundary. No new worker search or full-suite loop.
Representative packets screening-representative-{add,attention}01 are128nodes/
544edges,D128/B8/T4/V257,8,995,632/17,384,240parameters. Not original wide.
Full-size packets TASK/inputs/fullsize-{add,attention}01/workload.json already
exist:480nodes/2208edges,D2048/B512/T12/V50304,
9,468,053,696 /17,521,117,376parameters. Never reduce logicalB512 and claim wide.

Eager sample slicing now bounds connected activations, but model/optimizer and
all persistent state remain live. Mixed still has one payload device;17BFP32
cannot fit one64GiB chip. Generic mixed multi-device placement remains open.
Resident still densely reserves batch×local_attention_nodes×KV capacity, retains
state/cache/journals and creates reverse buffers; coordinator journals also grow
with trace×width. Sample slicing needs an owner-level gradient accumulator and
independent continuation switching with a shared parameter generation and one
optimizer update, not separate optimizers per slice. It is not implemented.
Accurate simultaneous-lifetime accounting/compact storage remain important;
conservative admission refusal alone is not proof of physical impossibility.

Read-only finding: wide uses Region defaults observe_all=true. Unselected nodes
adopt/retain KV; clear=true clears selected nodes only. Therefore no clear-only
shortcut can discard unselected KV. Preserve arbitrary legal topology/input,
whole attention visibility/normalization, continuation and None flags. Full-size
inference/training and appropriate finite comparisons still must actually run.
F7 final portable qualification, CUDA real devices and other CANN tuples remain
pending. Historical CPU Attention supplementary is not a blocker.

## Preserved history

historical-cpu-attention-01 remains deliberately SIGSTOP, holds old timing.lock;
pause.json overrides running status. Never resume/kill/clean it. Historical
Add1.6438× means faster throughput but does not qualify these online flows.
Old restricted changes are preserved on pushed archive/restricted-flow-20260930
at964bf628c67270200dabe55b1bca026bd403cd37; TASK/restricted-flow-archive.json.

Last observed matrix current: matrix-settle-libtorch-streaming-screen01; completed14of34recipe children. Re-read live records.
