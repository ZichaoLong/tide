# Current handoff

Updated 2026-10-02. **ACTIVE; continue autonomously.** The user authorized continued
implementation, commits and pushes. No pause instruction; no subagents.
Repository /home/zlong/llm/graph-execution-foundation resolves to
/var/tmp/zlong-graph-execution-foundation/repository, branch graph-execution-foundation.
Latest committed/pushed source c96ebcd. Reference repositories and ObsidianVault are read-only.
[execution-flows.md](execution-flows.md) is authoritative; [ROADMAP F1–F7](ROADMAP.md)
is the sole backlog. The overall goal remains incomplete.

## Contract

Candidates independently consume common inputs/parameters/initial state; never
reference events/routes/results/gradients. General online greedy supports each
family's legal topology/input, including positive-delay PDG feedback. Preserve
int64, stable order, duplicate edges, missing/zero messages, None/zero gradients,
complete continuation and explicit differentiation boundaries. Performance:
PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch; CPU/NPU × streaming/prefill ×
inference/complete training. Five presets plus fine switches; FP32 main, FP16
separate. Python resident is a native C++/CANN client. No convergence requirement.
Implementation commit → immutable affected qualification → separate evidence
commit; push each. Contract outranks run-ml-experiments; reuse minimal records.
Do not rerun the unchanged 8,954 CPU checks. Own heavy timings are serial.

## Completed increments; do not repeat

- Training VJP records separated from optional diagnostics: ca26b47, evidence
  af263b5; public51, four component cells, 2,352 windows/588 updates. Representative
  allocator peak saves16.26MiB/0.70%; no throughput claim.
  [Evidence](evidence/resident-training-records-20261002.md).
- Eager Python/native/LibTorch CPU/mixed physical sample chunks: e6cc52b, evidence
  e05fe80; CPU60/NPU38, no skips. Logical batch/global sample IDs/loss normalization,
  accumulated gradients and one update preserved, including B5/chunk2 tail.
  CPU RSS -39.3%, mixed host RSS -18.1%; mixed NPU allocator unchanged in one
  D128/B16/T8/V257 observation, not throughput.
  [Evidence](evidence/consumer-sample-chunks-20261002.md).
- Resident canonical-device gradient accumulation:830904b, evidencea3c8d4b;
  C++384windows/48updates, Python21pass, independent trace0AiCPU. Explicit detach
  per accumulate; connected windows stay in one backward group. One shared
  optimizer update; no continuation switching or resident consumer slicing yet.
  [Evidence](evidence/resident-accumulation-20261002.md).
- Earlier optimizer recomputation, full-consumer capacity and append-only fiber
  staging are indexed in ROADMAP. They do not establish original wide execution.

## Latest qualified increment

Device continuation switching implementation c96ebcd is pushed. Five clean jobs
PASSED at TASK/sources/contexts-clean01: build-contexts-{standalone,python}-clean01
and contexts-{components,public,profile}-clean01. C++768windows/96updates including
accumulation regression, Python30/no skips. Trace26159Vector/350AI_CORE/363MIX_AIV,
0AiCPU. Audit launchers/contexts_evidence.py passed. [Evidence](evidence/resident-contexts-20261002.md).
Development test-interface failures are retained in public-dev01; dev02 passed.
Do not repeat these gates. Latest resident libraries contexts-{standalone,python}-clean01.
Same-live-owner opaque snapshots preserve complete NPU numerical continuation;
parameters/optimizer/accumulator shared, no tape copied. Detached boundaries only.
Snapshots are dense; resident consumer slicing and compact KV remain pending.

Next: resident consumer sample slicing with full logical-batch loss/global sample
coordinates/one optimizer update. Include all saved context buffers and gradient
accumulators in simultaneous-live capacity estimates, then validate small tails
and multi-window continuation before mid-scale memory calibration.

## Finite representative performance matrix — RUNNING

Parent tide-execution-flows-matrix-remaining01.service, fixed clean80dae6e14d41614d0cdb1056bb39b57ca10d07ed
at TASK/sources/fiber-append-clean01. Started01:58:22UTC, timeout9000s, deadline
04:28:22UTC. Leases physical3 → logicalnpu:0. Recipe PID3771316 (start1849248082)
is RESUMED at03:51:32UTC; all boundary-hold*.json records resumed. Measured child
matrix-timed-dag-python-prefill-confirm03 passed naturally; next missing cells
continue. No measured sample interrupted. Contexts builds/gates/profile finished.
Do not signal wrappers, edit running helpers/frozen source or touch others' jobs.
Read TASK/runs/matrix-remaining01/sequence.json and child status for current state.

Finite34-child recipe launchers/remaining_family_matrix.py; do not modify it or
family_matrix_screen.py while live. Each submatrix:20pilot +3×12confirmation
processes;1continued warmup +3measured complete steps,2windows/64tokens/step,FP32.
LibTorchCPU16packed/mixed4packed; Python default host policy;ATen/BLAS1.
No worker search or completed-cell reruns. Audits use family_matrix_evidence.py.

Completed: TimedDAG/LibTorch/prefill (host-policy evidence8695804); PDG/LibTorch
both schedules (af263b5); TimedDAG/LibTorch/streaming and Settle/LibTorch both modes
(audited TASK/libtorch-matrix-extra-audit.json, report pending); TimedDAG/Python/prefill
all repeats (audited TASK/python-prefill-matrix-audit.json). Missing: TimedDAG/Python/streaming, Settle/Python both.
If outer timeout occurs, retain completed children and resume only missing items.

TASK=/mi/data2T/zlong/tide-execution-flows.

## Environment and safe resumption

Public module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized /opt stack supersedes dated personal guide. Default shell Python
cannot run Torch tests. TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0;
retain module PYTHONPATH and prepend frozen source/python. Ascend910_9392,
16logical64GiB chips; leased/remapped devices only. freeze_run.py uses detached
background.slice/Nice10, two build workers and finite timeouts. Runtime env -C RUN.
Last free space data206GiB/root13GiB. Atomic handoff writes: durable_records.replace_text.
Current timing lock TASK/online-measurement.lock.

Qualified eager binaries TASK/builds/sample-chunks-{cpu,npu}-clean01/consumer/tidegraph-online-bench.
Qualified resident libraries TASK/builds/contexts-{standalone,python}-clean01.
Core builds placement-{cpu,npu,npu-python}-clean01. Separate standalone/Python
runtime owners; never link both. Source/header/options-checked reuse only.

## Remaining scale work

1. Implement resident physical sample
   slicing using one parameter/optimizer owner plus accumulation and independent
   contexts. Preserve full logical-batch loss/global coordinates/one update.
2. Dense live KV, retained/reverse/journal memory; simultaneous-live accounting
   and mid-scale allocator calibration. Snapshot handles add storage.
3. Mixed multi-device parameter/payload placement; currently one eager device.
4. Original wide actual execution, bounded full-size comparisons and final F7.
5. CUDA real devices and other CANN tuples require target-machine qualification.

Representative packets:128nodes/544edges,D128/B8/T4/V257,8,995,632 or17,384,240
parameters. Original wide packets TASK/inputs/fullsize-{add,attention}01/workload.json:
480body nodes/2208edges,D2048/B512/T12/V50304,9,468,053,696 or17,521,117,376parameters.
Do not reduce logicalB512 and claim wide passed.17BFP32 cannot fit one64GiB chip.
Region(budget) defaults observe_all=true: unselected nodes retain KV; clear only
clears selected nodes. No clear-only shortcut can discard other caches.

Historical CPU job historical-cpu-attention-01 deliberately remains SIGSTOP,
holds old timing.lock; never resume/kill/clean it. Historical1.6438× means faster
throughput but is not these online flows' evidence. Restricted prior work is
preserved on pushed archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
