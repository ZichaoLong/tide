# Current handoff

Updated 2026-10-02. **ACTIVE: user resumed and accepted overhead reduction.**
Continue the overall goal; commit/push authorized. No current pause instruction.
No subagents. Reference repositories and ObsidianVault are read-only.
Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
[execution-flows.md](execution-flows.md) is the contract; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog and remains incomplete.

## Contract and priorities

Candidates independently consume common inputs/parameters/initial state, never
CPU reference events/routes/results/gradients. General online greedy permits legal
feedback/input and natural streaming degeneration. Preserve int64, stable ordering,
parallel-edge identity, missing/zero messages, None/zero gradients. Matrix: PDG
LibTorch; TimedDAG/Settle LibTorch+PyTorch; CPU/NPU × streaming/prefill × inference/
complete training. Python resident is a C++/CANN client, not independent PyTorch
scheduling. Five presets retain fine switches. FP32 main, FP16 separate; CUDA
execution remains target-machine-pending. Training includes loss interface,
backward/VJP, optimizer, continuation and throughput; downstream convergence excluded.
Current alignment outranks run-ml-experiments; use minimal existing records.

Next priority: public scale consumers, then bounded representative/full-size F6
five-preset screening. Reuse qualified APIs and atomic updates; do not continually
polish internal fragments. Use affected checks and byte-verified terminal builds,
not repeated unchanged 8,954 CPU checks. Profiling guides implementation.
Implementation commit → immutable qualification → evidence commit; push each.

## Latest verified work

Public multi-device training implementation **0d7c45eb417e827f6bc77c1ee9f876d9663ab215**
is committed/pushed. Its ten fixed-source jobs are now terminal PASSED and audited:
[report](evidence/public-sharded-training-20261002.md),
[audit](evidence/public-sharded-training-20261002.json). Two isolated runtime builds,
CPU25 checks, C++ FP32/FP16 each32 trajectories/512 windows/128 updates, four explicit
owner/card-count transitions, Python41 cases/no skips, installed two-card C++ client,
legacy single-card FP16 cache training, separate two-card profile. No tolerance change.

Public C++/Python loss roots, owner-local state/KV/gradients, canonical parameters,
separate backward/atomic SGD or AdamW, actual forward publication, continued windows
and portable schema1 restore across placement/card count/legacy single owner verified.
Initial model may explicitly stay CPU; no coordinator forward state/KV replica and
no CPU-reference prepass. Public struct ABI changed: rebuild clients and bindings.
Profiling found16 AiCPU boolean scatters; integer flags/exact cast removed fallback.
Clean trace:17,690 AI_VECTOR_CORE/256 AI_CORE/302 MIX_AIV, no observed AiCPU;
112 model executes/816 matching device notify pairs/1,909 switches/7,063 DMA.
Includes setup/checks/refusals, not throughput. Eight development failures preserved.

Prior compact retained state/cache reverse and device completion chain qualified at
49541be ([evidence](evidence/device-state-reverse-20261002.md)); single persistent
CANN task buffer overflow507002 was fixed by one device-chained program per window.
Earlier forward/canonical/internal qualifications remain in ROADMAP and evidence.

## Current work and next action

Public qualification source: TASK/sources/public-shards-clean01 (clean0d7c45e).
Builds: TASK/builds/public-shards-clean01 and public-shards-python-clean01.
All public-shards-*-clean01 jobs passed, including installed consumer and legacy.
Audit command: python TASK/launchers/public_shards_evidence.py
0d7c45eb417e827f6bc77c1ee9f876d9663ab215. Evidence-only commit/push is next.
No new build/test job remains live. Historical CPU Attention stays paused.

Then implement actual consumers under tools/online_bench (currently empty) using
public APIs and common independently consumed parameter/input packet. Inspect
cpp/scale/model.cpp, scripts/flow_topology.py and current native/Python interfaces.
The wide v1 packet has per-edge D×D projections; match real models and parameter
counts rather than calling approximate counts equivalent. v1 state_boundary says
empty state per window: extend/version the packet for continuous operation without
reinterpreting historical evidence. Preserve general topology/online scheduling.
No new full-size CPU/mixed/resident throughput claim has been established.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows; RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service; RUN/status.json/task.log own lifecycle.
Module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Public /opt stack supersedes dated personal guide by user authorization.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0; preserve module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392. Use leased logical devices only.
Last space: data224GiB/root14GiB; heavy writes go under TASK, recheck capacity.
Core builds: placement-cpu-clean01,placement-npu-clean01(standalone),
placement-npu-python-clean01(Python-owned). Never mix SDK and Python runtimes.
Existing builders: launchers/build_public_shards_v3.py, build_public_consumer.py;
recursive header dependencies, byte-verified objects, fresh links; not full vendor
rebuilds. Builds read frozen source cwd. Device/test commands use env -C RUN and
absolute source paths: vendor fusion_result.json must not contaminate the snapshot.
Use freeze_run.py, frozen sources, background.slice/Nice10, queue120s/run600s/build900s.
Atomic handoff writes via scripts/durable_records.py.

## Preserved historical boundaries

Earlier24 dirty files are SHA256-verified on pushed archive/restricted-flow-20260930
at964bf628c67270200dabe55b1bca026bd403cd37; TASK/restricted-flow-archive.json.
Historical historical-cpu-attention-01 remains intentionally SIGSTOP; pause.json
outweighs running status and retains host memory/TASK/timing.lock. Do not resume or
terminate it while developing the new consumers.
Historical Add CPU78.793172/NPU4 47.932888ms/token is throughput1.6438× faster;
it does not certify the new resident path. No full CPU Attention training ratio.
