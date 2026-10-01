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

Public training evidence **b4aa8c1** committed/pushed. All ten clean0d7c45e jobs
passed and audited; no outstanding public training qualification job.

Active development: continuous v2 workload packet and common public scale consumers
under tools/online_bench (Python/native adapter/independent installed C++). Real
per-edge D×D projections, Add/fiber Attention, embedding/head and exact materialized
parameter count. Shared named CPU initializer, independently consumed inputs, full
loss/backward/finite/optimizer and continuous windows with explicit update detach.
Legacy v1 packet remains reset-window. Settle Python embedding now preserves body
owners directly without constructing a second full body. Uncommitted files belong
to this increment; affected tests/builds are next.

Important discovered gap: current resident reverse admits broadcast emission only;
wide packets use slot_affine. Scale resident training/emission parameter sharding
must be implemented and qualified; existing public tests do not certify this model.
No substitution of broadcast or approximate parameter counts. CPU/mixed consumers
first expose the actual workload and make this prerequisite concrete.

Development results: online-consumer-python-dev01 PASSED93 CPU checks;
build-online-consumer-{cpu,npu}-dev02 PASSED; online-consumer-cpp-dev02 PASSED24
independent standalone CPU FP32/FP64 trajectories; online-consumer-mixed-dev02
PASSED18 NPU FP32 complete training cases, independent Python/native/LibTorch,
three families/two memories/both schedules/three mixed presets, strict gradients
and updated parameters. No tolerance change. First CPU/NPUdev01 builds failed on
missing tide/kernel.h include; both failures retained, fixed in dev02.

Final development source online-consumer-dev03: CPU build PASSED; combined
CPU consumer/packet/Settle gate PASSED120 checks, no skips, including independent
standalone FP32/FP64, delayed arrivals, unified CLI, preserved failure output and
bounded diagnostics. No numerical changes after the18-case NPU gate; subsequent
C++ changes only simplify the equivalent dtype guard/add pre-allocation diagnostic
capacity refusal. All implementation development jobs are terminal. Next commit
this coherent implementation and push, then qualify an immutable clean snapshot.

Fixed-source plan: sources/online-consumer-clean01; independent CPU/NPU consumer
builds online-consumer-{cpu,npu}-clean01; online-consumer-cpu-clean01 (120 checks),
online-consumer-mixed-clean01 (18 NPU FP32 trajectories), separate bounded profile
of actual edge-affine attention training. Every job has build600s/run600s/queue120s
bounds. Only affected checks; no full-size or formal throughput job. External
builder launchers/build_online_consumer.py installs unchanged byte-verified core
and compiles public-header-only clients. Runtime commands use task output cwd.
Historical CPU Attention remains deliberately paused. No formal new full-size
throughput has been established. Next: consumer equivalence/continuation gates,
commit implementation, fixed-source qualification/evidence, then emission resident
support and F6 screening. Do not endlessly expand internal component gates.

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
