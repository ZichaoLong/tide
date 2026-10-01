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

Continuous public consumers implementation **fe2d8869f2c0bdb5f59da525a7687d91e341fc78**
is committed/pushed. All five fixed-source jobs PASSED and were audited:
[report](evidence/online-consumers-20261002.md),
[audit](evidence/online-consumers-20261002.json). CPU120 checks, NPU18 directed
FP32 trajectories, no skips or tolerance changes. Standalone CPU includes26
trajectories/104 windows/52 updates; Python/native/LibTorch NPU includes72 windows/
36 updates. Two fresh independent installed client builds reuse unchanged,
byte-verified core. Complete actual parameters, gradients/None, continuations and
independent v2 input generation compared. Separate actual attention training trace
records192 AiCPU tasks (72 int64 Sort,120 bool ScatterElements),18,927 vector,
1,712 AI_CORE,2,974 MIX_AIV; setup included, no throughput/bottleneck-share claim.

No consumer qualification job remains live. Two development build failures
(missing tide/kernel.h declaration include) retained; fixed-source builds pass.
Frozen source sources/online-consumer-clean01; builds online-consumer-{cpu,npu}-clean01;
runs build-online-consumer-{cpu,npu}-clean01,online-consumer-cpu-clean01,
online-consumer-mixed-clean01,online-consumer-profile-clean01. All source fe2d886.
Audit reproduction: python "$TASK/launchers/online_consumer_evidence.py" fe2d8869f2c0bdb5f59da525a7687d91e341fc78.

HARD slot-affine resident training implementation **96c75f8d9c2bee54a5000f4c410fe3d5764ec552**
committed/pushed and all eight immutable qualification jobs PASSED:
[report](evidence/resident-emission-training-20261002.md),
[audit](evidence/resident-emission-training-20261002.json). Two isolated fresh links
reuse recursively validated terminal objects/kernels. Two-card FP32/FP16 each16
trajectories (total512 windows/128 updates);3 placement transitions (48/12);
legacy FP16 12 trajectories (192/48);2 broadcast/cache regressions (32/8);
Python10/no skips. No tolerance change. Separate profile:19,022 vector/357 AI_CORE/
288 MIX_AIV,no observed AiCPU; actual emission link/plan/payload16/42/56.
Setup/checkpoint/checks included,not throughput. No emission job remains live.

Frozen source sources/emission-reverse-clean01; builds emission-reverse-clean01,
emission-reverse-python-clean01. Runs build-emission-reverse-{clean01,python-clean01}
and emission-reverse-{matrix,placements,legacy,regression,python,profile}-clean01.
Reproduce audit: python "$TASK/launchers/emission_reverse_evidence.py" 96c75f8d9c2bee54a5000f4c410fe3d5764ec552.

Emission evidence **7b76dd9** committed/pushed. Active uncommitted work is actual
public resident consumer integration in tools/online_bench/{resident*,host.py,
consumer.h,config.cpp,run.cpp,CMakeLists.txt,main.cpp}, scripts/flow_resident_options.py
and unified launcher wiring, tests/test_online_resident_{loss,consumer}.py.
HARD FP32 head/loss device output compaction, explicit output VJP, packed boundary
input→embedding gradient and staged head/embedding updates after graph agreement.
Single-device inference and single/multi-device training. CPU reference never
feeds intermediate values. Projection banks/partials remain coordinator-owned;
consumer FP16, compact projection owners, multi-device inference and F6 still pending.

CPU boundary math/optimizer tests4 PASSED. online-resident-python-dev01 failed
forward operator budget (RmsNorm required16MiB but divided budget1MiB); dev02
failed bounded compact reverse-packet allocation. Both preserved. Consumer
config defaults now forward512MiB/backward2GiB (declared limits,not total-memory
admission); trace/capacity/chunk/budgets and locality/memory ownership are explicit
CLI options. dev03 Python8 PASSED with unchanged thresholds: actual three-family
Add/Attention complete training and delayed continuous inference. Standalone
build-online-resident-dev03 PASSED,installed public packages plus unchanged
byte-verified resident/core libraries; new consumer compile/link/loader passes.

Standalone dev03 built but actual resident execution failed (retained eager builtin
kernel handles from fixture configuration). Matrix dev04 failed after6 Python
passes; profile-dev04 independently caught the same startup refusal. These remain
failed. Fixed only the known builtin consumer fixture: clear eager handles and
let resident construct the declared graph modules. No generic custom-kernel bypass.

build-online-resident-dev05 PASSED. CPU consumer build and affected59 CPU checks
PASSED (online-consumer + boundary loss/optimizer). Development gates terminal: matrix-dev05 PASSED18 (12 actual model training
trajectories +4 continuous inference +2 CLI success/refusal paths); mixed-dev05
PASSED18; CPU-dev04 PASSED59. int64-dev06 PASSED1 at coordinates beyond2^54
with vocab17,independent integer/gradient reference. No skips or tolerance changes.
Profile-dev05 preserved as failed: D32 fiber reverse workspace refused16-row
physical batches. Same actual workload preflight PASSED with explicit reverse rows4;
no logical batch/loss/window/update change. Automatic budget-aware reverse chunk
selection remains part of scale admission. No consumer profiling conclusion yet.

No current consumer job is live. Next commit implementation,then freeze
online-resident-clean01 at that exact commit. Build CPU with
launchers/build_online_consumer.py --backend cpu --name online-resident-cpu-clean01;
standalone resident with build_online_resident.py --build builds/emission-reverse-clean01
--out builds/online-resident-clean01. Reuse byte-verified unchanged libraries;
these are new installed-client compiles/links,not a backend source change.

Fixed-source gates: CPU tests/test_online_consumer.py +test_online_resident_loss.py
(expect59); NPU tests/test_online_resident_consumer.py (expect19,2 leased cards,
TIDE_BUILD_DIR=placement-npu-python-clean01,TIDE_RESIDENT_LIBRARY=emission-reverse-python-clean01,
TIDE_ONLINE_BINARY=new resident consumer,TIDE_ONLINE_DEVICE=npu:0); mixed18 via
existing test_online_consumer_npu.py. Separate profile_resident_consumer.py --preset
resident (no --development),2 cards. It uses D32 actual Attention,4 windows/2 updates,
trace512,reverse rows4. Source/status/log/hashes and failed development runs must
be audited; evidence-only commit. Queue120/run600/build900 and2 build workers.

Continue compact projection owners,total-memory admission,safe chunking,FP16 consumer
and representative/full-size F6; do not end the task at small consumer qualification.

Historical CPU Attention remains deliberately paused. No formal new full-size
throughput established. Keep focus on complete public flows; affected gates first,
not repeated full unchanged suites or endless internal component polishing.

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
