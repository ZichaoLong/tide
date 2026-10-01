# Current handoff

Updated 2026-10-02. **ACTIVE: user resumed and accepted overhead reduction.**
Continue the overall goal; commits/pushes authorized. No current pause instruction.
No subagents. Reference repositories and ObsidianVault are read-only.
Repository /home/zlong/llm/graph-execution-foundation resolves to
/var/tmp/zlong-graph-execution-foundation/repository; branch graph-execution-foundation.
[execution-flows.md](execution-flows.md) is the contract; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog. Overall task remains incomplete.

## Contract and focus

Every candidate independently consumes common inputs/parameters/initial state,
never CPU reference events/routes/results/gradients. General online greedy accepts
legal topology/input including positive-delay feedback and natural streaming
fallback. Preserve int64,stable ordering,parallel-edge identity,missing/zero messages,
None/zero gradients. Matrix: PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch;
CPU/NPU × streaming/prefill × inference/complete training. Python resident is a
C++/CANN client,not an independent PyTorch scheduler. Five presets retain fine
switches; FP32 main,FP16 separate. CUDA real execution remains target-machine-pending.
Training means loss/VJP/optimizer/continuation/throughput,not downstream convergence.
Current contract outranks run-ml-experiments: minimal existing durable records;
Trackio does not block. Public complete consumers and F6 take priority over
repeated internal polishing. Only affected checks,not unchanged8,954 CPU checks.
Implementation commit → immutable qualification → evidence commit; push each.

## Latest qualified work

Public multi-device FP32/FP16 training,canonical atomic optimizer,compact
Full/state/KV and portable repartition: clean0d7c45e,
[evidence](evidence/public-sharded-training-20261002.md). Earlier internal device
completion,retained reverse and other qualifications remain in ROADMAP.
HARD slot-affine emission reverse/publication: clean96c75f8,
[evidence](evidence/resident-emission-training-20261002.md);8 audited terminal jobs.
Projection weights and physical partial gradients still live on coordinator.

Continuous actual edge-affine Add/Attention CPU/mixed consumers: cleanfe2d886,
[evidence](evidence/online-consumers-20261002.md),CPU120/NPU18 passed.
Actual FP32 resident consumers: clean0d61cb94e7b911ff883de9af7b81042f87897a36,
[evidence](evidence/online-resident-consumers-20261002.md),
[audit](evidence/online-resident-consumers-20261002.json);6 jobs terminal PASSED,
CPU59/resident19/mixed18,no skips/no tolerance change.16 actual model trajectories,
64 windows/24 updates,CLI refusals and >2^54 integer/gradient boundary. Two installed
clients reuse byte-verified backend96c75f8. Separate actual D32 Attention2-card
trace:15,004 vector/633 AI_CORE/218 MIX_AIV,no observed AiCPU;32 model executes,
570 notify pairs,2,370 switches,4,552 async copies. Not formal throughput.
Evidence commit f7173d8 is pushed.

Resident source sources/online-resident-clean01; builds online-resident-{cpu-,}clean01,
emission-reverse-{,python-}clean01. Gate runs online-resident-{cpu,matrix,profile}-clean01
and mixed-clean02. mixed-clean01 omitted target and skipped18;excluded,retained.
Other five development failures remain failed,including D32 reverse16-row rejection.
Audit: python "$TASK/launchers/online_resident_evidence.py" full0d61cb9 hash.

## Current change and next commands

Uncommitted implementation: reverse_budget.h centralizes original event/fiber
reservation formulas and selects safe physical owner/query/key caps before
allocation. Logical fibers/KV/loss/update boundaries unchanged; smallest complete
owner still refuses explicitly. Existing disjoint budgets retained. Backward
statistics expose effective caps and reservation estimates;these are not measured
active row counts or total allocator peaks. Public ResidentGradients ABI changed:
rebuild clients/bindings. Consumer records these statistics without per-event reads.

All development jobs terminal PASSED from frozen sources/reverse-budget-dev01:
- build-reverse-budget-dev01 and build-reverse-budget-python-dev01:60/69 recursive
  affected objects rebuilt;unchanged kernels/core byte-verified.
- build-reverse-budget-consumer-dev01:installed standalone client.
- reverse-budget-cpu-dev01:1 CPU reservation gate,geometry throughD2048.
- reverse-budget-native-dev01:1 D32 actual consumer case (2 trajectories).
- reverse-budget-consumer-dev01:10 selected standalone consumer cases.
- reverse-budget-cache-dev01:83 event/fiber/FP16 public cases,no skips.
- reverse-budget-matrix-dev01:FP32/FP16 each16 trajectories,512 windows/128 updates total.
- reverse-budget-profile-dev01:actual D32 two-card training,reverse maximum16.
D32 auto-selected12 owner/query rows (64 key rows) and matched independent CPU
full training plus manual1-row execution. No tolerance relaxation. No new failures.

Next commit implementation,push;freeze reverse-budget-clean01 at exact commit.
Build using launchers/build_reverse_budget.py:
  reverse-budget-clean01 --reuse-host reverse-budget-dev01
  reverse-budget-python-clean01 --runtime python --reuse-host reverse-budget-python-dev01
Then build_online_resident.py --build builds/reverse-budget-clean01
  --out builds/reverse-budget-consumer-clean01. These fresh links reuse byte-identical
completed development objects only after recursive source/header validation.

Fixed-source jobs to launch: reverse-budget-cpu-clean01(test_reverse_budget.py,1),
reverse-budget-cache-clean01(test_resident_{event,fiber,half}_training.py,83,one card),
reverse-budget-matrix-clean01(emission_reverse_gates.py --scope matrix,two cards),
reverse-budget-consumer-clean01(test_online_resident_consumer.py +
test_online_resident_chunking.py,21,two cards),reverse-budget-profile-clean01
(profile_resident_consumer.py --preset resident --reverse-chunk-rows16,no development,
two cards). Use new Python resident build plus placement-npu-python-clean01 core;
TIDE_ONLINE_DEVICE=npu:0 for consumers,TIDE_RESIDENT_DEVICE=npu:0 for public cache.
Audit launchers/reverse_budget_evidence.py full implementation hash,write reviewed
report/update support boundaries,then evidence-only commit/push. No immutable
qualification claim for this change yet.

Continue compact projection owners,total-memory admission,consumer FP16,multi-device
public inference and representative/full-size F6. Do not end at small qualification.
Projection banks/partials and per-window copied projection versions are still scale
bottlenecks; place_full currently counts Full parameters only. Safe reverse splitting
is local,not total-memory admission. Multi-device inference must not use training tapes.
No new formal full-size performance result. Historical CPU Attention is supplementary.

## Environment and job bounds

TASK=/mi/data2T/zlong/tide-execution-flows;RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service;RUN/status.json/task.log and queue.json.
Module libtorch-npu/2.10.0-cann9.0.0;Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Default shell Python cannot run Torch tests. Public /opt stack supersedes dated
personal guide by user authorization. TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0;
preserve module PYTHONPATH,prepend snapshot/python. SoC Ascend910_9392;leased logical
NPUs only. Separate standalone and Python runtimes. Core builds placement-cpu-clean01,
placement-npu-clean01(standalone),placement-npu-python-clean01(Python-owned).
Frozen-source launcher freeze_run.py;runtime cwd env -C RUN avoids vendor files in
source. background.slice/Nice10,build2 workers,queue120s/run600s/build900s. Last disk:
data223GiB/root14GiB;check before heavy writes. Atomic handoff durable_records.replace_text.

Historical historical-cpu-attention-01 remains intentionally SIGSTOP;pause.json
outweighs running status and retains memory/TASK/timing.lock. Do not resume or kill.
Historical Add CPU78.793172/NPU4 47.932888ms/token = throughput1.6438× faster;
it does not certify resident. Earlier24 dirty files retained on pushed archive/
restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37,sha256 inventory
TASK/restricted-flow-archive.json. Preserve all cited artifacts and failures.
