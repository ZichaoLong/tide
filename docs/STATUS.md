# Current handoff

Updated 2026-10-01T07:55:04.988735+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
Commit/push authorization remains active; no requested pause. No subagents.
Reference repositories and ObsidianVault are read-only. Repository
`/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
[execution-flows.md](execution-flows.md) owns the contract;
[ROADMAP F1–F7](ROADMAP.md) is the only backlog. F1–F7 remain incomplete.

## Contract and scope

Candidates independently consume common inputs/initial state/parameters. No CPU
reference event, route, numerical result or gradient becomes a candidate input.
Online greedy accepts legal topology/input, including positive-delay PDG feedback;
it may naturally degenerate to streaming. Preserve int64, stable order, parallel
edge identity, missing/zero messages and None/zero gradients. Performance matrix:
PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch; CPU/NPU × streaming/prefill ×
inference/complete training. Python resident is a C++/CANN client, not an independent
PyTorch device scheduler. Five placement presets retain finer switches. FP32 main;
FP16 separate. CUDA execution is target-machine-pending.
Current alignment outranks run-ml-experiments: minimal source/input/config/environment
identity, raw failures/results, synchronized complete timing, bounded resources/stops.
No duplicate tracking or Trackio blocker. Training means forward/loss interface/
backward/VJP/optimizer/continuation/throughput; downstream convergence is out of scope.
Use affected-path checks and byte-verified terminal build reuse. Do not rerun the
unchanged portable core's 8,954 CPU checks/23 optional skips. Commit implementation,
qualify clean immutable source, commit evidence separately.

## Latest clean qualification

Implementation **2d7cee1ab9838dcaeacf639781053a53a822c308** committed/pushed.
[FP16 actual fiber reverse report](evidence/resident-fp16-fiber-reverse-20261001.md)
and [audit](evidence/resident-fp16-fiber-reverse-20261001.json). All6 jobs PASSED/exit0:
standalone/Python-owned builds,4 component cells,FP32 event/fiber regression,
61 Python cases(no skips),separate half profile. Each dtype90 cases/180 replays;
bias bridge3 cases/6 replays/6 refusals;reverse links64 windows. Five pools,
7 root modes,streaming/greedy,adopt/clear,width1/4/257,int64 above2^55,
permuted slots,missing/zero-scale sources,FP32 cache boundary sums beyond half.
Half roots×256;rtol2e-3/atol2e-5 half,unchanged1e-5/1e-6 FP32.
Profile162004 AI_VECTOR_CORE,11706 AI_CORE,1560 MIX_AIV;no observed AiCPU/
logged CPU fallback,not throughput. Physical9/1/13/11→logical0.
Source/archive/object/loader/log/CSV audit passed,including authenticated
production objects recovered from checker-failed dev01. FP32 event/fiber
regression66/172 roots,8/20 trajectories. All jobs terminal;no new speed ratio.

Retained dev01 compile failure:string/const-char helper mismatch. Dev02 FP32
fiber training failed because complete-graph dtype guard read an optional tanh
bank;changed to always-present source scales. Dev03 build/gate/regression
passed. No tolerance relaxation or failed-job relabeling.
Prior local fiber f2ec4f1/evidencecaa6088,event4f195d2/evidence3a8f2e8,
local attentionf20c2cc/evidence63824ba,normalized Aggregate/LH/SwiGLU
5ce5346/evidence588ed1d remain separately qualified.

## Active work and next action

Current evidence ready for separate commit/push. No uncommitted production code.
Next:half Emit/control/Read adjoints and actual HST/SOFTP forward rounding,
then complete graph reverse,retained windows,master/checkpoint/public FP16
training. Keep full graph/public training guards until real integration passes.
Control probabilities stay FP32 for complete-frame softmax;half Emit must use
rounded public controls/delta and actual unmixed Full values. All adjoints FP32.
Do not turn half forward into a whole-FP32 recomputation to bypass integration.

Remaining F1–F7:above half integrations,device peer progression/communication/
training,five-preset screening,representative/full-size CPU/mixed/resident
performance,version/migration/CUDA records. No subagents or requested pause.
Implementation commit→immutable qualification→separate evidence commit,push each.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows; RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service; RUN/status.json and task.log own lifecycle.
Gate/profile result.json and per-case logs are under their named subdirectories.
Module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Public /opt stack supersedes dated personal guide under user authorization.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0; preserve module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392. Use leased logical device indices.
Last space check:232GiB data,25GiB root; recheck before large writes.
Core builds:placement-cpu-clean01,placement-npu-clean01 (standalone),
placement-npu-python-clean01 (Python-owned). Never load standalone SDK into Python.
Freeze with `python TASK/launchers/freeze_run.py --name NAME --snapshot SNAPSHOT
[--commit REV] [--npu --npu-count N --max-wait 120] -- timeout --signal=TERM
--kill-after=10s 900s '{python}' ...`. Long jobs use background.slice/Nice10.
Never mutate active snapshots or terminal evidence. Atomic handoff:durable_records.py.

## Preserved historical boundaries

The24 earlier dirty files are SHA256-verified on pushed archive/restricted-flow-20260930
at964bf628c67270200dabe55b1bca026bd403cd37; TASK/restricted-flow-archive.json owns hashes.
Historical historical-cpu-attention-01 remains intentionally SIGSTOP; pause.json
outweighs running status. Do not blindly resume/stop. It retains host memory and
TASK/timing.lock. Resolve interrupted timing before formal throughput.
Historical Add CPU78.793172/NPU4 47.932888ms/token is throughput1.6438x faster;
it does not certify resident execution. No complete CPU Attention training ratio.
