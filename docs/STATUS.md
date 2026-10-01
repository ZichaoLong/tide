# Current handoff

Updated 2026-10-01T07:31:09.249898+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
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

Implementation **f2ec4f161c86c0af6bfaf9ec2ff9755b1cb4d0ea** committed/pushed.
[FP16 local same-fiber report](evidence/resident-fp16-fiber-vjp-20261001.md)
and [audit](evidence/resident-fp16-fiber-vjp-20261001.json). All6 jobs PASSED/exit0:
standalone/Python-owned builds,two dtype cells,FP32 event/fiber regression,
61 Python cases(no skips),separate half profile. Each dtype37 configurations/
74 replays,five pooling modes,seven roots,width1/4/257,multihead,changing
lengths,None/zero,poison padding,tick/budget refusal. Half roots×256;
rtol2e-3/atol2e-5 half,unchanged2e-5/2e-6 FP32. FP32 training66/172 roots,
8/20 trajectories. Profile15017 AI_VECTOR_CORE,1405 AI_CORE,170 MIX_AIV;
no observed AiCPU/logged CPU fallback,not throughput. Physical9/1/3/9→logical0.
Source/archive/object/loader/log/CSV audit passed. Byte-matched terminal host/
kernel reuse plus checker/Python host rebuild; not a full vendor rebuild.
Development fiber dev01/dev02 and regression passed; no failed fiber jobs.

Prior event cache/projection implementation4f195d2,evidence3a8f2e8,all6 jobs
passed; [report](evidence/resident-fp16-event-vjp-20261001.md). Local attention
f20c2cc/evidence63824ba,normalized Aggregate/LH/SwiGLU5ce5346/evidence588ed1d
remain separately qualified. Complete graph/public FP16 training remains guarded.
No new full-size speed ratio.

## Active work and next action

Local fiber evidence committed/pushed as caa6088. Uncommitted next increment:
FP16 actual fiber cache/source reverse integration. Changed content_flow.cpp,
graph_vjp.cpp,reverse_links.cpp,fiber_cache_reverse.cpp,fiber_reverse.cpp,
fiber_tape.cpp,AscendC tide_fiber_reverse_pack,reverse_links_check.cpp and mapping.
Actual half journal access now possible; complete-graph dtype guard moved to
append_graph_vjp,not removed. Reverse links widen half scales to FP32. Fiber
packing rounds physical source products,loads half params/cache; all roots/carry/
returned gradients FP32. Sum-only dummy pooling banks are FP32.

Reverse-link checker extended to both dtypes,64 actual feedback/parallel-edge/
empty/policy/continuation windows each. New actual fiber reverse checker covers
90 cases/180 replays per dtype:5 pools,7 roots,streaming/greedy,adopt/clear,
permuted slots,missing/zero-scale physical inputs,large int64 times,nonempty/
empty caches,CPU FP32/FP64 quantized-forward gradients. Separate bias bridge:
3 cases/6 replays/6 refusals,FP32 sums beyond half range,capacity257,padding and
None/zero. Half complete-graph guard remains explicitly checked. No device result yet.

Preserved build dev01 failure: new checker passed std::string to const char*
helper; fixed with c_str(),also check physical input position. Kernel/six host
objects completed before the checker failure; never relabel failed dev01.
Recovered build-low-precision-fiber-reverse-dev02 PASSED/exit0 from frozen
TASK/sources/low-precision-fiber-reverse-dev02 into matching build directory.
Launcher TASK/launchers/build_precision_fiber_reverse_recover.py,900s. Validate
all source bytes except fixed checker and every completed archive member against
its object or qualified parent;reuse production objects from terminal failed
dev01,compile both checkers. Keep failed status/log/source unchanged. Dev02 component gate PASSED:90/180 actual fiber cases/replays and64 reverse-link
windows per dtype;bias bridge3/6/6. Regression event passed,then fiber failed:
new complete-graph dtype guard checked the optional tanh weight bank,which is
undefined for identity-only graphs. Fix uses always-present source scales to
identify payload dtype. No numerical/tolerance change. Preserve original failure.
Corrected dev03 build,component gate and FP32 training regression all PASSED/
exit0. Source/build low-precision-fiber-reverse-dev03,launcher
build_precision_fiber_reverse_fix.py. Rebuilt only graph_vjp.cpp/checkers;
remaining authenticated production objects reused. FP32 identity-only full
training now passes; half complete graph remains rejected. No tolerance change.

Implementation ready to commit. Next fixed clean source:
freeze_run.py --commit NEW_REV --snapshot low-precision-fiber-reverse-clean01;
launchers/build_precision_fiber_reverse_recheck.py low-precision-fiber-reverse-clean01
and build_precision_fiber_reverse_python.py low-precision-fiber-reverse-python-clean01.
Build900s,two Python host compiler workers;gates/profile600s,lease120s. Run
fiber-reverse/reverse-links both dtypes,FP32 event/fiber regression,Python
precision/event/fiber three modules,and separate half fiber-reverse profile.
Audit precision_fiber_reverse_evidence.py NEW_REV after every job is terminal;
it reauthenticates the completed objects from checker-failed dev01 too.
Implementation commit→immutable qualification→separate evidence commit,push each.

Remaining: finish fiber integration,control/graph half adjoints,retained windows,
master checkpoint/public FP16 training; then peer progression/communication/
training,five-preset screening,representative/full-size CPU/mixed/resident
performance,version/migration/CUDA records. F1–F7 incomplete. No subagents/pause.

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
