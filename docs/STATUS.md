# Current handoff

Updated 2026-10-01T02:54:40.944986+00:00. **ACTIVE: user confirmed the execution contract and resumed implementation.**
Commit/push authorization remains active; no requested pause. No subagents.
Reference repositories and ObsidianVault are read-only. Repository
`/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
[execution-flows.md](execution-flows.md) owns the contract; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog. F1–F7 are still incomplete.

## Contract and scope

Candidates independently consume common inputs/initial state/parameters. No CPU
reference event, route, numerical result or gradient becomes a candidate input.
Online greedy accepts legal topology/input, including positive-delay PDG feedback;
it may naturally degenerate to streaming. Keep int64, stable order, parallel-edge
identity, missing/zero messages and None/zero gradients. Required performance:
PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch; CPU/NPU × streaming/prefill ×
inference/complete training. Python resident is a C++/CANN client, not an independent
PyTorch device scheduler. Five placement presets retain finer switches. FP32 main;
FP16 separate. CUDA execution is target-machine-pending.

Current alignment outranks run-ml-experiments. Preserve minimal source/input/config/
environment identity, raw failures/results, synchronized complete timing, bounded
resources/stops. Do not build duplicate tracking or block on Trackio. Training
means forward/loss interface/backward/VJP/optimizer/continuation/throughput;
model convergence belongs to later consumer experiments.

Use affected-path development checks and byte-verified terminal build reuse.
Do not rerun the unchanged portable core's8,954 CPU checks/23 optional skips.
Commit implementation, then qualify clean immutable source, commit evidence
separately. Event+fiber share one attention qualification milestone. Dirty-source
development checks do not certify a fixed commit.

## Qualified baseline and preserved work

HEAD before this increment is pushed **3b840a72c052c14a2a6c9dcb3800985903c9e26f**:
event attention training. Control/Read training has clean qualification at
**4b8ced41f422301a4ced2bf2999c5d1d3cb63e3d**, recorded in
[resident-control-training-20261001](evidence/resident-control-training-20261001.md).
Earlier resident forward, Full/state/graph VJP, optimizers, retained windows,
public clients and placement qualifications are linked from ROADMAP.

Cleanup e0afef4 and evidence e53abf5 are pushed. The24 earlier dirty files remain
SHA256-verified on pushed `archive/restricted-flow-20260930` at
**964bf628c67270200dabe55b1bca026bd403cd37**; TASK/restricted-flow-archive.json
records their hashes. This archived restricted consumer is not the general backend.

Numerical boundaries remain: LH/SwiGLU/Aggregate/control trajectories use AdamW
eps1e-5; public eps1e-8 and normalization epsilons are unchanged. RMSNorm eps1e-8
near-zero strict end-to-end trajectory still fails. Only the pre-existing Full
regression uses explicit conditioned controls (13 frames,max4.470348e-6); new
attention tests remain strict. Width257 SwiGLU declares2 GiB reverse budget.
Historical failures remain failed and are linked from prior evidence/records.

## Attention milestone — development passed, clean qualification next

Event implementation3b840a7: actual cache/event associations, tiled GQA VJP,
Q/K/V/O alias reduction/publication, independent cache roots/initial gradients,
retained KV bridges, SGD/AdamW and continuation. Development standalone66 root
cases/8 trajectories; Python17 new+154 regression cases;8 component cells passed.
Separate event profile:216166 tasks,6460 event-reverse tasks; no observed AiCPU
or logged CPU fallback. It includes construction/assertions, not throughput.

Current fiber increment implements complete-current-fiber attention VJP, all5
pooling profiles, QKV/output biases, repeated-tick decay, direct physical source/
scale adjoints, key/value/log-bias roots, actual device cache chains and retained
bridges, alias reduction and optimizer publication into live banks. Grouped reverse
copies are never optimizer destinations. [Contract](resident-fiber-vjp.md).
Single-NPU FP32 only; no new full-size performance claim.

Development results, all terminal PASSED/exit0:
- build-fiber-vjp-dev02 / fiber-vjp-dev02: local37 configurations/74 replays,
  CPU FP32/FP64,widths1/4/257,five pooling policies,independent NPU cache forward.
- build-fiber-training-python-dev03: separate Python-owned production build/loader.
  fiber-training-python-dev03:25 new public cases; three families,two schedules,
  five pooling policies,HST/SOFTP,SGD/AdamW,None/zero,loss cotangents,fresh-process resume.
  fiber-training-python-regression-dev03:171 old public/event cases.
- build-fiber-training-dev04/dev05: directed checker relinks with byte-verified
  completed standalone production libraries from terminal dev03. No failed
  parent is relabelled passed. Builds record source/binary/reused-artifact hashes.
- fiber-training-standalone-dev05:172 isolated/combined roots and20 complete
  trajectories,80 updates/320 retained windows,CPU FP32/FP64,feedback/DAG,
  logical slot permutations,physical scales/aliases,clear/adoption,None/zero,
  NaN padding,mixed event/fiber groups,periodic clocks,widths1/4/257.
- fiber-training-components-dev04:all8 affected fiber-VJP/event-training/graph/
  parameter/retained/public/control/Aggregate checks passed.
- fiber-training-profile-dev05:bounded --profile-smoke,512MB collection;
  13267 tasks,275 fiber-reverse tasks; AI_VECTOR_CORE/AI_CORE/MIX_AIV,
  no observed AiCPU or logged CPU fallback. Includes CPU assertions,not throughput.

Sources: sources/fiber-training-dev03 for Python;dev04 for component regressions;
dev05 for final standalone/profile. All production bytes match dev03; later
changes are checker/whitespace only. Current tools/device_online bytes match dev05.
NPU jobs leased physical9 or13, each mapped npu:0. No new job remains live.

Retained fiber failures:
- build-fiber-vjp-dev01:Ascend C GM float passed to Muls; local float fixed it.
- build-fiber-training-dev01:development overlay omitted CANN public includes.
- build-fiber-training-dev02:host plan lacked fiber declaration include; corrected
  plan/publication/trajectory includes. Its completed CANN kernels were reused
  with exact source/header/hash checks; the overall build stays failed.
- build-fiber-training-dev03:checker Tensor={} ambiguous; use Tensor{}.
- fiber-training-standalone-dev04:new test's sparse SourceDomain was illegal;
  replaced with a legal dense permutation, no production math change.
A short CPU-fixture probe initially requested nontrainable gradients; filtering
requires_grad as the real tests do passed15 family/pooling autograd fixtures.

Next actions: review/commit/push this coherent fiber increment. Then freeze that
40-character commit as attention-training-clean01; build standalone and Python-owned
backends separately with scripts/build_device_control.py (matching core dirs below,
--ascendc-soc Ascend910_9392 --jobs2,1800s). Run the complete component gate,
all public Python training cases and a separate bounded profile; preserve original
failures. Write audited event+fiber evidence in a separate commit. Then continue
FP16,peer progression/communication/training,representative/full-size CPU/mixed/
resident comparisons and migration/version evidence. No claimed new speedup.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows; RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service; RUN/status.json and task.log own lifecycle.
Gate/profile result.json and per-case logs live under their respective subdirectories.
Module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Public /opt stack supersedes the dated personal guide under user authorization.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0; preserve module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392. Use leased logical indices only.
Last free-space check:240 GiB data,25 GiB root.

Core builds: placement-cpu-clean01,placement-npu-clean01 (standalone),
placement-npu-python-clean01 (Python-owned), all under TASK/builds.
Never load the standalone SDK into TorchNPU Python. Freeze with
`python TASK/launchers/freeze_run.py --name NAME --snapshot NEW [--commit REV]
[--npu --npu-count N --max-wait 120] -- timeout --signal=TERM --kill-after=10s
900s '{python}' ...`. New edits require a new snapshot. Builds use2 workers;
all long jobs use background.slice,Nice10,finite bounds. Never mutate active
snapshots or terminal evidence. Use durable_records.py for atomic handoffs.

Historical historical-cpu-attention-01 remains deliberately SIGSTOP; pause.json
overrides running status. Do not blindly resume or stop. It retains host memory
and TASK/timing.lock. Resolve interrupted timing before formal throughput.
Historical Add CPU78.793172/NPU4 47.932888 ms/token means throughput1.6438x faster;
it does not certify the resident backend. No complete CPU Attention training ratio exists.
