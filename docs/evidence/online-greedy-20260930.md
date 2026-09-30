# General online greedy scheduling — 2026-09-30

Source `2038d8867ad0133af9b95589379435c95a326f03`, clean frozen checkout.
The [manifest](online-greedy-20260930.json) records terminal statuses, report/log
hashes, runtime identities and the exact six accelerator fixtures.

The full CPU FP64/FP32 regression passed **8849 tests** in 1713.58 seconds.
A clean core/binding/client build and four real CTests also passed: independent
C++ Greedy and Settle checks in both precisions. The directed tests include
positive-delay feedback, unequal delays, physical parallel edges, complete
observables, isolated VJPs/None gradients, attention cache, continuation and
explicit live-capacity/time-overflow refusal. Historical LH qualification was
not rerun in this increment; its existing evidence retains its own scope.

Both matching NPU builds passed: Python-owned bindings and standalone
LibTorch-NPU. Six FP32 public fixtures (Python/native × PDG/TimedDAG/Settle)
passed independent CPU observables/VJPs, chunk continuation, trace-disabled
values/VJPs, three AdamW updates, fresh-process checkpoint and CPU handoff.
The fixtures exercise linear, SSM, delta and attention modules, HST and batched
local VJPs. These six cells do not exhaust every module/topology/option combination.

The device was Ascend910_9392, aarch64, CANN9.0.0, Torch2.10.0+cpu with
TorchNPU2.10.0, FP32 and HF32 disabled. Tolerance was atol1e-6/rtol1e-5;
discrete structure and gradient absence remained exact. The standalone build
is build evidence; Python-native qualification is not an independent C++ NPU
execution gate. The four standalone runtime checks above ran on CPU.

Commands and artifacts are retained through `artifacts/execution-flows-*`:
`greedy-clean-cpu01` runs `scripts/build.py --backend cpu`, four CTests and
`scripts/verify.py --dtype both`. `greedy-npu-semantic01` uses the hashed
external driver to call public `qualify` with `schedule=greedy`; its original
six reports remain available beside `semantic.json`.

This is a **host scheduler with optional NPU tensors**. It has no numerical
route prepass or whole-window potential-event expansion, but per-atom host
bookkeeping remains. It does not certify NPU-resident scheduling, byte-budgeted
model chunking, complete large-flow consumers, multi-device progression, FP16
Greedy, CUDA execution or performance. Those remain separate F1–F7 obligations.
