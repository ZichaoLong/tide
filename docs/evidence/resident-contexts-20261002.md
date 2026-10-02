# Detached device continuation switching

Qualified source: `c96ebcd276e58d5bcdf18aa66223ef5d57463bbc`.
[Audited receipts](resident-contexts-20261002.json),
[public contract](../resident-training.md).

The public C++ and Python resident owners can save an opaque numerical
continuation on its original NPU devices and restore it into the same live
owner. Independent input streams retain their own state, clocks, presence,
complete event/fiber KV and bias/lengths, selection history and pending messages.
Input-ledger/cut metadata remains on the host. No numerical state is downloaded
for switching; whole tensor buffers are copied on their existing devices.

Training shares current parameters, optimizer and accumulated gradients across
these continuations. Save/restore requires a detached boundary and refuses live
retained windows or an unconsumed backward. A saved state from before a parameter
update can continue under the current parameters. Tests keep connected windows
within each backward group and accumulate groups before one shared update.
Foreign handles and insufficient snapshot budgets are rejected before mutation;
restoration clears previous-window diagnostics. Handles remain immutable.

## Verification

Five clean jobs passed: separate standalone and Python-runtime builds, component
and public gates, and an independent profile. Exact source/options checks allowed
reuse of development objects; affected content archive members were replaced,
public libraries freshly linked and loader closures audited. Core/CANN kernels
and existing public limits/checkpoint layouts are unchanged.

- C++: eight trajectories per dtype for contexts and eight for accumulation
  regression, FP32/FP16, totaling 768 windows and 96 optimizer updates. Independent
  CPU FP32/FP64 forward/VJP/optimizer references check state, pending/KV, gradients,
  masters, slots, counters and None/connected-zero behavior. Contexts alternate
  two distinct input streams on two explicit owner maps; ordinary accumulation
  retains its legacy single-owner checkpoint-resume coverage. Includes both
  schedules, SGD/AdamW, HARD/HST/SOFTP and event/fiber caches.
- Python: 30 tests, no skips: six inference context cases, three shared-update
  context cases, eight accumulation and thirteen ordinary training regressions.
  Covers all three families, both schedules, one/two devices, identity/EMA/Add
  and event/fiber attention, independent CPU continuation, capacity/ownership
  refusal and restore across parameter updates.
- The first development Python run retains two new-test failures: identity was
  specified as a memory kernel rather than a node flag, and Settle internal
  continuation was compared with the projected public graph identity. Corrected
  fixtures passed without production-code or numerical-tolerance changes.

One separate FP32 two-device trace observes 26,159 Vector Core, 350 AI Core and
363 MIX_AIV tasks, with zero observed AiCPU tasks or CPU-fallback warning. It
includes construction and CPU assertions and is not throughput evidence.

Snapshots still reserve dense capacity storage. Their byte bound covers only
the newly saved tensors; callers must account for all live handles and the active
owner. This increment does not implement compact KV, automatic sample slicing,
full-size execution or new CUDA/CANN-version support. Resident consumer sample
slicing must still compose these contexts with whole-batch normalization and
one shared optimizer update.
