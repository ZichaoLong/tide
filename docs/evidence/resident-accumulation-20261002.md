# Resident gradients across explicit backward groups

Qualified source: `830904b712f458c3d268cde88811b84157a26e6e`.
[Audited receipts](resident-accumulation-20261002.json),
[public contract](../resident-training.md).

`accumulate(max_bytes)` now freezes the latest canonical parameter gradients
into a separate FP32 device bank, then sums later backward groups and ORs their
connection flags. The single-device and compact multi-device owners share this
policy. It leaves parameters, optimizer slots and generation unchanged; the final
`step()` performs one finite agreement, update and parameter publication.

Each call explicitly detaches the current state/history/pending/KV. Windows that
require connected differentiation must stay within one backward group. This is
not independent sample continuation switching or resident sample slicing; neither
has been implemented by this increment. There is no loss averaging or implicit
normalization. Default backward/step and checkpoint formats remain unchanged.

The byte bound checks simultaneous old and replacement accumulator storage plus
tensor metadata across devices before execution. Capacity refusal preserves the
current gradients and existing accumulator. Current backward tensors and program
arenas use their separate budgets. Step refuses an unaccumulated final backward
or outstanding windows; checkpoint refuses accumulated gradients. Explicit
detach discards a rejected accumulation without changing parameters.

## Verification

Five clean jobs passed: two independent runtime builds, standalone component
gate, Python public gate and a separate placement profile. Builds verified exact
source/options before reusing development objects, freshly linked each clean
runtime, and audited unchanged core/content/CANN dependencies and loader closure.
The public limits/checkpoint structures and kernel implementations are unchanged.

- Standalone: eight trajectories for each of FP32 and FP16, 384 forward windows
  and 48 optimizer updates. Independent CPU FP32/FP64 oracles check full forward
  continuation, individual VJPs and summed optimizer/master/slot/counter results.
  Coverage includes streaming/prefill, SGD/AdamW, HARD/HST/SOFTP, positive feedback,
  duplicate edges, aliases, event/fiber attention, two explicit owner maps and
  checkpoint resume onto the legacy single-device owner.
- Python public client: 21 tests, no skips. Six accumulated trajectories cover
  all three graph families and both schedules, three groups per update and
  retained windows within each group. Two rejection tests cover capacity,
  lifecycle, nonfinite refusal, explicit discard and storage isolation. Thirteen
  existing ordinary training/lifecycle tests pass unchanged.
- Both gates check real+real sums, connected-zero and all-None groups. Optimizer
  progress happens once per final step. Checkpoint export/resume retains exact
  optimizer state; caller mutation of a consumed backward export cannot rewrite
  its frozen accumulator.

One separate FP32 profile observes 18,360 vector-core tasks, 350 AI Core tasks
and 267 MIX_AIV tasks across the two leased devices, with no observed AiCPU task
or CPU-fallback warning. It includes the existing parameter accumulation kernel
used both within retained reverse and between detached groups. It also includes
construction and CPU assertions and is not a throughput measurement.

No full-size, performance, CUDA or additional CANN-version claim is made.
The larger logical-batch consumer still needs device continuation switching and
capacity-aware sample slicing before it can use this owner without per-slice
optimizer updates.
