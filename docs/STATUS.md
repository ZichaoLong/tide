# Current handoff — five workstreams in progress

Branch: `graph-execution-foundation`. Fixed mathematics:
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`; all 108 files verified.
The continuing FC/HB instance route remains active; no general foundation route.
The four original untracked user documents are preserved and excluded from commits.

## Implementation and fixed evidence

- `778c65fb`: explicit BF16 matrix/FP32 parameter-state policy, mode-consistent
  BF16 Attention, measured NPU query fusion and KV compaction. CPU retains its
  measured alternatives. The seven-point main matrix uses this source.
- `9a1f600f`: real cached A/B/C/D continuous comparisons, complete Timed flush in
  restored updates, and required state detach included in complete training time.
- `5185dc0d` / `0e0023ff`: bounded restored-owner/backward and whole/chunk VJP
  diagnosis, with the latter fixing FP64 diagnostic metadata.
- `c172c606`: default multi-NPU host staging after two-device qualification;
  explicit transport identity, sharded AdamW ownership/checkpoint semantics,
  exact meta-only parameter partition budgets and corrected private-state bounds.
- `6fc53d81`: finite whole-model capture probes, including compiler descendant
  supervision. `c8026e47`: observable resource waits before NPU admission.

[Hotspot evidence](evidence/roadmap-hotspots-20261118.md) preserves adopted NPU
compaction/query fusion and rejected full bucketing/extra gate-up fusion. The
observed training reductions are not strict same-trajectory speedups: some final
work differs. CPU32 was selected from 4/8/16/32 for B8/L128 representative timing.

The original four FP32 NPU failures and AdamW deviations remain in
[their original report](evidence/instances-fp32-20261118.md).
[Initial numerics](evidence/roadmap-numerics-initial-20261118.md) and
[follow-up numerics](evidence/roadmap-numerics-followup-20261118.md) preserve all
new failures and thresholds. Seven BF16 inference/grad-forward cases now agree
exactly on both CPU and NPU, but all four HB cases still change support/KV times
against FP32. Independent Decimal witnesses agree with each rounded allocation.
All new NPU FP32 integrated cases retain finite deviations; restored d512 also
has a finite deviation in the latest observation. BF16 FC restored updates retain
finite errors. Same-owner FC-S backward varies at the same scale; selected actual
AdamW update differences agree with independent FP64 scalar calculations.

Whole/chunk full-backward witnesses pass FP32 on CPU/NPU and CPU FP64. BF16
FC-K/Exchange/Timed retain finite value/state differences under auto and tiled;
no support, logical metadata or disconnected-gradient difference was found in
these witnesses. No thresholds were relaxed. Nine independent Attention/VJP
cases pass in each new CPU/NPU FP32/BF16 qualification record.

## Completed and active execution

All durable records are under ignored `artifacts/NAME/`, with exact Git archives,
commands, status/logs and cooperative queue identities. RSS limits are sampled
process-tree bounds, not hard cgroup limits. Terminal numerical failure is kept
separate from execution completion. Inspect actual records for live status.

Completed: seven `representative-npu-*-v5` jobs (216 rows / 1080 samples),
`sparse-npu-v7`, `shape-npu-v7`, `combinations-npu-v7` (360 rows / 1080 samples),
four `scale-npu-{fc-0p5b,fc-1b,hb-0p5b,hb-1b}-v7` jobs, and
`attention-sites-npu-v7` (60 primitive rows). Some samples are noisy; source-scoped
reports retain every sample and actual matching/work differences.

`placement-host-stage-npu-v8` completed FP32/BF16 Exchange/Timed: inference,
backward, three continuing updates, exact saved data and actual restored update.
There are finite whole/chunk and multi-step deviations, including on the single
owner reference. Direct transport failed with an invalid event; isolated copy
success did not qualify it. Host staging explicitly pays CPU transport, without
claiming peer DMA, HCCL, replicas or full trajectory equivalence.

`resident-npu-v11` selected installed TorchAir, but all four fullgraph attempts
stopped before backend invocation at a static-role tensor guard. This is an
observed capture boundary, not proof that resident execution is impossible;
no resident scheduler was implemented or qualified.

Active: `representative-cpu-v5` (CPU32, cores 0–31); four
`continuous-npu-{fc-k,fc-s,hb-multiscale-x-settle,hb-multiscale-x-timed}-v8`
jobs (B1/B4, both precisions, three modes, Tide/A–D, 128 prefill + 8 decode,
three continuing segments and actual restored updates).
`scale-npu-hb-9b-v12` and `scale-two-npu-hb-17b-v12` wait for the CPU main
matrix to release resources **before** device admission. Their weight
initialization must not compete with that CPU timing.

Next: finish the active matrices, measure 9B four-NPU training and 17B CPU
capacity alternatives, run the compact CPU continuous matrix after a finite
thread calibration, then integrate reviewed performance/continuation/backend
reports and the final installed smoke/current CPU suite. All five workstreams
are not yet complete; stage reports do not stop this task.

## Capacity and resources

9B has 8,778,358,224 parameters. Four layer partitions have at most 44.428 GiB
FP32 parameter/gradient/AdamW persistent storage; disabling foreach leaves about
15.45 GiB after steady state on a planning 60 GiB device. The largest Parameter
shard is 2.25 GiB. This permits a bounded trial, not an allocation guarantee.
17B has 16,707,220,176 parameters. Two-card weights are 41.901/20.338 GiB;
B1/L8 inference has headroom. Its widest layer alone requires 83.795 GiB for
AdamW persistent storage, so even all five layers on separate 60 GiB devices
cannot train it under current layer placement. CPU memory is a feasible
alternative; no unrun capacity point is a pass.

Latest preflight: about 43 GiB account use / 300 GiB soft / 700 GiB hard;
about 1.3 TiB disk and 1.8 TiB host memory available. Public environments,
reference and downstream repositories remain unchanged. Private environment:
`ascend/dev-workspace-9.2.0`, Torch2.12/TorchNPU2.12.post3/CANN9.2.
Current-conversation shared-NPU authorization has not been needed: all device
runs so far used cooperative exclusive admission, with external activity outside
our control. User authorization includes bounded experiments and ordinary pushes.
