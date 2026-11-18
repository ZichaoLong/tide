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
- `5059ff8f`: composite `matmul`/`linear` audit witnesses, without arithmetic changes.
  `0992b467`: bounded capacity training with actual storage/finite scans and
  complete wall time. This source passed all 424 CPU tests and the clean installed
  wheel checks; [installed evidence](evidence/roadmap-installed-20261118.md) retains
  the two finite FP32 restored-update deviations. `e967a02d` publishes the
  reviewed Attention, sparse/shape/combinations and 0.5B/1B evidence.

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

Active: `representative-cpu-v5` (CPU32, cores 0–31). All four NPU continuous
programmes completed 240 children / 720 continuing segments. They cover B1/B4,
both precisions, three modes, Tide/A–D, 128 prefill + 8 decode and actual restored
updates. CPU continuous and remaining scale work are still running.

The not-yet-computing scale/CPU jobs were superseded before device admission;
original stop statuses and `superseded.json` remain in each old run. New jobs use
fixed `0992b467`, with observed NUMA policy/affinity in `numa-policy.json`:

| Job | Placement and sequencing |
| --- | --- |
| `continuous-cpu-v16` | NUMA 7, cores 280–295; finite 4/8/16-thread calibration, then compact B1/B4 Tide/A–D continuous matrix |
| `attention-sites-cpu-v16` | NUMA 7, cores 280–311; waits for CPU continuous |
| `scale-cpu-hb-17b-inference-v17` | NUMA 6, cores 240–279; finite 16/32/40-thread calibration, then 17B A–D FP32/BF16 inference |
| `scale-cpu-hb-17b-train-v17` | interleave NUMA 6/7, cores 240–279; waits for CPU17B inference and CPU Attention, then two BF16 CPU updates using the selected thread count |
| `scale-npu-hb-9b-v16` | NUMA 4, cores 176–191; cooperative one-NPU A–D inference |
| `scale-two-npu-hb-17b-v16` | interleave NUMA 4/5, cores 192–207; cooperative two-NPU inference |
| `scale-four-npu-hb-9b-train-v16` | NUMA 5, cores 208–223; waits for two-NPU run, then four-NPU FP32/BF16 training, two continuing updates each |

CPU17B admission checks global available RAM ≥700 GiB. Inference requires
NUMA 6 free plus file LRU/reclaimable slab ≥160 GiB (RSS bound 192 GiB);
training requires each of NUMA 6/7 ≥210 GiB (RSS bound 384 GiB). These checks
are margins, not reservations. The earlier combined CPU17B v16 was also
superseded before computation so independent inference could start earlier.
Other jobs/users may still create contention.

`scale-two-npu-hb-17b-v16` completed both precisions: FP32 median 1.283656 s,
BF16 1.478270 s at B1/L8; per-card timed peaks were 42.406/20.778 GiB and
42.980/21.351 GiB. These are host-staged capacity observations, without A–D
multi-device references. Four-NPU 9B FP32 completed two continuing updates;
its BF16 attempt is ongoing. Current large evidence is still being reviewed.

`ca6199d4` fixes launcher-only CANN Python-path loss. The previous launcher
hid `tbe` by replacing module-provided paths; the fix preserves the selected
baseline and records actual paths/module origins. No model arithmetic changed.
`environment-cpu-v18` passed an actual service import/isolation check;
`placement-environment-npu-v18` is a bounded two-card follow-up of auto/BF16,
synchronized/FP32 and direct/FP32 in separate processes. Original environment
failures and active source/environment identities remain unchanged.

Next: finish these actual runs, investigate execution failures within the finite
scope, integrate reviewed comparison/continuation/placement/capacity evidence,
and complete the handoff and artifact review. All five workstreams are not yet
complete; stage reports do not stop this task.

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

Latest preflight: about 44 GiB account use / 300 GiB soft / 700 GiB hard;
about 1.3 TiB disk and 1.8 TiB host memory available. Public environments,
reference and downstream repositories remain unchanged. Private environment:
`ascend/dev-workspace-9.2.0`, Torch2.12/TorchNPU2.12.post3/CANN9.2.
Current-conversation shared-NPU authorization has not been needed: all device
runs so far used cooperative exclusive admission, with external activity outside
our control. User authorization includes bounded experiments and ordinary pushes.
