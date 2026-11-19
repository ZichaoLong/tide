# Initial ROADMAP numerical qualification and bounded routing diagnosis

This report preserves the first FP32/BF16 outcomes of the current ROADMAP work.
[Reviewed machine-readable evidence](roadmap-numerics-initial-20261118.json.gz) retains
original record hashes, source/commands/device mapping, outcomes and bounded witnesses;
it omits raw tensors and repetitive operator records.
It does not relabel the four original FP32 NPU deviations or the restored AdamW
deviations in [the previous qualified report](instances-fp32-20261118.md).
The mathematics remains `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`.
Current implementation policy is owned by [precision](../precision.md); later
programme qualification is reported in [STATUS](../STATUS.md).

## Fixed initial observations

Initial implementation/qualification source:
`050dafa305d45162b2c86e736253455d7b476d27`. The seven configured cases are FC-K,
FC-S, Lattice-K, Exchange-X, Multiscale-X Settle/Timed, and FC-K d512, with their
actual configured widths (`--width 0`), B2/L5, three continuing AdamW updates,
and an actual update on both owners after checkpoint restore. All start from
the same CPU-created parameter/input identity for a given case. Every update
detaches the preceding private state; this is an explicit truncated-gradient
boundary, not a full-history backward claim.

BF16 here means BF16 projection arithmetic with FP32 parameters/gradients,
master parameters and AdamW moments. KV keys/values use BF16; scalar controls,
KV summaries and SSD recurrence/state use FP32. It is not a whole-model BF16
storage policy. The recorded operator audit is separate from performance timing.

| Initial record identity under `artifacts/` | Case outcomes | Restored actual update |
|---|---|---|
| `qualification-cpu-fp32-v1` | 4 within threshold; FC-K and both Multiscales have finite deviations | FC-K finite deviation; other 6 within threshold |
| `qualification-cpu-bf16-v1` | 3 FC finite deviations; all 4 HB cases retain metadata failures against CPU FP32 | FC-K, FC-S and FC-K d512 finite deviations; 4 HB within threshold |
| `qualification-npu-fp32-v1` | 7 finite deviations against CPU FP32 | All 7 within threshold |
| `qualification-npu-bf16-v1` | 3 FC finite deviations; all 4 HB cases retain metadata failures against CPU FP32 | FC-K, FC-S and FC-K d512 finite deviations; 4 HB within threshold |

The two initial BF16 jobs terminate with failure because the metadata checks
fail. The FP32 jobs allow recorded finite deviations and therefore exit zero;
that exit code does not turn their numerical rows into passes. These seven
actual-width cases are a different scope from the previous six d64 cases, so
the counts are not a before/after regression comparison.

No nonfinite payload, gradient or update was observed in these records.
All same-device checkpoint data, next-window forward and next state were exact.
The deviations after the actual next update remain distinct from those exact
data/forward claims. Initial NPU FP32 used physical device 9 and BF16 device 1,
remapped to logical `npu:0`, through the cooperative exclusive admission queue.
The software baseline was Torch 2.12.0+cpu, TorchNPU 2.12.0.post3 and CANN 9.2;
HF32 was disabled. These are correctness observations, not exclusive timing claims.

Initial full-tree comparison thresholds were FP32 `atol=1e-6, rtol=2e-3` and
BF16 `atol=2e-3, rtol=5e-2`. The restored-update check kept `1e-6/2e-3`, with
`rtol=1e-2` for input gradients; saved data and next forward/state used exact
comparison. No threshold was relaxed during diagnosis. The initial fixture
directories predate the new source/policy/file-hash manifest: their source is
fixed by the launch record, and their parameter/input hashes remain in the
records. New reference consumption records actual fixture hashes and labels
legacy directories without a manifest.

The BF16 HB metadata failures include real KV timestamp differences after
different positive-write supports, rather than floating dtype differences
alone. For example initial NPU Lattice's final `memories.9.K.times` differs;
Exchange and both Multiscales retain their own timestamp/support failures.
Some initial checkpoint gradient records also label signs of parameter tensors
named `weight` as “support”; those entries are not message-routing support.
Routing conclusions below use regional controls, returned strengths and KV
weight/timestamp state, not parameter/gradient signs.

## First-divergence investigation

Diagnostic source: `c09e5d7c` (full identity in the frozen launch record).
`routing-npu-v3` captured complete first-window trajectories on physical NPU 9;
`routing-analysis-v3` contains the CPU analyses for Lattice, Exchange and Timed
Multiscale. The capture records each actual region's score, reception, control
and requested/effective Attention path. It does not feed an oracle trajectory
to the model. A separate CPU analysis replays each first differing row in FP64
and solves the same **rounded inputs** using an independent 80-digit Decimal
quad breakpoint solve, examining at most four sample/time rows per comparison.
This oracle checks the forward allocation mapping, not the whole model or VJP.

Every examined production allocation agrees with the Decimal oracle's support.
The largest production-versus-Decimal control error among the examined NPU
witnesses is `8.195638656616211e-8`. Thus these witnesses do not identify an
allocation solver error. They identify score/reception perturbations crossing
a true zero-support boundary. The original failed metadata checks remain failed.

| Comparison | First differing region/layer | One affected coordinate's support margin before → after |
|---|---|---|
| Lattice FP32 → BF16 | 2 / 1 | `-0.0180062992 → +0.00863284618` |
| Exchange FP32 → BF16 | 4 / 1 | `-0.0149473488 → +0.00274202228` |
| Timed Multiscale FP32 → BF16 | 2 / 1 | `-0.0100597938 → +0.0186460391` |
| Lattice BF16 grad-enabled → inference | 4 / 2 | `+0.00185544789 → -0.000955155492` |
| Exchange BF16 grad-enabled → inference | 10 / 3 | `+0.0109446883 → -0.00106384754` |
| Timed Multiscale BF16 grad-enabled → inference | 6 / 2 | `+0.00430939347 → -0.000360414386` |

The margin is the solved score-minus-multiplier-minus-lower-bound at the
recorded coordinate. It is a diagnostic distance to that face, not a clipping
epsilon or a new acceptance tolerance. Cross-precision score differences cannot
be undone by promoting only the final allocator: the Decimal and FP64 replays
already agree with the support of each separately rounded score/reception row.

The initial BF16 Attention dispatch also changed arithmetic when gradients were
enabled. NPU positive trainable priors used a rank-one Q/K extension, while
frozen priors used PSE or a direct unit-prior shortcut. Visible zero trainable
priors used tiled math, while inference used native fusion. On the recorded
Timed example, grad-enabled execution used 44 tiled and 3 native site calls;
inference used 47 native calls. Explicit `tiled` produced exact grad-enabled/
inference readouts in all three NPU examples. Its trajectory can itself differ
from the original native trajectory, so this is evidence for consistent
representation/dispatch, not proof that either low-precision trajectory equals
FP32. Their original cross-mode readout relative-L2 differences were
0.0118723 (Lattice), 0.0125324 (Exchange) and 0.0114576 (Timed).

Finite CPU FP64 model replays on Lattice and Timed were also used during
development. They retained the corresponding FP32 first-window support, while
BF16 crossed the same early faces. These exploratory CPU captures are diagnostic
corroboration; the fixed NPU capture plus independent Decimal row witnesses
above own the source-scoped evidence in this section.

## Corrective policy and follow-up limit

The resulting BF16 policy sends visible zero priors to tiled math regardless
of grad mode and uses the same rank-one NPU representation for positive explicit
priors, including explicit all-one priors. Omitted unit priors keep the native
unit path. FP32 retains its previous dispatch. No mathematical support is clipped
and the zero-prior right derivative is preserved. Independent Attention/VJP,
mode-invariance and integrated checks cover this change; it is not a whole-model
FP32 substitution.

The subsequent fixed NPU BF16 record `qualification-npu-bf16-v5` at
`778c65fb2961624bb416343eb3ad0f6f031deff7` completed all seven cases on physical
device 6. All seven grad-enabled/inference comparisons are exact for values,
supports and state. **All four HB metadata failures against CPU FP32 still
remain**, as expected from the separate cross-precision boundary crossings.
That qualification job still exits with failure; the cross-mode fix does not
certify cross-precision equivalence or the complete performance programme.

At this follow-up source, exact checkpoint data/next forward/state are retained,
but FC-K and FC-S still fail the restored-update tolerance. Their input-gradient
maxima are both `4.8828125e-4`; parameter-gradient maxima are `1.953125e-3` and
`3.90625e-3`; updated-parameter maxima are `3.527291119098663e-5` and
`8.28281044960022e-5`. FC-K's failing update parameters are common regional
second-layer biases, consistent with sensitivity of a near-null direction.
FC-S's failures also include regional output/value matrices and description
parameters. Those broader BF16 backward differences cannot all be attributed
to a near-null bias from this record alone. Finite backward perturbations and
AdamW amplification are a bounded hypothesis; a same-owner versus restored-owner
backward witness can distinguish repeated-execution variability from an
owner/loading effect. This report does not claim a bitwise resumed trajectory.

## Reproduction entries

Use the stated source with the private environment and detached/admitted jobs
described in [execution](../execution-flows.md). The initial qualification
commands, run from that frozen source, are:

```sh
python -m scripts.qualify_precision --device cpu --precision fp32 --width 0 \
  --batch 2 --length 5 --steps 3 --threads 8 --checkpoint \
  --save-tensors NEW-cpu-fixtures --output NEW-cpu-fp32.json
python -m scripts.qualify_precision --device npu:0 --precision bf16 --width 0 \
  --batch 2 --length 5 --steps 3 --threads 4 --checkpoint \
  --reference /absolute/NEW-cpu-fixtures --output NEW-npu-bf16.json
```

The fixed routing diagnosis uses the following pair for each of
`hb-lattice-k`, `hb-exchange-x` and `hb-multiscale-x-timed`:

```sh
python -m scripts.diagnose_bf16 capture --device npu:0 \
  --config configs/hb-lattice-k.json --tiled \
  --trace NEW-lattice.pt --output NEW-lattice-capture.json
python -m scripts.diagnose_bf16 analyze --trace NEW-lattice.pt \
  --output NEW-lattice-analysis.json
```

CPU-only finite arithmetic diagnosis adds `--fp64` to a CPU capture. New
independent Attention formula/VJP checks can use
`python -m scripts.qualify_precision --device DEVICE --precision bf16
--primitives-only --output NEW-primitives.json`; that bounded scalar oracle
requires no CPU autograd trajectory inside an NPU process. Record identities
above name original run directories; they are not a promise to retain raw tensor
artifacts after the reviewed facts and reproducible sources have been preserved.
