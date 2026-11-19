# ROADMAP numerical qualification, continuation and restored-update follow-up

This report preserves the completed follow-up qualification and bounded diagnoses.
[Machine-readable evidence](roadmap-numerics-followup-20261118.json.gz) records the
frozen sources, commands, fixture and result hashes, admission mappings, thresholds,
case outcomes and selected numerical witnesses. The
[initial report](roadmap-numerics-initial-20261118.md) and earlier FP32 failures remain
unchanged. Neither a successful job exit nor an exact restored forward implies that
all numerical checks passed.

The fixed mathematics is `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`.
CPU qualification uses `9a1f600f2b712aaebf28a1a689d864b2358a5fa6`;
NPU qualification and restore diagnosis use
`5185dc0d1d3ae4c570e0712a42826e3974161bed`.
Their `src/tide` tree, qualification script and continuous/checkpoint harness
are identical Git objects; those identities are recorded in the JSON.
The new CPU fixture manifest includes its source, precision policy, Torch version
and individual file hashes; every NPU reference consumption records that manifest
and the actual case/Attention fixture hashes.

## Actual-width qualification

The seven cases are FC-K, FC-S, HB Lattice-K, Exchange-X, Multiscale-X Settle and
Timed, and FC-K d512. Each uses its configured width, B2/L5, identical CPU-created
initial parameter/input identities, three continuing AdamW updates and one actual
update after restoring model, private state and optimizer. State is detached
between update windows. This is an explicit truncated-gradient training boundary.

BF16 uses BF16 projection arithmetic with FP32 parameters, gradients, masters,
AdamW moments, scalar controls and SSD state. It is not whole-model BF16 storage.
Operator/dtype audits accompany the result records. Torch was `2.12.0+cpu`,
TorchNPU `2.12.0.post3`, CANN 9.2; NPU HF32 was disabled.

| Record under `artifacts/` | Overall case outcomes | Actual restored-update outcomes |
|---|---|---|
| `qualification-cpu-v8/fp32.json` | 3 within threshold; FC-K, both Multiscales and d512 finite deviations | FC-K and d512 finite; other 5 within |
| `qualification-cpu-v8/bf16.json` | 3 FC finite deviations; all 4 HB metadata failures against CPU FP32 | 3 FC finite; 4 HB within |
| `qualification-npu-fp32-v9/result.json` | All 7 finite deviations against CPU FP32 | d512 finite; other 6 within |
| `qualification-npu-bf16-v9/result.json` | 3 FC finite deviations; all 4 HB metadata failures against CPU FP32 | 3 FC finite; 4 HB within |

Both BF16 records retain the four HB metadata failures. These include logical KV
timestamp differences caused by differing positive-write supports; floating dtype
differences alone are not their explanation. The independent rounded-input
allocation witnesses in the initial report remain the bounded explanation for the
examined support crossings, without certifying whole-model cross-precision equivalence.

All seven BF16 inference/grad-enabled-forward comparisons are **exact** for
values, support and state on CPU and NPU. FP32 retains its established dispatch
and finite mode differences: both Multiscales on CPU, and Lattice, Exchange and
Settle on NPU exceed their value/state thresholds in these runs. No nonfinite
payload, gradient or update was observed. Nine independent Attention forward/VJP
checks—positive prior, zero prior and empty case, each with math/tiled/auto—are
within threshold in each of the four qualification records.

The NPU qualification queue admitted FP32 on physical device 11 and BF16 on
device 0, each mapped to logical `npu:0`. These were cooperative exclusive
admissions; unrelated external activity was not controlled. The records are
correctness evidence, not device-exclusivity timing claims.

## Checkpoint restore includes the real next update

For all 28 qualification rows, saved parameters/state/optimizer, next-window
forward, next state and the returned state after the update are exact between
owners. The Timed cases additionally compare the complete flush output exactly,
and include that flush in each owner's loss/backward/AdamW update. The restored
Timed window has 10 real input positions, 40 window output slots and 16 flush
slots; its final logical position is 22. Output slots are not token counts.

Gradient and parameter-update tolerances were not relaxed. Restored checks use
`atol=1e-6, rtol=2e-3`, with input-gradient `rtol=1e-2`; data and next-forward/state
checks are exact. NPU BF16 full-tree restored parameter-gradient maxima for
FC-K/FC-S/d512 are `3.90625e-3`, `3.90625e-3`, `1.953125e-3`; updated-parameter
maxima are `2.2370368242263794e-5`, `1.2369826436042786e-4`,
`3.223121166229248e-5`. All three retain finite deviations.
NPU FP32 d512 also retains one updated-parameter failure, with full-tree maximum
`6.321817636489868e-6`. A smaller or within-threshold observation in a later run
does not relabel an earlier failure.

`restore-backward-npu-v9` makes four unchanged-weight/state/input backward calls
in order: original owner, restored owner, original owner again, restored owner
again. It then performs a real AdamW update on each owner using its second
gradient. The source is fixed above; physical NPU 11 was admitted independently.

FC-S's saved data and all four forward/state comparisons are exact; inspected
parameter layouts, strides and NPU formats match. Nevertheless, repeating backward
on **the same owner** gives input-gradient maximum `4.8828125e-4`, and a retained
parameter-gradient witness at `inputs.D.weight` has maximum `3.90625e-3`.
Cross-owner comparisons have the same scale. This rules out a restore-only
explanation for the observed FC-S variability; it does not isolate one backend
kernel or prove deterministic training.

The FC-S update retains a reported tensor witness of `9.742379188537598e-5`.
Independent scalar FP64 AdamW calculations use each selected coordinate's actual
stored gradient, old moments, parameter, step and optimizer constants. Across the
eight selected coordinates, the maximum residual between predicted and actual
parameter **differences** is `9.902967681996167e-11`. For example,
`regional.0.o.weight[2073]` has gradients `1.1920928955078125e-5` and
`-5.662441253662109e-6`, predicting an update difference `6.722090176091919e-5`
versus actual `6.722100079059601e-5`. Thus measured backward perturbations and
ordinary AdamW sensitivity explain these selected updates; they cannot all be
assigned to a near-null common bias. No optimizer or threshold change was made.

FC-K in this separate diagnosis has much smaller backward differences and its
actual update is within threshold. Its earlier v5 and v9 qualification failures
remain recorded. All scalar and tensor maxima in this paragraph are retained
witnesses: the diagnostic preserves the eight tensors with largest scaled error,
which need not include the globally largest absolute error. The qualification
checkpoint `max_abs` fields above do cover their complete trees.

## Whole/chunk full-backward witnesses

`scripts.check_continuation` compares one whole sequence with the same sequence
split into chunks, using identical parameters, inputs and deterministic
cotangents. It retains the entire autograd graph across chunks, with no state
detach or optimizer. Both Timed losses include the complete flush. It compares
readout, strengths, timestamps, final private state, full input VJP, parameter
VJP and disconnected-gradient identities.

The finite scope is four d64 witnesses: FC-K and FC-S with one node/layer and
chunks `65+64+2`, crossing the 128-position window; Exchange-X and Timed
Multiscale-X with two nodes and chunks `2+3`. Batch is 2. FC processes 262 input
positions and returns 262 slots; Exchange has 10/10; Timed has 10/48 and finishes
at logical position 6. This does not qualify arbitrary shapes or long-run
optimization trajectories.

| Device/precision and record | FC-K | FC-S | Exchange | Timed |
|---|---|---|---|---|
| CPU FP32, `continuation-numerics-cpu-v9/fp32.json` | within | within | within | within |
| NPU FP32, `continuation-numerics-npu-v9/fp32.json` | within | within | within | within |
| CPU BF16, `continuation-numerics-cpu-v9/bf16.json` | finite | within | finite | finite |
| NPU BF16, `continuation-numerics-npu-v9/bf16.json` | finite | within | finite | finite |
| CPU FP64 diagnostic, `continuation-fp64-cpu-v10/result.json` | within | within | within | within |
| NPU BF16 tiled, `continuation-tiled-npu-v10/result.json` | finite | within | finite | finite |

All these comparisons have zero nonfinite, support, logical metadata and
disconnected-gradient differences. The BF16 deviations are in readout/private
values. Input and parameter VJPs are within the existing BF16 threshold in all
four cases on both devices, including tiled. For NPU auto, the largest scaled
value/state errors for FC-K/Exchange/Timed are `4.3516`, `7.6183`, `8.3824`;
forced tiled gives `3.5587`, `7.4078`, `8.3824`. A scaled error above 1 is a failure.
Tiled does not eliminate the discrepancies, so a fused Attention kernel alone
cannot explain them. CPU FP64's four witnesses are within `1e-10/1e-8`;
FP32 uses `1e-6/2e-3` and BF16 `2e-3/5e-2`. This corroborates a finite-arithmetic
explanation for this bounded schedule comparison without waiving the BF16 failures
or identifying every contributing rounding operation.

The v9 continuation source is `5185dc0d…`; FP64 and tiled use
`0e0023ff773f56920da78374e26ded6f10ceac06`. The CPU v9 outer job failed only when
the later FP64 metadata construction rejected the diagnostic precision label;
its completed FP32/BF16 results are retained. The metadata fix and separate FP64
v10 job are distinct evidence. NPU auto and tiled used admitted physical device 5.
No wider shape or kernel investigation is claimed.

## Reproduction

Use frozen sources and new output paths with the private environment and finite
detached/admitted execution in [execution](../execution-flows.md). Full launch
commands and bounds are retained in the JSON; representative workload entries are:

```sh
python -m scripts.qualify_precision --device cpu --precision fp32 --width 0 \
  --batch 2 --length 5 --steps 3 --threads 8 --checkpoint \
  --save-tensors NEW-fixtures --output NEW-cpu-fp32.json
python -m scripts.qualify_precision --device npu:0 --precision bf16 --width 0 \
  --batch 2 --length 5 --steps 3 --threads 4 --checkpoint \
  --reference /absolute/NEW-fixtures --output NEW-npu-bf16.json
python -m scripts.diagnose_restore --device npu:0 --precision bf16 \
  --configs fc-k fc-s --output NEW-restore-diagnosis.json
python -m scripts.check_continuation --device npu:0 --precision bf16 \
  --output NEW-continuation.json
python -m scripts.check_continuation --device cpu --precision fp64 \
  --output NEW-continuation-fp64.json
python -m scripts.check_continuation --device npu:0 --precision bf16 \
  --attention-backend tiled --output NEW-continuation-tiled.json
```

These diagnostics do not establish task quality, bitwise resumed training,
general BF16 cross-precision equivalence, a resident device scheduler or backend
coverage beyond the recorded CPU/NPU paths. Current implementation and remaining
work are owned by [STATUS](../STATUS.md) and [precision policy](../precision.md).
