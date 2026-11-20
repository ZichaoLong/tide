# Shannon active-face arithmetic for the R3 training failure

The downstream `tide-learning-lab` R3 campaign reached a real FP32 failure after
603 completed updates. Its saved input, parameters and private state reproduce
the same guard on CPU. The first regional call contains scores
`[6.040277004241943, -10.60705280303955]`, zero floors, unit caps and total one.
An independent 80-digit Decimal KKT solve gives
`[0.9999999410944113, 5.8905588703119336e-8]`. The weak sender is representable;
this case is neither final FP32 range exhaustion nor an NPU-only FTZ diagnosis.

The previous multiplier solve rounded the strong sender to its cap, then
reconstructed the remaining face using `target - sum(capped) == 0`. That erased
a positive weak sender. Related rows can also lose the tail when deciding
whether the initial total is slack, or when subtracting several actual caps.

## Current evaluation

Only constrained Shannon changes. Quad and pool retain their numerical paths.
Shannon uses a finite bounded exponential active-set solve. Each iteration
normalizes the remaining coordinates to the remaining total, then observes
the mass after applying their bounds. If that mass exceeds the target, every
below-floor coordinate is necessarily on its lower face at the solution.
If it is below the target, every above-cap coordinate is necessarily capped.
Only that proven set is fixed before repeating; an unfinished iteration must
remove a coordinate. Failure to certify progress remains an explicit error.

Pairwise two-term sums retain a tail beside a much larger total. The residual
after fixed bounds retains both terms; scaled exponent products are restored
only after normalization. Bound comparisons retain product rounding errors
and the normalizer's tail. This is arithmetic organization in the requested
FP32/FP64 dtype, without a CPU arithmetic fallback or a new public mass format.

The selected mathematical face is saved separately from the stored value.
An interior value may round to its cap; this does not make the cap an active
constraint. Actual cap/floor ties retain the declared boundary VJP, including
fixed-through-mu and slack-total conventions. No nextafter adjustment, floor,
epsilon, support clipping or changed recovery threshold is used.

For the coupled Shannon VJP, cotangents are centered at an actual interior
candidate before taking the weighted mean. Retaining these differences and
binary exponents prevents `1 - round(1 - tiny)` from erasing a representable
score derivative. Public values and gradients retain their existing dtype.

## Evidence and limits

CPU checks cover the real row, 16 candidates with budgets 1/2/4/8, true caps
and floors, small positive tails and gradients, cancellation in `.1 + .9`,
the exact cap convention, broadcasting/version guards, and strict rejection of
truly unrepresentable positive senders. Existing independent allocation, range,
cap and reachability checks remain part of the targeted suite:

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/test_allocation.py tests/test_allocation_range.py \
  tests/test_shannon_cap.py tests/test_shannon_active_set.py \
  tests/test_numerical_reachability.py
```

This suite passed 143 checks on the current CPU source. Another 32 fixed-seed
16-candidate bounded cases matched independent Decimal values and analytic
VJPs. Downstream `pilotlab.diagnose_shannon3` retains the original failure and
completed three full CPU updates from its saved input. A fresh-process repeat
of the final update matched parameters, gradients, AdamW state and private
state exactly across 706 fields, under unchanged `atol=1e-5`, `rtol=1e-4`.
Those repeated-input diagnostics are not learning results or qualification of
the campaign's original checkpoint as a new-source resume.

The frozen `allocation.py` with SHA256
`a8e4346837e7db763d3b778126090c82719c8192679606b4b1b508e576a4e74e`
passed limited NPU FP32 qualification: 18 fixed value/support/score-VJP/guard
cases, followed by three updates using the real captured failure input. A new
process restored the diagnostic prefix saved after update two and replayed
update three. Loading all 602 model/optimizer/private-state fields and RNG was
exact. The resulting 706-field update trajectory was finite and within the
original `atol=1e-5`, `rtol=1e-4` thresholds, with maximum absolute difference
`3.725290298461914e-9`; the trajectory was not bitwise exact.

The content-bound records are in the sibling `tide-learning-lab` repository:

- [NPU repair review](../../tide-learning-lab/docs/evidence/pilot-r3-shannon-npu-repair-review.json),
  SHA256 `f1cc59521560efbef687d510e82c7f85e960818be4d3c597b0044c80225d9d4d`.
- [Transfer compatibility review](../../tide-learning-lab/docs/evidence/pilot-r3-shannon-repair-transfer-review.json),
  SHA256 `6e73686118db5f4773a285b59091c4602bbbf78f44817b20259ed5dec1821c94`.
  Quad/pool code projection is unchanged; transferring stage-A evidence still
  requires the separate exact three-update proof for all 24 selected models.

These repeated-input diagnostics do not certify learning quality, throughput,
long-trajectory recovery or migration of the original campaign checkpoint.
Old source/device outcomes and recovery failures retain their original identity
and thresholds; the repaired-source result does not relabel the failed run.
There is no general all-range guarantee: genuinely unrepresentable positive
senders still raise, and interior slopes still use stored c rather than a new
unquantized log-mass representation. Long trajectory stability, recovery and
performance need source-specific evidence.
