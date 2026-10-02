# Original-width Add after private projection borrowing

Clean **44674936efcc4177d4683a7e8542816aec0928eb** completed a nine-card LibTorch
resident TimedDAG/prefill FP32 SGD update with 9,468,053,696 parameters,
D2048/B4/T12/V50304, physical B2 ×2 and two connected windows. This is a cold
original-width pilot, not original B512 training or a formal throughput result.
[Audited record](original-width-add-projection-borrow-20261003.json);
audit: `TASK/launchers/wide_add_projection_borrow_evidence.py <full SHA>`.

The physical grouping, operator chunks, queue/journal limits, 60 GiB/card cap,
53.875 GiB usable allowance and input/parameter rules match the
[preceding B4 pilot](original-width-add-gradient-lifetime-20261003.md).
Private projection borrowing is the implementation change; both runs have
independent construction and numerical execution. No CPU reference supplies events
or gradients. Small-model independent parity is covered by the
[implementation qualification](resident-projection-borrow-20261003.md).

| Measurement | Result |
| --- | ---: |
| Construction | 72.963 s |
| Complete update | 20.770 s |
| Maximum incremental card allocator peak | 43,197,837,312 bytes (40.231 GiB) |
| Outputs / logical events / final cut | 96 / 9,265 / 408 |

All cards and saved-context pools pass memory calibration. Loss and logical work
match the prior B4 result under the existing FP32 policy. Previous peak was
47,394,238,464 bytes; separate leases and one cold sample do not establish a formal
speed recommendation. This pilot has no full-size CPU gradient/update oracle or
new profiler trace. The separate implementation profile remains scoped to its
own smaller fixture.

The unchanged rule projects **20.769816367 × 128 × 1.15 = 3,057.317 s**, exceeding
**3,000 s**. **B512 was not started.** The service terminated successfully and
released all leases. This refusal stays recorded. The rule scales even the fixed,
once-per-update optimizer cost by 128; measuring synchronized sample work and
optimizer work separately is the next diagnostic question. Any revised projection
needs new measurements while preserving the 3,000-second cap and safety margin;
the old estimate is not silently reclassified as a passing run.
