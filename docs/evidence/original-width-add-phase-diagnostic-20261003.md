# Original-width Add synchronized cost diagnostic

Clean **26176de888013fda5eccfe039fa504e87c2e7e95** passed one nine-card
LibTorch resident TimedDAG/prefill FP32 SGD complete update. The unchanged packet
is D2048/B4/T12/V50304, 9,468,053,696 parameters, physical B2 ×2 and two connected
windows. This uses the [qualified phase timer](consumer-phase-timing-20261003.md).
[Audited records](original-width-add-phase-diagnostic-20261003.json); audit helper:
`TASK/launchers/wide_add_phase_diagnostic_evidence.py <full SHA>`.

| Observation | Seconds |
| --- | ---: |
| Construction | 68.780819999 |
| Synchronized sample work | 19.105143379 |
| Final checks / optimizer / publication | 1.556835798 |
| Complete update | 20.661979177 |

All prior capacities, operator chunks, 60 GiB/card cap and 53.875 GiB usable
allowance are unchanged. All allocator/context checks pass; peak is 43197837312 bytes.
Outputs96, events9265, final cut408 and loss31.58603858947754 match the preceding
B4 result under the existing FP32 checks. Candidate execution is independent;
no CPU reference supplies events, routes or gradients.

The old total-based estimate remains **3041.443335s >3000s**.
Separating the once-per-update work gives
`(sample_work*128 + optimizer)*1.15 = 2814.067467s`, below the
unchanged3000s cap. This estimate assumes B512 has128times the B4 sample work at
physicalB2, with one same-shape SGD update. Input-dependent work, context
residency and external load can change scaling. The1.15 margin and all memory
checks remain; the estimate is admission for a separate bounded execution,
not proof of B512 correctness, time, or a formal throughput recommendation.

**This diagnostic did not execute B512.** Its service ended exit0, empty control
group, and released its nine-card lease. Earlier refusals remain unchanged.
No profiler was collected; phase timing adds its declared synchronization.
The user renewed continuous F1–F7 execution authorization after the handoff audit;
this evidence checkpoint is not a pause.
