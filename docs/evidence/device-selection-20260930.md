# Device selection and grouped queue commits — 2026-09-30

Clean source `6220011f1843687a634f81ffc96b62ad77b3a006` passed its component
build, four CPU CTests, standalone loader closure and all12 NPU component cells.
The [manifest](device-selection-20260930.json) retains exact source/binary/result
hashes and four terminal jobs. The unchanged standalone core is reused from
clean eff5945 only after matching both core source and binary fingerprints.

The39 selector cases compare with the independent CPU `RegionKernel` for
count-v1/positive-v1. They cover count priority on/off, zero/multiple selection
budgets, stable ties, full-candidate softmax including nonpositive candidates,
exact counts and times above2^55, missing versus present-zero history, sequential
history continuation, downstream refusal, invalid metadata, nonfinite scores and
selected-count overflow. Scores are supplied FP32 descriptors; producing them
from real graph modules is not part of this component's qualification.

Queue cells each include46 original cases plus six grouped cases for FP32 and
FP16. Cross-queue payload swaps prove that proposals snapshot before any commit;
a later queue capacity failure preserves both queues and counters. All participants
must share one sticky error and finish every proposal before beginning commits.

| Separate placement profile | Custom tasks | Recorded device tasks |
| --- | --- | --- |
| Selector |39 frame-selection tasks |428 AIV |
| Queue proposals and grouped commits |63 queue-proposal tasks |475 AIV |

Both msprof runs and profiled applications passed without a host fallback warning.
Profiles include setup and CPU assertions; they do not measure steady-state
throughput. Scalar AIV metadata loops still require performance work. Other
component cells retain their earlier finite contracts; no full graph loop,
attention, general region program, peer progression, training, VJP or optimizer
support follows from these results. Failure-injection lifecycle qualification
remains pending. New content/state loop development is outside this source.
