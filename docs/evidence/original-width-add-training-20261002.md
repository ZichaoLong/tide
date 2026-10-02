# Original-width Add training and bounded profiling

Qualified source `475d4afc006117d0f63c297e25c94ead01d6c49a` and the
[window-counter qualification](resident-window-peaks-20261002.md) supplied the
installed standalone consumer. The [audited record](original-width-add-training-20261002.json)
pins binaries, inputs, successful child runs, retained failed parent runs and
profiler CSVs. Audit: `TASK/launchers/original_training_evidence.py`.

All runs use LibTorch resident TimedDAG/prefill, FP32, SGD, physical sample groups
of one and two connected windows per complete update. Each independently
consumes the declared packet. The original logical batch is **512**; the training
stages below deliberately use smaller batches and do not certify B512 training.

| Stage | Cards | Parameters | D / logical B | Construction | Complete step | Maximum card allocator |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial training stage | 11 | 630,573,248 | 512 / 8 | 15.632 s | 23.955 s | 6.088 GiB |
| Original parameter and width scale | 11 | 9,468,053,696 | 2048 / 2 | 249.211 s | 20.279 s | 42.216 GiB |
| Separate instrumented run | 10 | 9,468,053,696 | 2048 / 2 | 201.214 s | 21.890 s | 43.431 GiB |

The original-width runs each produced 48 outputs, 4,632 events and final cut
408, with finite loss 32.08929443359375. The pending peak was 384 and maximum
window events 1,177. Allocator growth and saved-context pools stayed inside their
admitted bounds. Complete-update semantics were independently qualified on small
trajectories; these large runs have no full-size CPU numerical oracle. Different
card counts, cold construction and profiling prevent a causal speed comparison.

Three failed parent jobs remain failed. The first completed D512/B8 but refused
D2048/B2 at backward preflight: the declared global 2 TiB capability ceiling was
partitioned below the required component bound. The corrected attempt raised
that capability ceiling to 8 TiB while retaining the same physical 60 GiB/card
admission and safety margin; it did not allocate 8 TiB. Its B2 child passed, but
the parent stopped at the predefined cost gate: 20.279 s × 256 = 5,191.5 s exceeded
3,000 s. This conservative heuristic includes fixed costs and is not a measured
B512 duration. The first profiling job could not acquire eleven cards within
120 seconds; a bounded ten-card run subsequently completed.

The profiler covered all ten leased devices and recorded **701,139 operators**,
including 690,211 AI_VECTOR_CORE, 5,386 AI_CORE, 5,530 MIX_AIV and 12 MIX_AIC;
**no AI_CPU operator was observed**. Across construction, the complete update
and cleanup, summed device task time was 35.279 s. The largest named sums were
optimizer values 21.646 s, optimizer planning 3.810 s, canonical owner streaming
2.417 s and ScatterUpdate 1.492 s. These sums overlap across devices and streams;
they are not wall-time percentages. Host runtime StreamSynchronize totaled
13.825 s across 5,635 calls, including construction; nested ACL/runtime API
durations must not be added together.

The trace supports investigating vector optimizer work, construction and
communication storage rather than attributing this run to AiCPU fallback.
Canonical stream buffers reserve 25.504 GB across ten cards, or 31.007 GB across
eleven, although each individual packet already streams bounded chunks.
Reusing storage across serially ordered packet groups is a possible general
improvement, pending ordering, replay, error-propagation and allocator checks.
Original B512 training, Attention training and the required full-size CPU/mixed/
resident comparisons remain open. No throughput recommendation follows here.

A later bounded **B4/physical B2** pilot on ten cards, clean `c3ed0f2`,
failed the post-run allocator calibration check: observed allocation exceeded
the estimate. [The retained failure record](original-width-add-b2-refusal-20261002.json)
pins its source, command, planner and logs. It reported neither OOM nor timeout;
the original B512 stage was never entered. That version's error writer discarded
the collected peak and timing details, so they cannot be inferred from this
failed run. Reporting and estimation require diagnosis before another large run.

After the reporting fix, one unchanged ten-card diagnostic on clean `3462dae`
retained the missing measurements ([audited record](original-width-add-memory-diagnostic-20261002.json)).
The diagnostic collector passed; **the consumer still failed** post-run calibration.
D2048/B4/physical B2, 9,468,053,696 parameters, FP32 SGD and the same two windows
completed before that check: 96 outputs, cut408, finite loss31.586021423339844.
Coordinator peak was **51.4198 GiB**, above its **50.0756 GiB** estimate by
1.3441 GiB. Other cards peaked at29.50–29.56 GiB against36.34–36.41 GiB estimates.
The physical cap60 GiB and usable53.875 GiB were unchanged; no OOM occurred.
Construction243.789 s and update39.390 s are diagnostic timings. One host stack
sample during construction caught CPU model finite-value validation; it does
not measure the fraction of time there or indicate an AiCPU fallback.

This motivates removing duplicated coordinator reverse-gather storage before
another original-width check. It does not justify reducing safety margins or
running B512: the same pilot's conservative cost projection still exceeds the
existing3,000 s gate. Failure at this post-run check does not roll back updates.

The phase-scoped gather implementation1757b90 then passed one unchanged ten-card
B4/physical B2 recheck ([audited record](original-width-add-reverse-gather-20261002.json)).
Coordinator allocation fell from55,211,536,896 to49,339,172,864 bytes
(**51.420→45.950 GiB**, saving5,872,364,032 bytes), below the unchanged
53,768,286,884-byte estimate. All ten cards passed calibration. Loss,
work counters, retained-storage counters, requested/effective operator chunks,
and estimates matched the failed diagnostic exactly. Construction254.961 s,
update27.757 s; different leased cards and one cold process prevent a causal
throughput recommendation. The conservative B512 projection4085.875 s still
exceeded3000 s, so the recheck contained no B512 execution. The previous
consumer failure remains failed; this is a separately identified passing run.

After vector optimizer finite checksbb40cff, the first ten-card recheck failed
with **NPU OOM** during canonical-gradient allocation; its external collector
then raised `KeyError(parameters)`. Both failures remain recorded. The device
reported26.11GiB allocated and28.16GiB reserved by this process,1.79GiB free of
61.27GiB, against a3.46GiB request. No contemporaneous process inventory identifies
the other occupancy; do not attribute it conclusively to another job.

One bounded resource recheck with the corrected collector then **passed**
([audited record](original-width-add-optimizer-finite-20261002.json)). It retained
the same model, B4/physicalB2, ten cards, FP32 SGD, two windows and capacity limits;
both attempts explicitly used8 ATen CPU threads instead of the earlier1. Loss,
all work/retained counters, estimates and measured allocation peaks match1757b90
exactly. Construction66.567s and update25.073s are cold observations. The changed
thread count and separate leases prevent a causal whole-graph speed claim, despite
the independently measured isolated optimizer improvement. The conservative B512
projection is3690.815s, still above3000s; **B512 was not executed**. Further
implementation work is required before another full-batch admission.
