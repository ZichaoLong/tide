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
