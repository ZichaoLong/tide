# Original-width Add after greedy chunk selection

Clean source **e82f97199c63dc8505137595adaae9a9c2d12221** completed one
nine-card LibTorch resident TimedDAG/prefill training pilot. The
[audited record](original-width-add-chunk-selection-20261002.json) pins the
source, installed binary, helper, packets, static plan, raw results and retained
queue failure. Audit: `TASK/launchers/wide_add_chunk_planner_evidence.py <full SHA>`.
The [consumer qualification](consumer-chunk-selection-20261002.md) remains the
independent small-model correctness anchor; this pilot adds scale evidence.

The unchanged original packet has 480 body nodes/2,208 body edges,
D2048/B512/T12/V50304 and **9,468,053,696 Add parameters**. The executed pilot
changes only its logical batch to **B4**, split into four physical B1 groups.
It completes two connected windows and one FP32 SGD update, including loss,
backward, finite checks and synchronization, with eight ATen CPU threads.
It has no warmup and is not a formal throughput recommendation.

The preceding ten-card request failed its configured 120-second queue limit
(136.61 seconds observed including polling/probes); no consumer started.
One resource adaptation used nine cards and the previously executed B1
queue/arrivals512, outputs64, trace2048, KV256 and KV-trace8192 capacities.
The logical topology, width, inputs/initialization rules, dtype, loss, update
boundary, 60 GiB/card cap and original safety margin were preserved. This is a
new passing run; the ten-card timeout remains failed.

Before numerical execution, the generic planner used the original B512 geometry
and selected Full16/emission4/aggregate8/attention8/keys128/reverse1/head64.
The maximum static B512 estimate was **53.012 GiB**, below **53.875 GiB usable**.
The head and saved-context limits were each 4 GiB. This shape-only plan is not
evidence that B512 actually completed or a numerical prepass of its events.

| Observed pilot measurement | Result |
| --- | ---: |
| Construction, separately timed | 65.793 s |
| Complete B4 training update | 27.300 s |
| Maximum incremental card allocator peak | 44,127,544,320 B (41.097 GiB) |
| Loss | 31.586036682128906 |
| Outputs / logical events / final cut | 96 / 9,265 / 408 |
| Pending peak / maximum events in a window | 384 / 1,177 |

Every card passed allocator calibration, and every saved-context pool stayed
inside its admitted budget. Parameter count, outputs, logical events and final
cut match the prior ten-card B4/physicalB2 result. Loss differs by
0.0000152587890625, inside the existing FP32 atol=1e-6, rtol=1e-5.
Changed physical grouping and placement legitimately change physical work and
retention counts; they were not required to match. There is no full-size CPU
gradient/update oracle or new operator profile in this run. No CPU-fallback
warning was observed; that alone is not an AiCPU attribution.

The unchanged cost rule projects **27.299887766 × 128 × 1.15 = 4,018.543 s**,
above the **3,000 s** limit. **B512 training was not launched.** This is a
conservative admission projection, not measured B512 latency. Different card
counts, batches and leases also prevent a causal speed or memory comparison
with the earlier 25.073-second ten-card pilot. Larger retained operator maxima
have not established a whole-graph throughput improvement.

Both jobs are terminal and their device leases are released. Further work must
address the full-size training cost and retained-gradient/Attention memory
lifetimes; repeating the same pilot or relaxing the gate is not completion.
