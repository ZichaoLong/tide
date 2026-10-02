# Original-wide Add resident inference execution

Source `be380db7451742c3fe50b57cb02263e81c64a4d6`, frozen
`source-values-clean01`; qualified standalone consumer `source-values-npu-clean01`
with the qualified retained-journal library. Reviewed raw receipts and hashes:
[JSON](original-wide-add-inference-20261002.json). Audit helper:
`launchers/wide_add_evidence.py` under the external task root.

`wide-add-inference02` passed with exit code 0 and released all eight leased NPUs
at 2026-10-02 06:39:49 UTC. Logical devices 0–7 ran FP32 LibTorch resident
TimedDAG/prefill, locality placement and aggressive safe chunking. The common
packet retained 480 body nodes, 2,208 body edges, D2048/B512/T12/V50304 and
**9,468,053,696 parameters**. The logical batch used 128 physical B4 groups;
there was no logical batch or KV reduction.

| Observation | Result |
| --- | ---: |
| Construction | 182.037855 s |
| One complete inference step, two connected windows | 278.573709 s |
| Output tokens | 12,288 |
| Events | 1,183,429 |
| Final continuation cut | 408 |
| Finite loss | 30.508380889892578 |
| Maximum per-card allocator peak | 8,387,755,008 bytes (7.812 GiB) |
| Process CPU peak RSS | 100,636,151,808 bytes (93.725 GiB) |

All per-card allocator growth was within the recorded admission estimates;
all compact continuation pools stayed within their individual budgets. Input,
model count, native packet, binary, output coverage, cut and terminal status
were checked against immutable records. No CPU fallback was reported.

This is one cold-process capacity/execution observation, with no warmup,
repetitions, full-size CPU oracle or profile. Independent correctness comes from
the prior small/representative qualification. It supplies no formal throughput
recommendation and does not isolate initializer speedup by comparing a different
model's construction time. The earlier [Attention execution](original-wide-inference-20261002.md)
also passed at its own recorded source. Complete training, other required flows
and finite full-size CPU/mixed/resident comparisons remain open. Failed
`wide-add-inference01` is retained; its prerequisite failure occurred before
NPU allocation and was not relabelled.
