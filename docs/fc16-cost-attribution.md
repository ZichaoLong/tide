# FC16 cost attribution and matched tensor baselines

The frozen downstream FC16 task is defined by `benchmark_dense_query.py`:
four adjacent regions of four KV nodes, B8/L128/d128, two heads, FFN512,
qry/cat/ind/context/cat and regional_total/quad, temperature .25. Its k=2
constrains sending strength, not a hard active count. Dense edges do not imply
all candidates execute Full. The original task/parameters/loss/AdamW and
minimal versus complete returns keep separate identities.

`--profile stages` runs a separate instrumented update after restoring the
same initial parameters/optimizer and uses the measured input seed again.
The native `operator_profile.h` records exclusive calling-thread elapsed
time at preparation, state snapshots, queries, region computation, publication
and the corresponding first-order adjoints. Read these counters only after
workers finish; they include incidental host waits, are not device kernel
durations, and cannot be summed into coordinator wall time across workers.
They do not introduce per-operation device synchronizations. Disabled counters
perform no clock reads. `--profile cpu|npu` additionally records named
`tide::stage::*` ranges; shape retention is opt-in because full-shape profiles
can be large. The uninstrumented complete training update remains the cost.

The source-side adapter exposes reset/snapshot only; standalone clients can
use the same C++ header independently of Python. `test_stage_profile.py`
checks real native query/VJP work and unchanged independent roots. Existing
query, state and allocation contracts continue to own the mathematics.

Throughput divides actual input positions by complete wall time. FC16 has
1024 positions, while staggered graph generators count their actual positions,
not B times maximum length or repeated input ports. This is distinct from
TTFT, which requires a separately timed inference request and output head.
Training forward alone does not certify inference latency.
