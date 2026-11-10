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

## Independent matched-shape anchors

`fc_tensor_baseline.py` has an independent region schedule and explicit batched
Aggregate/KV Read/Describe/Full formulas. Quadratic allocation enumerates finite
affine intervals independently of the foundation solver, then differentiates
the chosen exact solution. It checks its domain before computation: adjacent
equal-width FC regions, dense inputs, zero left state, qry without positions,
cat/ind/context/cat/quad and L<=256. Full evaluates actual positive selected rows.
It returns the output projection, controls/activity and actual terminal KV fields;
it does not claim full trace, None/zero connectivity or checkpoint equivalence.
The scalar oracle independently checks small outputs, controls, terminal fields
and numeric input/shared-parameter derivatives; a separate bisection and FP64
directional finite differences check the allocation map.

`benchmark_dense_query.py --baseline fc-tensor` keeps the original graph
parameters, inputs, task head, loss and optimizer. `--baseline transformer-16`
(or transformer-4) instead uses a declared serial bias-free pre-RMS causal SDPA
and SwiGLU stack, d128/head2/FFN512. Sixteen blocks match the number of FC nodes;
four blocks match its serial depth. Neither is a function-equivalent Tide graph.
Both retain the original embedding/head; actual total parameter counts are
recorded, not padded to match. No pretrained or model-quality claim is made.

`--mode inference` measures a separate warm-loaded request through prompt
prefill, the last-position language head and greedy first-token selection.
`first_token_seconds` includes synthetic input creation/upload, finite checking
and host return, but excludes service queueing/tokenization/model construction.
It is batch first-token latency, not per-token amortization or later decode.
Training and inference timings are never mixed. Full-shape comparisons remain
numerically unchecked until explicitly recorded, even when small checks pass.

## Default internal private-root packing

The first full-shape counters identified query/region private autograd as the
largest host elapsed categories. `private_rows.py` / `weighted_private_rows.h`
pack each structurally homogeneous private field column before building that
internal graph. Forward reads use actual contiguous view intervals when present;
independent rows are never identified by numerical equality. Output-mask and
allocation-branch groups remain separate in backward, and their column gradients
are sliced only into participating original rows. None/zero, duplicate uses,
parameter aliases and original-version guards remain public invariants.
Tests include literal interval transposes plus partial field roots/duplicate rows,
existing independent KV/recurrent query/region cases, optimizer and exact resume.
