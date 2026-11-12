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

## Matched FC comparison and diagnostic boundaries

`benchmark_dense_query.py --fc-shape fc4|fc16` selects two regions of two nodes
or the original four regions of four nodes; d128/H2/FFN512 and the strength ratio
remain fixed. Transformer2/4/16 anchors distinguish serial depth and node count.
`--phase-fences step` times a natural complete request/update with synchronization
at its boundaries, including actual checks/host return. Its internal phase times
are host enqueue/wait intervals, not synchronized device phase costs. The default
`phase` preserves the original per-phase fences; compare that explicit identity
separately when estimating observer cost. Profiling is always another pass.

`run_fc_comparison.py` supplies finite main, CPU-thread, batch/sequence-axis,
boundary, profile and projection-check suites. Choose the suite that answers the
current question; these are not a mandatory regression Cartesian matrix. CPU and
NPU profiles can observe graph, direct FC and Transformer implementations.
Formula ranges label Aggregate, Prepare/Read, regional head/solve and Full;
`stage-operators.json` attributes CPU ATen events to their nearest named range,
leaving unrelated worker events unattributed. Nested totals are not wall shares.
Large NPU traces may use `--profile-output` on bounded project-owned scratch;
small reviewed results, source/config identities and job records stay durable.

The Python-only `--diagnostic-boundary query-direct|region-direct|both-direct`
temporarily removes selected independent-root wrappers while retaining the local
numeric functions and graph schedule. This intentionally omits a public duty;
it is not a production mode or complete-root qualification. Its ordinary task
output/derivative projection must be checked before interpreting its cost.
`--formula-attention sdpa|sdpa4d` changes only the direct formula's selected Full
Attention operator (three-dimensional inputs or an explicit leading batch axis);
`manual` remains the default independent explicit algebra. An SDPA API call does
not prove fused dispatch: shape, bias gradients and backend capabilities must
be checked in the actual profiler and derivative comparison.
Neither option changes the default foundation executor.

`check_fc_comparison.py` reads the same pinned downstream task and compares
actual hidden/logits/loss, terminal KV fields and input/parameter gradients.
Small `--trace` cases additionally compare every candidate control/support.
None/present gradients, finite numeric errors, integer/support mismatches and
execution errors are recorded separately. The full-scale minimal-return check
does not certify all intermediate roots, complete records, optimizer or restore.
`--dtype float64 --initial-dtype float32` promotes the original FP32 task
parameters rather than drawing another FP64 initialization. Diagnostic wrapper
omissions report actual read/full/region invocation and row counts, so an unused
patch is distinguishable from an observed implementation boundary.
`test_fc_comparison_diagnostics.py` checks the numeric projection against the
independent formula, SDPA derivatives and patch restoration after failure.

The [FC4/FC16 three-layer investigation](evidence/fc-three-layer-comparison-20261111.md)
records actual-scale CPU/NPU checks, same-FP32-material promotion, phase-fenced
versus natural complete intervals, wrapper omissions and real SDPA dispatch.
Full-scale FP32 strict failures remain; these costs are exploratory, not a
strict-equivalence acceleration qualification. `collect_fc_comparison.py`
collects the completed finite suite/job layout into reviewed evidence without
rerunning any model. Source records and the downstream's historical dependency
pin are separate identities; the actual job/build source governs the run.

## Independent matched-shape anchors

`fc_tensor_baseline.py` has an independent region schedule and explicit batched
Aggregate/KV Read/Describe/Full formulas. Quadratic allocation enumerates finite
affine intervals independently of the foundation solver, then differentiates
the chosen exact solution. It checks its domain before computation: adjacent
equal-width FC regions, dense inputs, explicit raw left state, qry without positions,
cat/ind/context/cat/quad and L<=256. Full evaluates actual positive selected rows.
It returns the output projection, controls/activity and actual terminal KV fields;
it does not claim full trace, None/zero connectivity or general graph checkpoint equivalence.
The scalar oracle independently checks small outputs, controls, terminal fields
and numeric input/shared-parameter derivatives; a separate bisection and FP64
directional finite differences check the allocation map.

`run(..., start=absolute_position, left=node_raw_fields)` keeps original int64
times, nonzero KV state and retained cross-window adjoints. Common baseline state
validation is in `tensor_baseline_state.py`; it never calls an execution reference
to reconstruct state. `test_tensor_baseline_continuation.py` adds independent
scalar left-state/shared-parameter roots, three actual SGD/momentum updates,
cross-window backward and a fresh-process raw-state/model/optimizer restore.
That checkpoint is for this output/state projection, not a substitute for the
public graph checkpoint's full metadata, history and pending messages.

Full packs selected queries across samples into `[B,H,Qmax,Dh]` and contracts
against one K/V bank per sample. Query/output/FFN projections use only selected
rows; attention pads to the maximum selected count. It neither expands history
per query nor dispatches Full in a Python B/L loop. Dynamic shape extraction,
padding and node/source loops remain; this is not an optimal-kernel claim.
The [B-loop removal comparison](evidence/fc-batch-full-20261112.md) records
independent literal Full VJPs, stateful CPU checks and CPU/NPU three-mode costs.
Read and Full consume the same raw KV bank but retain their separate formulas.
The [stateful qualification](evidence/additive-interface-qualification-20261110.md)
separates this projection from public complete-root and restore checks; the
[three-level comparison](evidence/additive-interface-performance-20261110.md)
retains local forward gains, independent-adjoint costs and the unresolved FC16 gap.

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
pack each structurally homogeneous regional private field column before building
that internal graph. Forward reads use actual contiguous view intervals when present;
independent rows are never identified by numerical equality. Output-mask and
allocation-branch groups remain separate in backward, and their column gradients
are sliced only into participating original rows. None/zero, duplicate uses,
parameter aliases and original-version guards remain public invariants.
Tests include literal interval transposes plus partial field roots/duplicate rows,
existing independent KV/recurrent query/region cases, optimizer and exact resume.

A contiguous buffer alone is insufficient: temporal length buckets can cross
independent sample intervals. Falling back to cat of every view then creates
one full-buffer slice transpose per row. The packed boundary therefore uses
one exact index gather for noncontiguous intervals of the same actual base;
its transpose accounts for repeated indices. Strided or independent bases keep
the normal path. `test_ragged_bucket_crossing_owners_has_one_gather_transpose`
checks both the literal derivative and absence of per-row slice transposes.

The tested all-query column candidate still increased full FC16 CPU/NPU time
after that correction. Default queries therefore retain their previous independent
private leaves; region columns keep the measured reduction in regional adjoints.
`independent_copies.py` / `weighted_independent_copies.h` publish query/region
outputs via foreach copies into separate allocations. Cloning a whole batch and
returning ordinary views would couple versions and resize effects, so is not an
equivalent replacement. The copy tests compare bits/values and exact pre/post
mutation derivatives; scalar-formula numerical checks remain separate gates.

## Repeating the investigation

For the block-interface decision, `run_fc_comparison.py --suite blocks` runs
FC4/FC16, native auto and explicit pre-block packed-v2, direct FC and matched-node
Transformer, inference/grad-enabled-forward/training, plus Python default training.
`--batch`/`--length` select the bounded workload. Use a frozen checkout, explicit
native build, downstream source, device, threads and a fresh output directory;
profiling (`--suite profiles`) and checks (`--suite checks`) are separate processes.
This retains the same timed preparation/check/optimizer/synchronization boundary.
The original generic schedules remain available; the workload is not a custom
algorithm hidden in default dispatch.

`scripts/cost_comparison.py` reviews explicit result pairs without a fixed date,
job name or old run directory. Its small manifest is:

```json
{
  "schema": "tide-cost-comparison-v1",
  "question": "What cost remains between the default path and its formula anchor?",
  "records": [
    {"name": "default", "path": "default/result.json"},
    {"name": "formula", "path": "formula/result.json"}
  ],
  "comparisons": [
    {"candidate": "default", "reference": "formula", "relation": "task-projection"}
  ]
}
```

Run `python scripts/cost_comparison.py --manifest MANIFEST.json --output REVIEW.json`.
Allowed relations are same-function, task-projection and different-function;
Transformer uses the last. The collector preserves source/build/hash, configured
axes, actual work, phases, throughput arithmetic and strict/restore status. Failed
executions produce no throughput. Mismatched B/L/mode/device/dtype/thread/timing
axes are refused; same-function comparisons also require identical starting
parameter bytes. A cost ratio never implies correctness qualification.

Dense-query and independent KV/SSD layer-chain results already have adapters.
A future LH-like or other-path investigation reuses the measurement/runner owned
by that workload, adds a narrow record adapter if necessary, and supplies explicit
pairs. It does not need another task ledger, copied profiler or hard-coded job
collector. A new direct baseline must still declare its formula and return/root
domain; an adapter cannot confer independence or recover missing observations.
Keep the benchmark/profile/check commands and terminal job identities alongside
the review. The [block contract](weighted-query-blocks.md) names its actual dispatch
and private-graph counters; the complete return/gradient/restore tests remain
separate from these cost tools.

The same manifest can include `diagnostics`, independently of its cost pairs:

```json
[
  {"name":"small-native-profile", "kind":"profile", "path":"small-native/result.json"},
  {"name":"formula-check", "kind":"projection-check", "path":"check/result.json", "source":"FIXED_SHA"},
  {"name":"root-tests", "kind":"junit", "path":"qualification.xml", "source":"FIXED_SHA"},
  {"name":"measurement-job", "kind":"job", "path":"job/status.json"}
]
```

`execution_diagnostics.py` reads existing stage counters, CPU ATen shapes, NPU
exports and outer/private autograd census, preserving each record's own shape
and implementation identity. `exports` can explicitly relocate an NPU
`ASCEND_PROFILER_OUTPUT` directory; it does not relabel the measurement. Preserve
the reviewed small JSON before retiring scratch exports. No model is rerun by
this collection step. Failed tests, failed projection checks and terminal job
success remain separate entries; a live job cannot enter a terminal review.

Cost pairs report differences in actual work counts and corresponding synchronized
phase durations. Physical visibility, affinity, numerical policy, placement and
observation differences remain explicit possible confounders within each pair.
They do not subtract host-enqueue phases as device times. Equal
work counts do not certify equal support; input throughput uses actual recorded
positions, and TTFT uses its actual first-token interval when present. The
complete cleanup interval is never substituted for TTFT. For other workloads,
add a small explicit schema adapter, retain source/input/placement identities and
declare the comparison role; the collector makes no numerical-equivalence claim.
