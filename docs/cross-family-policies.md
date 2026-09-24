# Cross-family streaming and prefill policies

This extension preserves the Tide contract in [semantics.md](semantics.md).
Execution policies are outside graph/model/checkpoint identity. CPU FP32/FP64,
first-order AD; native defaults remain replay. The original foundation-v1 suite
and its evidence remain historical; foundation-v2 has separate definitions.

## Clients and independent schedules

Python `reference.run` and `specialized` retain scalar independent schedules.
`streaming.run` batches independent sample events at each tick. `frontier.run`
uses complete dependency frontiers. `specialized_blocks.run` owns fixed chain/
diamond layers; `settle_chain`/`settle_layered` own independent block schedules.
They share local block formulas, never call a generic executor as specialization.
`settle.run` remains the general region-major SettleGraph interpreter.

All Python block clients accept `packed`, `prefill`, `full_autograd` and
`aggregate_autograd`. Streaming has no time-prefill argument. One causal tick
can batch state preparation across samples without crossing any Next boundary.
`packed=False` disables Full/Aggregate grouping; legal time sequences may remain
per sample if prefill=True. Disable both for scalar steps. Native block execution
now also honors packed=False for Full/Aggregate (older versions always grouped
those two operations). This is an execution-policy correction, no semantic change.

Native Streaming, Frontier and independent chain/diamond reuse the same local
kernels. Native `SettleExecutor(spec,model,options,algorithm)` accepts `frontier`
(default) or `streaming`. Both consume/return encoded continuations; callers can
switch at complete position cuts while keeping the encoded state/ledger. Native
construction/encoding remains C++ owned. A body projection is not an inverse
encoding, and cannot reconstruct an arbitrary encoded continuation.

## Isolated Python VJPs

`full_autograd=batched` batches affine rows; nonlinear and Emit roots remain
per event. `aggregate_autograd=batched` batches tagged source arithmetic while
keeping separate summary/contribution roots. Both require packed=True and the
built-in ProjectionEmit/SourceAggregate implementations; explicit unsupported
programs fail even under no-grad. No-grad/inference retain numeric batch paths.

Python custom Functions receive independent Tensor arguments and return
independent roots, disable materialized gradients, and select defined cotangents
without testing numerical zero. Unused rows stay None; connected zero stays
connected. CPU FP32/64 only; create_graph backward is rejected. Normalization
keeps per-event softmax Jacobians and per-event weighted-mean graphs, preserving
near-zero AdamW behavior. With `packed=False`, scalar programs execute directly
under autograd; replay counters remain zero because no detached numeric batch is
being rebound. Retain the independent scalar formulas as the oracle.
State/Read semantic replay remains and must be reported in training cost.
Native state_evaluate.cpp is shared by tick streaming and causal block waves;
prefill-disabled, selected-adoption, clear and custom-Next cases still batch
independent samples of one node at one time. Selection/Next complete before
the following time begins. Counters distinguish causal batch from time prefill.

## Unified entry and option boundaries

```bash
python scripts/benchmark_foundation.py --device cpu \
  --suite benchmarks/foundation-v2.json --tier smoke --build \
  --output-dir artifacts/foundation-v2-smoke
python scripts/benchmark_foundation.py --device cpu \
  --suite benchmarks/foundation-v2.json --tier medium --ids TR01 TR02 \
  --variants native-frontier-optimized native-settle-optimized \
  --output-dir artifacts/foundation-v2-train
```

Use the selected Torch Python, clean source for timing, and unique output paths.
`--describe` prints configurations/counts without execution. `export_foundation.py`
exports all required sources and definitions for a target-local rebuild.

`foundation_policy.py` separates scheduler options from kernel policies and
Model projection layout. Native Settle gets scheduler-only core.Options; its
already configured kernels survive embedding. Graph-only runs have no vocabulary
head; head_workers=0 is explicitly recorded rather than forwarded to the graph.

Options: `--[no-]packed`, `--[no-]prefill`, `--full-autograd replay|batched`,
`--aggregate-autograd replay|batched`, `--[no-]packed-sources`, `--[no-]batch-next`,
`--[no-]parallel-regions`, `--[no-]compact-events`, `--[no-]defer-state-release`,
`--attention-packing exact|single`, `--fiber-pooling event|csr`,
`--fiber-cache cloned|owned`, `--attention-layout event|head`,
`--projection-layout input|linear`, plus workers/threads.
Explicit policy overrides require v2; they do not mutate v1 presets.

The optimized preset chooses packed Full/Aggregate VJPs, plus native transport,
Next, cleanup and independent-region policies. Same-fiber native profiles also
select single/CSR/owned/head/linear. Python implements packed batch/sequence and
isolated VJPs, but native-specific policy requests fail as unimplemented. Python
node parallelism is not required. Single/CSR/cache/head settings on event GQA,
SSM or Add fail as semantically inapplicable. A request is not evidence of use.

Worker records contain requested and resolved groups. Each measured segment
contains actual sequence/batch sizes, kernel calls, replay and fallback counters.
Clear, selected-only adoption, custom Next and missing exact state sequence
contracts retain causal fallback. Scalar recurrence work inside a sequence is
separately counted. Trace retention can suppress deferred release. No counter
or preset name implies a speedup or a particular worker occupancy.

## Measurement boundaries

V2 fixes at most12 logical workloads/6 variants/3 independent process repeats.
Smoke is D16/B4/T6. Transition workloads measure a prefix followed by individual
streaming positions, with separate phase, interval and effective-position
denominators. A streaming baseline can process that same prefix without time
batching; counters distinguish it. Timing separates forward with/without grad,
backward, optimizer and whole train step. Training uses declared detached windows.
Warmups reset parameters, state, messages and ledgers; profiling is a separate
pass. V2 training-window runs save separately hashed measured/profile
observations containing final canonical owner values, final-window gradients and
alias groups. These small audit artifacts are not resumable checkpoints. Coordinator phase profiling remains Streaming-only; operator counters
are shared. Existing resource discovery enforces dynamic half CPU/memory across
the task; CPU is required, Ascend is deferred and limited to8 cards if added.

Large graph-only presets compare legal ranked topology, not the exact LH
inet/onet function. Owner counts and target rounding are explicit. Wide Add uses
the tick-repeat Add state plus SwiGLU; it is a parameter-scale/workload comparison,
not identical per-token matrix work to the LH-style Add benchmark. Both families
retain input/output encoding overhead in native Settle measurements.

Implementation, passing correctness gates, actual performance and final
qualification are tracked separately in ROADMAP/STATUS. This contract does not
itself assert completed performance validation.
