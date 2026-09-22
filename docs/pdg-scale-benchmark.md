# Comparable-scale PDG Attention benchmark

The user requests a performance comparison without importing LH weights.
`scripts/pdg_scale_topology.py` reads only the four retained LH CSR blocks;
`tidegraph-scale-bench` instantiates an ordinary single PositiveDelayGraph and
runs the existing native StreamingCursor. This does not establish LH numerical
equivalence. Fresh seeded parameters and fixed token IDs can yield different
routes and cache lengths from the original LH unseeded random-token loop.

## Workload

Two independent body state namespaces share the static graph structure; a
separate readout node collects two body phases per token. Periodic local clocks
and phase-specific positive-delay edges use the existing single-PDG encoding.
Four-head fiber Attention, all-softmax source pooling, decay .01, SiLU/RMSNorm,
clear-after-selection and LH count/affect regions match the chosen module family.
The base region budget and local size come from the graph input. Forced hubs
mean nominal leaf activation is not the global realized selected-node ratio.

All attention, per-edge signaling, norm, source-pool, embedding and vocabulary
head parameters are unshared between logical owners. Physical phase aliases do
not multiply parameters. Biases, schema placeholder tensors and identity readout
wires are non-learned scaffolding. For N static nodes, E logical edges, width D,
vocabulary V and L body ticks, the model has
`(4*(2*N+1)+E)*D*D + 2*N*D + E+1+L + 2*V*D` learned elements. The retained wide
graph at D=2048/V=50304 therefore has exactly17,269,426,339 elements, matching the
original LH parameter count without loading its weights.

`--emission row` supplies a benchmark-specific hard-mode FullKernel: one dense
Linear per selected node row, with slices delivered through the graph ports.
It combines that node's intra-cortex and bridge outputs into one matrix.
`--emission slot` uses the existing per-slot kernel and serves as an independent
small numerical anchor. Neither path replaces the generic streaming schedule.
Row Emit is not a new general training profile: softp/HST are explicitly rejected.

## Measurements and validation

Each process constructs one model and advances growing state for a finite token
window. A declared warmup prefix is retained, recorded and excluded from summary
means. Tokens are fixed IDs `(3*sample+7*token)%vocab`, with no greedy feedback.
One token includes embedding, all body/readout ticks and the vocabulary head.
Logit validation, checksums and metric writing are outside timing. Construction
and whole-process peak RSS are separate; caches allocate on demand without LH's
eager initial per-node KV reservation. No final snapshot is taken during timing.
Grad-forward builds autograd and retains its connections across steps; it does
not call backward, optimizer or detach. Packed semantic replays are counted.

Report milliseconds per batch step and amortized milliseconds per sample-token,
raw growing-context time series, candidate/selected/source/edge counts, packed
call counts and maximum node batch. The latter distinguish model scale from
realized compute. Initial source seed alone does not make this an exact LH
workload. FP32/FP64 tiny tests compare trace, state, history, pending and routes
across scalar slot, scalar row, packed row and parallel packed row.

## Entry points

```sh
python scripts/pdg_scale_topology.py --graph-input PATH/input.json --output-file PATH/wide.txt
python scripts/build.py --jobs 4
python scripts/benchmark_pdg_scale.py --device cpu --dtype float32 \
  --topology PATH/wide.txt --width 2048 --batch 512 --steps 12 --warmup 4 \
  --workers 160 --threads 1 --packed 1 --grad 0 --emission row \
  --timeout-seconds 1800 --memory-gib 1280 --output-dir artifacts/NEW_RUN
```

Run from frozen source using the durable workflow. At most160 selected CPUs,
Explicit node/ATen thread counts bound the requested parallelism. OPENBLAS1
is a requested environment setting, not proof of the effective BLAS pool;
OpenMP-built BLAS may honor OMP_NUM_THREADS instead. Inspect runtime pools; see
[evidence/pdg-scale-profile.md](evidence/pdg-scale-profile.md). The address
space bound is not measured RSS. Each run owns its topology copy, configuration,
binary/source identities, raw metrics, log, terminal summary and optional
best-effort Trackio projection. Missing Trackio does not discard local records.
Preserve timed-out/partial cases and their missing observations. Measured results
belong in evidence; implementation alone is not performance evidence.

## Optional optimized schedules

Add `--head-workers 160 --parallel-regions 1 --compact-events 1` to select the
column-parallel vocabulary head, independent region tasks and compact event
grouping/cleanup. All default off; compare against the same binary with
`--head-workers 1 --parallel-regions 0 --compact-events 0`.
The [optimization contract](streaming-optimizations.md) explains state ownership,
autograd and the separate graph/head worker phases. Each phase observes the
worker × intra-op budget; idle pools can make total OS thread count larger.
`runtime/*` metrics report the pools actually visible to the native runtime,
with `-1` for unavailable introspection. They do not measure simultaneous CPU use.

## Optional streaming phase timing

`--profile 1` enables coordinator wall-clock intervals, reported in seconds per
batch token under `profile/`: events (queue/fiber/event grouping), update
(Aggregate/State/Read jobs and barrier), select (region decisions/comparison),
full (Next/Full/Emit jobs and barrier), commit (state/message publication), and
cleanup (tick-local destruction). Their sum equals `tick_seconds`, a subset of
`perf/advance_seconds`. Intervals include job construction and barrier time;
they are not sums of worker durations or isolated operator CPU times. Disabled
by default, this option does not change the schedule or semantic state.
The scale executable additionally reports `input_seconds` (previous logits
release, token/embedding/input preparation) and `head_seconds` (readout rows,
stack and vocabulary Linear). Input + advance + head partitions token time;
readout Attention itself belongs to the graph's update interval.
