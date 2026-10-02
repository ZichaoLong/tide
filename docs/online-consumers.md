# Continuous model consumers

`scripts/run_execution_flow.py` consumes a hash-validated v2 workload with one
command interface for independent Python execution, a Python native adapter,
and an independently linked LibTorch program. These are experiment consumers of
the public library. They do not add a scheduler, a loss requirement or a model
head to core graph semantics. CPU FP32/FP64 and directed mixed NPU FP32 complete-training qualification is
recorded at [clean fe2d886](evidence/online-consumers-20261002.md):120 CPU checks
and18 NPU cases, no skips. This is small-model correctness, not full-size throughput.

Resident training requests full Result trace/messages only with `--diagnostics`
(or a Python diagnostic observer). Its backward journals remain enabled in all
training runs. This avoids retaining export-only copies in ordinary benchmarks;
see [the training contract](resident-training.md) and STATUS for qualification.

## Workload and parameters

`prepare_execution_flow.py` now defaults to `--protocol continuous`, schema
`tide-complete-flow-workload-v2`. The old `--protocol reset-v1` still emits the
unchanged v1 reset-window contract. The continuous runner explicitly rejects v1.
Both records include the same reachable topology and exact parameter formula:

```
Add:       edges*D*D + nodes*D + edges + input_ports + 2*vocab*D
Attention: Add + 4*nodes*D*D
```

Every physical body edge has its own learned D×D output projection. Body Full is
SiLU followed by learned RMS normalization, with fixed zero projection biases;
output boundary projections are fixed identities. Read is the declared FP32 norm
profile. Add uses fixed .99 retention and learned all-source-softmax logits.
Attention uses four heads, learned QKV/output matrices and all-source-softmax
fiber pooling, fixed zero biases and .01 decay. Embedding and head are learned.
Non-learned scaffolding is shared where safe and excluded from the count.
Both implementations assert the actual learned tensor count against the packet.

`named-lcg31-v1` defines CPU FP32 initial values independently of framework RNG:
start `key=seed mod 2147483647`, fold each ASCII parameter-name byte using
`key=(131*key+byte) mod 2147483647`, then initialize element i to `x=(i+key) mod
2147483647`. Apply `x=(1103515245*x+12345) mod 2147483647` three times. Its FP32
value is `((x mod 65536)-32768)*2^-20`, then explicitly cast to payload dtype.
All integer intermediates fit signed int64. Norm and pooling coefficients start
at one. Canonical initializer names are in the matching Python/C++ builders.
No candidate runs another implementation to get numerical inputs or routes.

The wide packet remains 480 reachable body nodes, 2,208 body edges, D2048,
B512/T12/V50304: Add9,468,053,696 and Attention17,521,117,376 learned elements.
This is the reachable cross-family fixture in the execution contract, separately
identified from the historical two-cortex 8.8B/17B workload. Parameter count alone
does not establish model equivalence. `representative` offers a bounded D128
fixture; packet generation is independent of Torch and device availability.

Rank-aligned packets have equivalent Settle/TimedDAG/PDG encodings. Delayed packets
exercise TimedDAG/PDG unaligned arrivals and refuse a Settle equivalence claim.
Settle encoding shares body owners and allocates only the two boundary nodes;
it no longer temporarily constructs another full set of body parameters.
The fixtures do not constrain the underlying general online greedy scheduler.

## Execution and training boundary

The runner exposes `--schedule streaming|prefill`; prefill selects the public
general online greedy algorithm. It continues numerical state, history, KV,
pending messages and the absolute input ledger through warmup and measured
windows. At each optimizer boundary it explicitly detaches the graph, preserving
the numerical continuation. There is no per-window state reset.

Input token IDs are `(absolute_position*7+sample*3)%vocab`. The external head
predicts the next position. Cross entropy sums over present outputs and divides
by the requested token count across that update's windows, including positions
with no output. A wholly absent loss remains disconnected. SGD and AdamW include
all learned graph, embedding and head parameters; None gradients are preserved.
Gradients must be finite before any optimizer update. This short synthetic loss
checks the training mechanism, not downstream convergence or model quality.

FP32/FP64 eager training is implemented. The resident consumer also implements
FP16 payloads with FP32 loss, explicit adjoints and optimizer masters; its whole
consumer qualification is separate from the public resident FP16 library gates.
Eager FP16 consumer training still explicitly refuses its unqualified master path. The consumer is a bounded benchmark runner;
it does not claim to serialize an application bundle with head/data cursor.

CPU and mixed A/B/C use public Read/control/selection placement, with the fine
switches `--read`, `--control`, `--selection`, `--events`, `--scoring-dtype` retained.
CPU/mixed consumers currently use one payload device. `--preset resident` now
has a separate public C++/CANN consumer: FP32/FP16 single/multi-device inference and
single/multi-device complete training. Python uses `--implementation native` as
a client of that same backend, not an independent PyTorch resident scheduler.
Fixed-source qualification at [clean0d61cb9](evidence/online-resident-consumers-20261002.md)
passed59 affected CPU,19 resident and18 mixed regression checks,with no skips or
tolerance changes. A separate actual two-card Attention training trace observed
no AiCPU; this is not a full-size performance conclusion.
[HARD slot-affine reverse/publication](resident-emission-vjp.md)
is independently qualified; broadcast evidence is not used for this model.

The resident graph supplies actual packed output coordinates,values and presence.
Only present rows enter head/loss arithmetic. Output cotangents are scattered back
to the public window; its backward returns actual boundary input gradients.
External input coordinates map those gradients to embedding rows in one packed
accumulation per window,including repeated token IDs. Previous-cut pending leaves
stay detached. Head/embedding use staged FP32 SGD/AdamW proposals; all consumer
values and slots must be finite before requesting the graph's atomic optimizer
step. Consumer publication follows graph acceptance. None and connected zero are
distinct; no CPU reference result or gradient becomes a candidate input.

Output compaction and its count observation occur at the public window boundary.
Online graph readiness,selection and recursive event progression remain device
controlled. The current external-input adapter prepares host coordinates and
stacks payload rows in bulk. Neither boundary adapter is a claim of zero host
work across the entire application. Optional diagnostics explicitly materialize
CPU observables; normal training does not export graph state.

Explicit placement partitions physical projection banks and partial gradients on
Full owners; no dense coordinator replica is assembled. This is qualified on
[cleanacb84f3](evidence/resident-projection-shards-20261002.md). Complete resident
[consumer capacity admission](consumer-capacity.md) is qualified on
[clean0b1a5aa](evidence/consumer-capacity-20261002.md) at small/D128 scales.
Local reverse safe splitting is qualified below; public multi-device
inference is qualified on [clean7329c71](evidence/public-sharded-inference-20261002.md).
Actual FP16 consumers and bounded packed head rows are qualified on
[cleand178b86](evidence/resident-consumer-head-20261002.md); full-size performance
remains pending.

## Commands and records

From the repository with the intended Torch/runtime environment active:

```bash
python scripts/prepare_execution_flow.py --preset smoke --memory attention --output-dir artifacts/flow-input
PYTHONPATH=python python scripts/run_execution_flow.py --packet artifacts/flow-input/workload.json \
  --output-dir artifacts/flow-python --device cpu --dtype float32 --family settle \
  --implementation python --preset cpu --schedule prefill --training --optimizer adamw
```

The native Python client uses `--implementation native --native-library BUILD_DIR`.
Build the independent C++ program against an installed matching public core:

```bash
cmake -S tools/online_bench -B build/online -DCMAKE_PREFIX_PATH=/path/to/TideGraph/prefix
cmake --build build/online --parallel 2
python scripts/run_execution_flow.py --packet artifacts/flow-input/workload.json \
  --output-dir artifacts/flow-libtorch --device cpu --dtype float32 --family pdg \
  --implementation libtorch --native-binary build/online/tidegraph-online-bench \
  --preset cpu --schedule streaming --training
```

For the optional resident consumer,build with `-DTIDE_ONLINE_RESIDENT=ON` against
installed matching TideGraph and TideResident packages. The Python client supplies
`--resident-library BUILD_DIR` alongside its native library. Use `--preset resident`
and `--device npu:0`; inference and training can request `--devices 2` (or another declared count)
and `--owner-policy memory|locality`. The launcher leases/remaps physical devices
outside this portable command. Multi-device inference uses its own public forward
session without retained training tapes; all resolved devices are synchronized
at timing boundaries. Requested and effective ownership are recorded.

`--chunk-policy conservative|aggressive` and `--resident-...` options expose the
public queue,arrival,output,journal,stage,KV,physical-chunk and workspace capacities,
plus retained/backward/optimizer/program budgets. See `--help` for exact names.
Consumer defaults reserve a forward budget of512MiB and backward budget of2GiB;
these allow vendor workspace and nested reverse reservations at the default journal
capacity. They are limits,not a total peak-memory estimate or a guarantee of scale
admission. Exhaustion is explicit; increasing only one limit may leave another
unsatisfied. Requested and effective placement/limits are recorded.

`--device-memory-bytes` additionally controls the complete resident consumer's
per-card incremental memory budget (default0: live driver free memory). A shape
planner accounts for simultaneous lifetimes and reduces physical row maxima
without changing logical capacities. See [capacity admission](consumer-capacity.md)
for scope, headroom, estimates and the Torch-free offline command.

The resident consumer head also accepts `--head-workspace-bytes` (default4GiB).
Before model construction, a tensor planner reserves output cotangents, head
parameter-gradient accumulation/partial storage, the optional FP32 head copy,
row scratch,a32MiB operator allowance and10% aggressive or25% conservative
headroom. The operator allowance is twice the16MiB floor observed on the local
CANN matmul path; other environments still require peak calibration. It selects the largest
physical output-row chunk that fits, and rejects a budget unable to hold one row.
Only present outputs enter each packed matrix multiply. Every row retains the
complete vocabulary softmax; denominator and update boundary remain unchanged.
This is a consumer-head envelope, not total HBM admission or a vendor-workspace
guarantee. It excludes graph tapes, embedding/optimizer and caller parameter banks.

FP16 head matmul uses FP16 operands/results; log-softmax/loss, the explicit
first-order cast VJP and its accumulations use FP32. FP32 master updates publish
rounded payloads only after all graph/consumer finite checks succeed. Low-precision
whole-model comparisons use the declared FP16 tolerance and exact discrete/None
checks; FP32 retains its existing strict tolerance. `precision`, `head_memory` and
per-step `head_chunks` distinguish effective dtype, reserved bytes and actual work.

Attention reverse now treats `--resident-reverse-chunk-rows` as a physical upper
bound. Before allocating stage buffers, a shared planner fits complete owners,
queries and key tiles into the existing disjoint tensor reservations. It chooses
the largest owner batch that admits at least one query, then the largest query
batch and key tile that fit. It never retries OOM or truncates a logical fiber,
KV visibility, loss reduction or retained window. A single owner that cannot fit
still fails explicitly. This is local safe splitting, not total-memory admission.
Qualified on clean `106cbeb`: [reverse budget evidence](evidence/resident-reverse-budget-20261002.md).

Backward results expose `statistics`; consumer step records include
`reverse_event_*`/`reverse_fiber_*` group counts, requested maximum, effective
owner/query/key row minima/maxima, estimated tensor bytes and assigned budgets.
These are construction-time physical capacities across retained windows/owners,
not observed active row counts or allocator peaks. Their extraction requires no
per-event device scalar read. The public C++ gradient layout changed; rebuild
clients and bindings against the matching resident library.

The LibTorch launcher does not import Torch. It derives and hashes the exact v2
text input from the validated JSON, records the executable digest, and launches
the independent runtime. Direct C++ invocation validates structure but treats the
header SHA as a declared identity; use the shared launcher for a verified packet.

`--steps`, `--warmup`, `--windows-per-step`, `--threads` and new output directories
are explicit. `--parameter-budget` defaults to1GiB of learned payload storage,
checked before allocation; it is not a total peak-memory estimate. Raising it
alone does not certify safe full-size execution. Native and Python runs preserve
failures in `result.json`; the standalone log remains in `consumer.log`.

For eager native and standalone LibTorch consumers, `--workers` selects the
existing node worker pool independently of ATen intra-op `--threads`.
`--packed-sources` and `--batch-next` expose the existing packed source transport
and batched Next/reset policies. Defaults remain one worker with both policies
off; `host_execution` records the effective choices. These options change physical
execution only, preserving the logical batch, continuation and optimizer boundary.
The independent Python scheduler and resident device scheduler reject nondefault
host controls before model construction. Resident packing remains device-owned.
Choose bounded worker/thread counts when measuring; one-worker results do not
establish the best available CPU throughput. The wiring is qualified on clean
`2222d9d` ([CPU11/NPU10 evidence](evidence/consumer-host-execution-20261002.md));
the [bounded representative comparison](evidence/representative-host-policy-20261002.md)
selects CPU16 packed and mixed-a4 packed from its declared search. CPU wins
steady inference and resident wins complete training in that submatrix; these
choices do not establish the best policy for another workload or machine.

Construction and warmup durations are separate. A measured step includes token
preparation/upload, online graph execution, head/loss, backward, finite checks,
detach/optimizer and final synchronization. No independent reference is timed.
Actual output counts and available scheduler counters accompany each sample.
`--diagnostics` adds bounded states/routes/gradients/updates JSONL and is not a
formal timing mode. Profiler runs and three fresh-process recommendation repeats
remain separate requirements; one process's step timings do not satisfy them.
Resident fiber KV proposal duplication is removed on clean80dae6e;
[affected qualification and allocator comparison](evidence/resident-fiber-append-20261002.md)
show128MiB lower peak at the representative Attention shape, without reducing
logical KV capacity. This is a storage improvement, not full-size admission.
Eager and resident Python/native/LibTorch consumers accept `--sample-chunk-rows N`.
Zero preserves the whole logical batch; a positive value limits samples in one
physical forward/backward group. Each group keeps all requested windows connected
and carries its own state/history/pending/KV into the next step. The final group
may be smaller. Parameters and the optimizer are shared; gradients accumulate
across groups with None/connected-zero behavior preserved. One finite-gradient
check and one optimizer update follow the entire logical batch. Loss reduction
uses the original logical batch, and input/target generation uses global sample
IDs. This is an explicit physical maximum, not yet automatic memory admission.

`batch_execution` records requested/effective rows and group count. Diagnostic
window records for split runs contain `sample_range: [begin,end,logical_batch]`
and global sample IDs; each is a partial batch, so the ranges must be combined
when comparing a whole window. Graph scheduling within each sample is unchanged.
Persistent state for all samples and full model/optimizer storage remain live;
this reduces activation lifetime, not those fixed costs. All work stays inside
the complete-step timer. The eager increment is qualified on clean `e6cc52b` (CPU60/NPU38 and separate memory
observations; [evidence](evidence/consumer-sample-chunks-20261002.md)).

The resident consumer uses one live parameter/optimizer owner and opaque NPU
continuations for independent sample ranges. Each range's windows form one
connected backward group; explicit device accumulation separates ranges, then
one graph update and one embedding/head update complete the logical step. Saved
state from every range continues under the next shared parameter generation.
The fixed-size tail capacity receives only real input rows: absent samples do
not create events. Input IDs, output labels and embedding VJPs add the global
sample offset; loss retains the full logical-batch denominator.

Resident admission uses physical sample rows for the active state and tapes,
and additionally charges all saved numerical continuations plus simultaneous
old/replacement gradient accumulators. A restored handle is released before its
replacement is saved. Snapshot byte checks and allocator observations verify
those declared bounds. Live KV remains dense and all sample continuations
remain on their original NPUs. Explicit slicing is qualified on clean `75543a7`
([evidence](evidence/resident-sample-chunks-20261002.md)); automatic sample-size
selection remains pending.

`--resident-context-bytes BYTES` optionally enables compact saved continuations
and caps their combined tensor/index storage on each device. Zero keeps dense
saving and the existing dense envelope. For a positive budget, the consumer
subtracts all other live handles on each card before requesting the next save;
the library checks every card before copying payloads. Exceeding the pool fails
explicitly and does not discard messages/KV or change the logical batch. Initial
shared empty handles are conservatively charged once per sample range. Packing
metadata workspace is reserved separately in complete-memory admission. The
`context_storage` record reports policy, requested bytes per device, admitted
budgets and peak saved bytes. This is a bounded pool for an explicit sample
size; live state, retained tapes, parameters and vendor workspace still have
their own costs. This pool is qualified on clean `48e44b0`
([evidence](evidence/resident-context-pool-20261002.md)).

The sample-slicing FP16 gate separates cross-dtype rounding from slicing: its
CPU FP32 comparison uses tensor infinity-norm error bounded by
`0.002 + 0.02 * max(abs(reference))`, with exact discrete/None checks. A second,
independently executed whole-batch FP16 comparison uses the existing elementwise
`atol=0.002, rtol=0.02`. FP32 retains elementwise `atol=1e-6, rtol=1e-5` against
independent CPU execution. This avoids interpreting cancellation near zero as a
slicing defect; it does not relax event identities or gradient connectivity.

Full-size peak memory, aggressive-safe chunking and complete F6 comparisons remain
pending. The deliberately paused historical CPU job is not managed by this CLI.

Consumer results also include `memory` phase records. Each logical accelerator
reports process allocator current/peak allocated and reserved bytes before model
construction,after construction,after warmup (when present),and after measured
steps. Peaks reset at phase boundaries; these samples occur outside step timers.
Initial counter setup belongs to construction. They include caller allocations
in the same process,with the initial record as baseline,exclude untracked vendor/
driver memory,and are observations rather than admission guarantees. CPU peak RSS
is a process-lifetime high-water mark and cannot be reset per phase. Formal runs
use fresh processes; benchmark functions own/reset the selected allocator counters.
No per-event synchronization or graph-kernel instrumentation is added.
