# Continuous model consumers

`scripts/run_execution_flow.py` consumes a hash-validated v2 workload with one
command interface for independent Python execution, a Python native adapter,
and an independently linked LibTorch program. These are experiment consumers of
the public library. They do not add a scheduler, a loss requirement or a model
head to core graph semantics. CPU FP32/FP64 and directed mixed NPU FP32 complete-training qualification is
recorded at [clean fe2d886](evidence/online-consumers-20261002.md):120 CPU checks
and18 NPU cases, no skips. This is small-model correctness, not full-size throughput.
The current integrated CPU/NPU qualification and full-size measurement coverage
are listed in [STATUS](STATUS.md). The [current NPU consumer gate](evidence/integrated-npu-consumers-20261004.md)
adds all declared families/schedules,FP32/FP16,standalone/Python-owned native
resident training and fresh-process repartition. Keep these integration checks
separate from full-size timing and target-version qualification.

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

The C++ consumer composes the three integer affine steps modulo `2147483647`
and fills the final CPU FP32 tensor directly, under the existing ATen thread
budget. It avoids full-sized int64 temporary tensors while preserving the
`named-lcg31-v1` values bit for bit. Python retains the independent three-pass
definition. Construction is still reported separately from complete-step timing;
this change does not accelerate scheduling or model kernels by itself.
[Fixed-source verification and bounded CPU timings](evidence/exact-initializer-20261002.md)
record exact byte comparisons and the scope of the measured improvement.

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
Eager FP16 consumer training implements FP32 optimizer masters and slots, FP32
loss, and ordinary payload-dtype autograd accumulation. Each canonical leaf has
one master, including embedding/head; physical chunks share masters and one
update boundary. `--loss-scale` is a positive finite static scale, default1;
nonunit scales require eager FP16 training and are recorded with the precision
policy. No automatic scaling, skipped updates or implicit retries occur.
Nonfinite gradients fail before updating; cast overflow fails and requires a new
run or an explicitly restored application checkpoint. This eager half-backward
policy is distinct from the resident FP32-adjoint policy. Clean7b1fae5 passed
CPU61 plus55 FP32/FP64 regression checks,NPU49 checks,eleven memory calibrations
and a separate actual consumer trace
([qualification](evidence/eager-fp16-consumers-20261003.md)).
The consumer is a bounded benchmark runner;
it does not claim to serialize an application bundle with head/data cursor.

CPU and mixed A/B/C use public Read/control/selection placement, with the fine
switches `--read`, `--control`, `--selection`, `--events`, `--scoring-dtype` retained.
Eager multi-device consumer integration is implemented below; its qualification
is separate from the previously verified single-device consumers. `--preset resident` now
has a separate public C++/CANN consumer: FP32/FP16 single/multi-device inference and
single/multi-device complete training. Python uses `--implementation native` as
a client of that same backend, not an independent PyTorch resident scheduler.
Fixed-source qualification at [clean0d61cb9](evidence/online-resident-consumers-20261002.md)
passed59 affected CPU,19 resident and18 mixed regression checks,with no skips or
tolerance changes. A separate actual two-card Attention training trace observed
no AiCPU; this is not a full-size performance conclusion.
[HARD slot-affine reverse/publication](resident-emission-vjp.md)
is independently qualified; broadcast evidence is not used for this model.

### Eager consumer payload placement

Python, native and standalone eager consumers accept `--devices N`,
`--owner-policy memory|locality` and optional `--owner-map 0,1,...` through the
same CLI. Devices are consecutive logical indices starting at `--device`.
The explicit map covers all encoded nodes, including both identity boundaries;
node zero, boundaries, embedding and head stay on owner zero. Every requested
device must be used, and CPU permits only one device. Unavailable or conflicting
owners fail before model construction.

Automatic placement counts each node's actual learned matrices, normalization
and source coefficients, plus embedding/head on owner zero. It places heavier
nodes first. The locality policy minimizes the projected maximum load, then
prefers adjacent owners and lower load; the memory policy selects the least loaded owner.
Only static topology and parameter sizes enter this planning. It runs no graph
and makes no claim of optimal placement or total peak-memory admission.
`payload_placement` records the chosen map, policy and per-card learned elements.
The latter sum to the packet's exact parameter count.

Builders initialize each learned tensor independently by its canonical name on
its destination. Immutable scaffold constants are cached once per device and
shape/kind; their physical aliases may differ from the single-device cache.
They are excluded from the optimizer. Shared scalar port constants keep a canonical
value and use the library's differentiable device-copy boundary. This does not
silently repartition an arbitrary caller-owned model with tied trainable leaves.

All devices participate in timing-boundary synchronization and allocator
observations. Finite-gradient checks reduce locally, then agree on owner zero
before any update. Output rows gather to the declared head owner. Physical sample
splitting retains all connected windows per slice and applies one update after
all slices. Packed eager execution now groups completed remote messages by
source/destination, dtype and shape with bounded 8 MiB tensor packs and isolated
output VJPs; see [transport](execution-placement.md#packed-eager-transport).
Packed transport and FP32 total-memory admission have separate qualifications
([transport](evidence/eager-packed-transfer-20261003.md),
[capacity](evidence/eager-consumer-capacity-20261003.md)). Eager FP16
training uses the explicit master policy above. STATUS separates development
checks from clean immutable qualification and formal throughput.

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

The [current-core integrated qualification](evidence/integrated-npu-consumers-20261004.md)
passed49 eager and93 resident checks with a newly built combined standalone
consumer and separate Python-owned backend. Build both packages against the same
core source and intended runtime owner; successful linking alone does not qualify
an older resident library combined with a newer core. Reusing a verified static
archive also requires valid CMake install metadata. Fresh `scripts/build.py`
directories provide that metadata; an incremental artifact directory may not.

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
Aggressive planning halves the single operator maximum that most reduces the
sum of per-card peak excess, keeping nonlimiting batches larger. Equal-peak
plateaus use joint halving; conservative planning always retains joint halving.
This deterministic shape-only heuristic does not execute the model or promise
optimal throughput. `memory_admission.row_selection` records the selected rule;
the memory envelope, safety margin and logical queue/KV capacities are unchanged.

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
Post-run allocator underestimation remains a nonzero failure. Its result retains
all measured devices, phase peaks, admission estimates, timings, losses, counters
and final cut, with `failure_phase=post_run_memory_calibration` and
`allocator_within_estimate=false`. This check happens after execution: updates
may already have completed, and the record does not imply rollback or a certified
benchmark. The LibTorch wrapper carries forward a failed child record only when
its workload identity matches; malformed or mismatched records remain failures.

`--phase-timing` optionally separates synchronized wall time within each complete
step. It works through the shared launcher and direct C++/Python consumers,
including multi-device resident training. The default is off, with empty phase
arrays and no added synchronization. When enabled, `phase_timing.measured` and
`phase_timing.warmup` align with `seconds` and `warmup_seconds`:

- `sample_work_seconds` covers all physical sample chunks and connected windows:
  input preparation/upload, forward/loss, backward, accumulation and continuation.
  Existing per-chunk checks and eager gradient zeroing remain in this phase.
- `optimizer_seconds` starts after **all resolved devices** have synchronized,
  before final finite checks, the single logical update and parameter publication;
  it ends at the existing final synchronization. Inference records its entire
  elapsed time as sample work and zero optimizer time, without an extra boundary.

The two values sum to the existing complete-step elapsed time. The extra training
synchronization is included in sample work; this is an instrumented diagnostic,
not an uninstrumented throughput result. Diagnostic observables remain timed
where they were already collected. Neither phase is pure kernel time. The split
does not move any numerical work, update boundary or scheduling decision.
Do not multiply once-per-update costs by sample count when making an estimate,
or treat a phase-based estimate as measured larger-batch throughput. Preserve
the measured pilot, extrapolation assumptions, original failures and safety caps;
larger models/batches still require actual execution and memory calibration.
Qualified on clean `26176de`: [CPU13/NPU8 timing and semantic checks](evidence/consumer-phase-timing-20261003.md).

For eager native and standalone LibTorch consumers, `--workers` selects the
existing node worker pool independently of ATen intra-op `--threads`.

`--threads` controls ATen intra-op execution. Vendor BLAS libraries may retain a
separate startup thread setting; record both for CPU comparisons. In the tested
aarch64 OpenBLAS build, ATen16 left BLAS1 unchanged; an explicit OpenMP/BLAS16
startup improved a bounded graph diagnostic ([evidence](evidence/cpu-blas-policy-20261004.md)).
Set the declared environment before starting the consumer and verify the actual
library/thread configuration on each target. Do not reinterpret earlier
ATen16/BLAS1 measurements as measurements of the new policy.

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
Resident step statistics also record `window_events_max`, `window_stages_max`
and `window_outputs_max` across physical sample groups and windows, plus the
largest observed pending-queue high-water mark `pending_peak`. Pending history
travels with restored continuations, so this mark can include prior windows.
These int64 device counters are cloned at existing window boundaries, reduced
on device and downloaded together after step timing. They need no numerical
Result export or per-event host read. Use them to diagnose declared queue/trace
capacities; an observed peak does not prove a bound for future inputs/updates.
Counts cover the execution graph, including Settle's identity boundary nodes
that its projected body diagnostics omit.
`--diagnostics` adds bounded states/routes/gradients/updates JSONL and is not a
formal timing mode. Profiler runs and three fresh-process recommendation repeats
remain separate requirements; one process's step timings do not satisfy them.
Resident fiber KV proposal duplication is removed on clean80dae6e;
[affected qualification and allocator comparison](evidence/resident-fiber-append-20261002.md)
show128MiB lower peak at the representative Attention shape, without reducing
logical KV capacity. This is a storage improvement, not full-size admission.
Eager and resident Python/native/LibTorch consumers accept `--sample-chunk-rows N`.
By default zero preserves the whole logical batch; a positive value limits samples in one
physical forward/backward group. Each group keeps all requested windows connected
and carries its own state/history/pending/KV into the next step. The final group
may be smaller. Parameters and the optimizer are shared; gradients accumulate
across groups with None/connected-zero behavior preserved. One finite-gradient
check and one optimizer update follow the entire logical batch. Loss reduction
uses the original logical batch, and input/target generation uses global sample
IDs. This is an explicit physical maximum. Optional
[eager admission](eager-consumer-capacity.md) now applies a full-run shape envelope
before allocation and can automatically reduce this maximum. Its static traffic
bounds do not drive actual routes or narrow the generic scheduler.

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
([evidence](evidence/resident-sample-chunks-20261002.md)). Optional
`--auto-sample-chunks` uses [static memory admission](consumer-capacity.md) to
halve physical sample rows only after a memory refusal, before model allocation.
It records every attempted size, keeps the full logical batch and recharges
all saved state and gradient accumulators. It does not search by running the
graph or OOM, and fixed sample selection remains the default.

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

A known original-width FP32 limitation is a strict CPU/resident route mismatch
when rounded scores approach a tie. The [retained witness](evidence/original-add-route-witness-20261004.md)
shows CPU scores separated by one FP32 ULP while the resident scores tie;
subsequent event counts differ. Exact discrete checks,stable tie rules and
fixtures remain unchanged. The user accepts this separately listed numerical
limitation without blocking independent execution or performance measurement;
a failing pair is never labelled strictly equivalent. Report each run's actual
work counts. This witness does not explain every future discrepancy.

Original-width/full-size memory observations and aggressive-safe chunking are
qualified for the finite cases in [STATUS](STATUS.md);the complete F6 comparison
matrix remains pending. The deliberately paused historical CPU job is not
managed by this CLI.

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
