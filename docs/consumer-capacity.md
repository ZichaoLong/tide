# Complete resident consumer capacity admission

The declared Add/Attention consumers in `tools/online_bench` now plan incremental
memory for every logical NPU before constructing model tensors or sessions.
`--device-memory-bytes 0` uses the current driver free memory on each card. A
positive value caps each card at the smaller of that value and its driver free
memory. This is an incremental run budget, not total physical HBM or an allocator
quota. Existing allocations are excluded from the budget and recorded separately.
The local small/D128 qualification is recorded in
[consumer-capacity evidence](evidence/consumer-capacity-20261002.md).

The planner uses topology and declared module shapes, never a CPU execution,
observed reference events or a numerical prepass. Its static placement is the
same stable memory/locality heuristic used by the resident backend initially.
The bounded aggressive fallback below can refine automatic physical owners. Physical
projection/state owners and canonical optimizer owners are accounted separately;
Python's explicit owner maps remain supported. Core/resident ABI is unchanged.

The per-card record includes forward parameter banks, state/KV and proposals,
routing/owner packets, journals, retained projections and window banks,
physical/canonical gradients, FP32 graph masters/slots, bounded communication,
embedding/head/loss, consumer optimizer proposals, construction transients and
program/vendor allowances. Construction, forward/loss, backward and optimizer
lifetimes have separate envelopes; their maximum is the estimate. Local
`workspace_bytes`, `retained_bytes`, `backward_bytes` and `optimizer_bytes` are
still independent capability ceilings, **not amounts summed as actual HBM**.
Exceeding those ceilings still fails explicitly.

The training envelope follows qualified storage lifetimes. Attention QKV/output
matrices, parameter biases, decay and pooling weights are shared across each
backward group's windows. With aggressive sharded execution, the declared
consumer has one complete ordered fiber head group per owner and borrows its
private frozen projection/Attention banks. These banks are already charged in
forward `parameters`; `retained` no longer charges a second copy. Conservative
and legacy single-device consumers still charge the independent snapshots.
This requires the matching qualified private-bank backend; it does not assume
that arbitrary mixed-head/subset library tapes can borrow their gathers.

`retained_parameter_copies` reports actual independent-copy allowance;
`borrowed_parameter_banks` reports the footprint already included in forward
parameters. `retained_attention_parameters` remains the included logical shared
Attention footprint, whether copied or borrowed. None is an extra sum. State,
dynamic KV/log-bias, lengths, journals and all Full/state/message snapshots retain
their previous charges. API admission limits and safety margins are unchanged.
[Private projection](evidence/resident-projection-borrow-20261003.md) and
[Attention](evidence/resident-attention-borrow-20261003.md) allocator measurements
establish the backend storage change; [separate accounting qualification](evidence/consumer-private-bank-capacity-20261003.md)
keeps actual allocator peaks unchanged and validates the revised envelope.

With aggressive multi-device training, ordered per-window canonical reduction
allows the physical projection and event/fiber Attention parameter-adjoint banks
to be reused within a backward group. The planner charges their numerical storage
once. Conservative training and the consumer's legacy single-device path still
charge each window. State, cache/message bridges, other parameter adjoints and
reverse workspace keep their existing per-window bounds; no logical capacity or
margin is reduced. `projection_parameter_gradients` and
`attention_parameter_gradients` are included components of
`physical_and_canonical_gradients`, not additional allocations. This accounting
requires the matching backend's ordered reduction and Attention-adjoint reuse.

For physical sample splitting, the first accumulation copies the completed
canonical gradients into a private FP32 bank; subsequent accumulations reuse
that private numeric bank with separate connection flags. The consumers release
each public gradient export before the next sample group. Accordingly,
`gradient_accumulation_live` charges one extra FP32 bank plus the existing
16MiB flag/metadata allowance. The current canonical output is separately
charged in the backward/optimizer envelope. `gradient_accumulation` remains
the unchanged, conservative two-input **API admission limit**, passed to
`accumulate(max_bytes=...)`; it is not summed again as live memory. This does
not authorize reuse of public exports or alter the runtime's budget checks.

These corrections depend on the shared-snapshot and private-accumulation
implementations in the matching resident build. They do not change actual
allocations, dtype, gradient order, logical capacities or safety margins.
Newly admitted plans still require observed allocator calibration; a smaller
estimate by itself is neither a memory saving nor a full-size acceptance result.

Fiber KV append proposals reuse unused cache tails. The Python/C++ envelopes
subtract exactly the removed two payload buffers on each state owner:
`2 × payload_bytes × (batch × local_attention_nodes × kv_rows + 1) × width`.
The separate bias proposal, retained snapshots, gradients and all other
conservative allowances remain. This layout change does not shrink KV capacity.

Training can disable optional Result trace/message exports while keeping its
required VJP journals. The current admission envelope still reserves the former
diagnostic buffers conservatively; disabling exports does not reduce declared
queue/journal capacity or relax the estimate. Allocator observations record the
actual saving separately ([qualified comparison](evidence/resident-training-records-20261002.md)). Both clients continue charging training journals even
when the public diagnostics flag is false.

Conservative mode leaves25% plus128MiB of the incremental budget unused;
aggressive mode leaves10% plus128MiB. The shape formulas also reserve512MiB per
device for backend allocations and the declared caps of simultaneously retained
CANN program arenas. These are conservative estimates for the declared consumer
modules, not proofs about arbitrary modules, every vendor version or external
processes. The coefficients must be checked against allocator peaks on a new
stack. A post-run peak above the estimate fails calibration and preserves the
failed run; shared-resource changes can still cause a real allocation failure.

If a requested physical configuration does not fit, aggressive mode considers
halving each Full/emission/aggregation/attention-query/key/reverse/head maximum
separately. It chooses the greatest reduction in the summed positive peak excess
over all cards, breaking ties in that field order. This keeps nonlimiting batches
larger. If equal peak phases hide every individual improvement, it halves all
fields together; conservative mode always uses that joint halving. Every accepted
plan still fits the original per-card envelope and safety margin. This heuristic
does not promise optimal throughput or the fewest reductions.

If all operator maxima have reached one and automatic placement still fails,
aggressive multi-device admission may jointly move a node's Full/state ownership.
It preserves nonempty caller-supplied Python owner maps and conservative placement.
Empty owner tuples/lists select automatic placement, as does an omitted map. The
fallback uses the complete per-card envelope at one-row maxima, including
coordinator costs. Each move strictly improves the tuple of summed positive
excess,maximum excess and descending card peaks. Shape-equivalent donor nodes
share a cost trial; physical-edge locality,then node/device index,resolve ties.
No owner is emptied. Search stops on a fit,no improving move,2×node-count moves
or4096 envelope trials. It then retries requested operator maxima on the new map;
all original budgets/margins and runtime allocator checks still apply. Records
include `owner_moves` and `owner_evaluations`; this heuristic need not find every
feasible partition. The new fallback is implemented pending qualification.

The finite search performs no device workload and does not search by OOM. It
records requested/effective maxima, reduction iterations and `row_selection`
(`greedy_peak_excess` or `joint_halving`). The backend may further reduce rows to
meet local budgets. Logical batch, queue/output/journal capacity, retained window
count, KV capacity, visibility, loss denominator and update boundary do not
change. A one-row refusal reports the offending logical card and estimated versus
usable bytes. It does not establish that every possible implementation is
physically impossible; conservative estimates may require further calibration.

An offline plan needs neither Torch nor model allocations:

```bash
python scripts/plan_execution_flow.py --packet /path/workload.json \
  --devices 8 --dtype float32 --training --optimizer adamw \
  --chunk-policy aggressive --device-memory-bytes 64424509440 \
  --output /new/path/capacity.json
```

Use the same `--resident-*`, `--head-workspace-bytes`, owner/chunk policy and
window options as the actual consumer. `--sample-chunk-rows` plans the physical
sample extent while charging saved state for the full logical batch. Optional
`--resident-context-bytes` caps simultaneous saved tensor/index storage per card
and separately charges packing workspace; zero retains dense snapshots. Runtime
admission also enforces the remaining pool budget on each snapshot. This does
not shrink live KV or retained tapes. The offline result is explicitly
`planned` or `refused`, not a device run or verified throughput. Actual clients
repeat admission using live driver availability and record `memory_admission`
beside phase allocator observations. CPU/mixed consumers retain their existing
memory observations; this resident planner does not certify their peak memory.
Full-size F6 and qualification on other CANN/CUDA environments remain separate.

The `attention_parameter_gradients` component reports the QKV/output-matrix part
of `physical_and_canonical_gradients` (it is included there, not an extra sum).
Only the Attention consumer owns those matrices. Add retains its edge projection,
scalar Aggregate, vector LH/state/Read and canonical gradient charges without
fabricating attention parameters. Model-inventory tests check this distinction;
the same per-device safety margins and allocator acceptance still apply.

Optional `--auto-sample-chunks` applies the same static admission before model
allocation to progressively smaller physical sample groups. It starts at the
`--sample-chunk-rows` ceiling (zero starts at the logical batch), tries operator
row reductions first, then halves samples with upward rounding after a memory
refusal. It recharges saved continuations and gradient accumulation for the new
group count on every attempt. The search stops at the first fit or an explicit
one-sample refusal; invalid geometry and overflow errors are not retried.
`memory_admission.sample_admission` records attempted/selected rows and groups.
Fixed sample selection remains the default. This is a finite conservative
heuristic, not an optimal-throughput search or a numerical prepass. It does not
shrink queue/journal/KV capacities, change dtype, reduce the logical batch, or
change window connections, loss normalization or the shared update boundary.
Memory admission does not certify that a particular queue or per-window KV
journal can represent all future work. A larger admitted sample group or a changed
owner map can expose a previously unfilled journal limit. Failed owners remain
failed; completed-program error messages report committed queue/journal counts
and per-shard KV journal capacities, without adding successful-path reads.
Explicit capacity failures still apply to actual future inputs. CPU/eager mixed
flows reject this resident-only option; their explicit sample slicing remains.

Post-run allocator underestimation preserves its complete failed diagnostic record
([qualified reporting](evidence/consumer-failure-records-20261002.md)). The failed
status remains authoritative even when completed update measurements are present.
