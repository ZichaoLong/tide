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
same stable memory/locality heuristic used by the resident backend. Physical
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

Fiber KV append proposals reuse unused cache tails. The Python/C++ envelopes
subtract exactly the removed two payload buffers on each state owner:
`2 × payload_bytes × (batch × local_attention_nodes × kv_rows + 1) × width`.
The separate bias proposal, retained snapshots, gradients and all other
conservative allowances remain. This layout change does not shrink KV capacity.

Conservative mode leaves25% plus128MiB of the incremental budget unused;
aggressive mode leaves10% plus128MiB. The shape formulas also reserve512MiB per
device for backend allocations and the declared caps of simultaneously retained
CANN program arenas. These are conservative estimates for the declared consumer
modules, not proofs about arbitrary modules, every vendor version or external
processes. The coefficients must be checked against allocator peaks on a new
stack. A post-run peak above the estimate fails calibration and preserves the
failed run; shared-resource changes can still cause a real allocation failure.

If a requested physical configuration does not fit, the planner halves the
Full/emission/aggregation/attention-query/key/reverse/head row maxima until it
finds a fit or reaches one row. This bounded search performs no device workload
and deliberately does not search by OOM. It records requested/effective maxima
and the number of reductions. The backend may further reduce physical rows to
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
window options as the actual consumer. The offline result is explicitly
`planned` or `refused`, not a device run or verified throughput. Actual clients
repeat admission using live driver availability and record `memory_admission`
beside phase allocator observations. CPU/mixed consumers retain their existing
memory observations; this resident planner does not certify their peak memory.
Full-size F6 and qualification on other CANN/CUDA environments remain separate.
