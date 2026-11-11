# Hidden KV query blocks

Fixed definition: `0770593`, KV §2–4 and architecture Read/Full. This is an
execution representation of the same weighted functions and declared first-order
VJP. Graph v13 and checkpoint v5 remain owned by [semantics](semantics.md).

## Boundary and dispatch

Default host `auto` can use `kv-block-v1` before the existing `packed-v2` query
implementation. It receives actual prepared writes, a raw left state, original
int64 times, owner/prefix ranges and query payload/control columns. A matrix
query contracts owner, head, query time and history axes together. It does not
read a reference trajectory or reconstruct state by replaying another schedule.

The caller selects this representation only for untraced prepared KV sequences.
Only hidden positive prefixes before the final positive write carry provenance.
The final state, public left state and explicitly traced states retain the
ordinary independent-root interface. No persistent state or checkpoint slot is
added. Provenance is an internal validated handle; moving/casting/rebinding its
actual slot tensors invalidates it. Identity-preserving placement can retain it.

The initial capability is position-free, unexpired KV, strict support, FP32/FP64,
with at least two eligible queries and a 32 MiB attention-workspace estimate.
It supports sum/qry, finite Describe heads, Read+/Read- (Read0 uses the ordinary
bound initial state), arbitrary original int64 times and ragged independent owners.
It is selected by actual capabilities, never by FC/HB workload name. Short-weight
underflow, rotary/distance terms, expired prefixes, heterogeneous requires-grad
flags and incompatible placement retain the old legal path. The finite workspace
limit is a per-call dispatch boundary, not a cumulative logical-time limit.
Explicit `packed-v2` remains available as the pre-block execution comparator.

## Forward and adjoints

The bank joins each owner's raw left slots and actual prepared writes once.
Visibility is `j < prefix_end && timestamp[j] <= query_time`; the expiry check is
an integer eligibility condition. Masses remain trainable float fields. Weighted
Attention uses `softmax(QKᵀ/sqrt(d) + log(weight)) V` over positive visible entries,
then the specified `min(total,1)` attenuation. Sum and Full keep their own
projection/residual/FFN formulas. No boolean mask substitutes for trainable mass.

Private autograd leaves represent whole write/query columns. Original row tensors
remain outer inputs, including aliases. Return fields remain independently
rootable read-only views. Backward groups *defined output field masks*, accumulates
bank column adjoints and returns them only to the participating owner and causal
write prefix. A future write receives None; an actually connected zero derivative
remains zero. Original input/state/parameter versions are guarded. Higher-order
backward remains explicitly rejected.

The Attention transpose uses QK/PV matrix products rather than expanding
`[B,H,Q,K,D]`. Centering V against a bank's common first positive entry preserves
the stable first-order expression while reducing to bank-sized K/V adjoints.
The eligibility conditions retain that common anchor. Padded/unused bank entries
are masked before projections, and an empty padded query has a zero result.
Inference computes directly under no-grad; it does not construct a private graph.

This path bypasses hidden snapshot adjoints, not terminal-state gradients.
Terminal queries still depend on the returned terminal root; earlier outputs do
not acquire a gradient with respect to a future terminal state. Raw state and
pending are decoded from the candidate's actual execution as before. Changes in
floating reduction order remain subject to existing strict checks, with bounded
numerical diagnosis recorded separately from causality or restore errors.

## Implementation and maintenance

| Responsibility | Python | C++ |
| --- | --- | --- |
| Provenance and eligibility | `weighted/kv_block_state.py`, `kv_block_query.plan` | `weighted_kv_block_state.cpp`, `weighted_kv_block_plan.cpp` |
| Numerical column formulas | `weighted/kv_block_math.py`, `block_attention.py` | `weighted_kv_block_math.cpp`, `weighted_block_attention.cpp` |
| Independent root boundary | `weighted/kv_block_query.py` | `weighted_kv_block_vjp.cpp` |
| Default/fallback integration | `packing.py`, `weighted/query_execution.py`, query/read/full batches | `block_prepare.cpp`, `weighted_programs.cpp`, `weighted_kv_block_dispatch.cpp` |

Native `StateExecutionView` is generic ephemeral provenance, never serialized.
`StateKernel::hidden_sequence_views` and `ReadKernel::uses_hidden_sequence_views`
keep generic scheduling independent of weighted KV details. Python State keeps
private execution metadata outside its mathematical dataclass fields.

`query_block-v1_nodes` reports configured capability; actual
`packing/projection/kv_block_read|full/{calls,rows}` reports dispatch. Existing
packed-v2 node counts include its use as the fallback capability. The execution
policy identity includes `kv-block-v1+packed-v2+describe-batch-v2`, including the
standalone checkpoint manifest. Installed consumers must explicitly migrate;
frozen old source/evidence identities are not relabelled.

Directed checks: `test_block_attention`, `test_kv_block_query`,
`test_bank_attention`, `test_kv_sequence_vjp`, `test_kv_prefix_roots`, query field
roots, selected-boundary training/restore and the SDK query gate. Literal formulas
validate local numerical maps; independent scheduling validates dependencies;
actual public-state roots, optimizer and exact-next-window tests validate their
composition. Performance results require separate fixed-source evidence.

When changing this boundary, maintain this contract, architecture/semantic links,
execution identity and actual-dispatch counters. Use the [FC diagnosis entry](fc16-cost-attribution.md)
and the [performance skill](../.agents/skills/optimize-tide-execution/SKILL.md)
for reusable three-layer comparisons. Current qualification and limitations belong
in STATUS/ROADMAP, not inferred from this implementation description.
