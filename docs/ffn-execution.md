# Full FFN execution candidates

These optional paths preserve the selected instance's mathematics, parameters
and private histories. They do not narrow KV, share independent FFNs, alter
allocation, or modify the fixed `math/` snapshot. The default remains
`ffn_weight_access="gather", ffn_packing="rectangular"`.

## Local function and ownership

For a selected node i and content row x, the unchanged function is

```
z = RMSNorm(x)
y = x + W_down[i] (SiLU(W_gate[i] z) * W_up[i] z)
```

Content width is d and hidden width F. Gate/up store [owners,F,d]; down stores
[owners,d,F]. Each projection retains its existing base groups and, when shared,
its two node-private adapter matrices. No new learnable tensors are introduced.
Every use of a shared base contributes to its gradient. Parameters and optimizer
slots keep their identities; execution options serialize in ModelConfig and
checkpoints. No projected weight is cached across updates.
Ordinary FFNs reuse one discrete owner plan across gate/up/down. Placed wrappers
preserve the option and ownership but currently rebuild the group plan per
projection; they do not inherit that scheduling improvement.

## Independent choices

| Option | Execution |
| --- | --- |
| gather + rectangular | Existing owner-indexed BMM and [active_nodes,B,Q,d] Full layout |
| grouped + rectangular | Direct original weight-bank views, unchanged rectangular FFN rows |
| gather + power2 | Compact FFN rows within each node; retain indexed weight contractions |
| grouped + power2 | Combine both changes; measure their interaction independently |

For a projection with owner group g(r), group the activation rows by g, with
int64 cumulative row counts. A single weight bank supplies
`Y_g = X_g @ W_g.T`. The explicit first-order adjoints are
`dX_g = dY_g @ W_g` and `dW_g = dY_g.T @ X_g`; repeated owners contribute to
the same group. Adapter projections retain their private node groups. Static
possible-owner metadata may bound a bank slice but never establishes the actual
dynamic owner order. Weight storage is borrowed through a view, not detached.

The NPU capability probe on the site's TorchNPU 2.12/CANN 9.2 stack found FP32
group_type=0 forward/input adjoint and group_type=2 weight adjoint within its
local thresholds, including empty groups. Native autograd was unavailable.
The implementation therefore needs an explicit adjoint; that primitive result
alone does not qualify integrated training. CPU group-wise matrix operations
are an explicit reference implementation with a different dispatch cost.
The grouped candidate supports NPU FP32 and CPU FP32 (FP64 for diagnosis only),
and rejects autocast and other backends explicitly. Higher-order differentiation
is outside its declared first-order adjoint contract. Existing gather execution
and its precision/backend choices remain available.

For packing, let `m_i = count(control_i > 0)` over sample and original-time
coordinates in the current Full tile. The unchanged query rectangle has
`E_old = |A| B Q` rows. Put each positive m_i in one capacity bucket c_i from
1,2,4,..., capped at BQ. Then

```
S = sum_i m_i
E_new = sum_i c_i
S <= E_new <= E_old, and E_new < 2*S for S > 0
```

Each owner occurs in exactly one bucket. Gate/up/down reuse its packing map.
Return only selected rows to the existing publication indices; values in unused
rectangle slots are unobservable. All-zero Full still bypasses the FFN, preserving
disconnected-gradient behavior. Attention keeps its original node/sample/history
axes, original int64 positions, visibility, RoPE anchors and weighted derivatives.

L=1 decode padding is primarily inactive samples within active owners. Packing
only query lengths or splitting regions cannot remove that sample padding.
The new row bound excludes hardware tile padding and says nothing by itself
about speed. Bucketing adds dynamic selection, indexing and calls; grouped
weight access adds activation organization and an explicit backward.

## Validation and measurement boundary

Implementation status, fixed source, qualification and measured recommendations
belong in STATUS and the current evidence. Keep original thresholds and finite
failures. Validate values/VJPs, input and parameter gradients, shared sums,
empty reception, original-time eviction/chunking, continuing AdamW and an actual
new-process restored update before large inference. CPU FP64 is bounded diagnosis.

Normal timing is independent of synchronized module/operator passes. WorkAudit
reports `query_packing` for Attention and `ffn_packing` from actual FFN gate
operands; gate/up/down matrix row counts must agree. Capacity estimates retain
the same parameter/private-state floors; lower rows do not imply a lower measured
allocator peak. Record weight/optimizer/private state/saved activation storage,
allocator peaks and unclassified workspace separately.

The intrusive allocation observer separates sorted solving, safeguarded bisection
and saved-face construction. Requested fallback rows and all executed rows are
different counts. Observer metadata reads occupy their own diagnostic scope;
they are excluded from normal performance and cannot be mistaken for production
host-synchronization cost.

Representative configs are in `configs/ffn-execution/`. Use
`python -m benchmarks.execution_candidates --help` for the finite normal runner
and `python -m scripts.check_execution_candidates --help` for qualification.
The existing paired breakdown runners provide independent synchronized replays.
All long runs use the frozen-source launcher in [execution](execution-flows.md).

The [finite CPU/NPU comparison](evidence/ffn-execution-devices-20261119.md)
records the implemented options, retained formula/update failures and actual
fresh-process restores. The measured medium NPU training candidates saved memory
without improving throughput; W/P/WP were not extended to large inference.
Keep the default gather/rectangular path unless a separately qualified workload
provides a reason to select another option.
