# LH versus PDG: current Attention implementation

This describes the measured FP32/no_grad LH Attention instance and the current
PDG packed kernel, not an intrinsic restriction of PositiveDelayGraph semantics.
The measured call counts and qualification are in
[evidence/lh-pdg-operator-work.md](evidence/lh-pdg-operator-work.md).

## Shared operation and the meaning of a call

A node may receive multiple source vectors for one sample at one body tick.
Let q_b be that number of current source/query rows, c_b the old KV row count,
L_b=c_b+q_b, H the head count and d=D/H. Each query attends to the same sample's
old cache and all current-fiber keys. Source rows at the same tick are not
sequential token positions; no triangular mask is imposed inside that fiber.
Neither implementation allows cross-sample scores.

Both first concatenate the node's source rows and execute one D→3D QKV Linear.
Both pool per-source attention results back to one vector per node/sample/event,
then execute a batched D→D output projection. The attention call counter counts
one group executing QK, softmax and AV, not QKV calls, node invocations, heads,
or a hardware-level kernel-launch count.

## Original LH CROSSBATCH

`AccumulateLocal.cpp::Attention::forward_via_multi_batch` constructs
`SI=sampleids.repeat_interleave(I)`, associating each query row with its sample.
`BatchHidden.cpp::BatchPtrKVHidden::attention` chooses the maximum KV length
among participating samples. The custom matmul Functions gather the cache with
`cache.slice(2,0,Lmax).index({SI})`.

For S=sum(q_b), its shapes are:

- Q: `[S,H,d,1]`; gathered K: `[S,H,Lmax,d]`;
- QK scores: `[S,H,Lmax,1]`, with padded keys masked by −infinity bias;
- AV: `[S,H,1,Lmax] @ [S,H,Lmax,d]` → `[S,H,1,d]`.

All queries share one call per node update, even with different q_b/c_b.
If a sample has multiple queries, its K/V rows appear repeatedly in the gathered
cache. This trades padding and gather/storage work for fewer larger operations.
The persistent cache is a node-level `[batch,H,capacity,d]` tensor, initially
capacity16 in this profile, extended geometrically and updated/cleared by indexed
writes. The original custom Functions also support the original autograd path;
this comparison concerns their no_grad forward execution only.

## Default PDG exact packing

[fiber_packing.cpp](../cpp/src/fiber_packing.cpp) groups samples by the exact pair
`(initial_cache_rows, total_query_rows)`. For one streaming tick this is `(c_b,q_b)`.
Each bucket of G samples builds:

- Q: `[G,H,q,d]`; K/V: `[G,H,c+q,d]`;
- QK: `[G,H,q,c+q]`;
- AV: `[G,H,q,c+q] @ [G,H,c+q,d]` → `[G,H,q,d]`.

K/V is represented once per sample inside the bucket and can be reused across
its q queries. Samples with different shapes are separate calls. The current
implementation concatenates old/current KV, stacks bucket tensors, builds decay
bias/masks, and clones the final per-sample slices into compact persistent state.
Sparse `(sample,node)` state is created only when used. This differs from LH's
node-wide reserved cache and indexed updates.

The same PDG kernel also accepts multiple-time prefill blocks. There q is the
total query count across the block, and each event gets its own visibility/decay
mask. Consequently even exact-shape buckets can compute masked future scores
in prefill. The measured streaming case has one event per owner per call, so
its executed and valid score counts agree.

## Small shape example

| Sample | Old KV c | Current queries q | Total KV L |
| --- | ---: | ---: | ---: |
| A | 4 | 2 | 6 |
| B | 9 | 1 | 10 |
| C | 4 | 2 | 6 |

LH uses S=5, Lmax=10: one attention call and50 score elements per head.
PDG uses bucket(A,C)=(4,2) and bucket(B)=(9,1): two calls and
`2×2×6 + 1×1×10 = 34` score elements per head. Useful attention work agrees;
padding and operation shape differ. QKV remains a single projection over5 rows
in both implementations. Padding more coarsely is a permissible optimization
if outputs, masks, state ownership and VJPs continue to satisfy the contract.

## Pooling and interpretation of the observed counts

For the measured all-softmax Confluence, LH computes slot weights once for the
node batch, selects source coefficients, and uses the CSR `SumCoe` mapping to
reduce all source outputs to sample rows. Current PDG
[fiber_pool.cpp](../cpp/src/fiber_pool.cpp) calls `fiber_pool_rows` separately for
each event: it constructs a slot-index tensor, computes/selects coefficients,
and reduces that event's source rows. The output Linear is then batched again.
These pooling/index/stack/cat operations are not included in the attention-call
counter or the four dense projection FLOP totals.

Measured means per batch token: LH921.875 attention calls; PDG5769.25 (6.258×).
QKV calls were921.875 versus921.625. LH score padding was1.808×; PDG1.0×.
QK/AV represented only0.132% versus0.0728% of the counted matrix FLOPs.
Thus smaller score tensors do not guarantee lower elapsed time. Extra operator
calls and intermediate tensors are plausible costs to investigate, but the
6.258× count alone neither proves a bottleneck nor explains the entire15.2%
latency gap. A separate controlled operator/profile experiment is still needed.

The portable [CPU comparison kit](../tools/cpu_compare/README.md) exposes both
unchanged strategies and the same counters for target-machine measurements.

The optional [single-batch policy](attention-packing-policy.md) now implements
padded query-owner gathering in PDG. The measurements above predate that option
and describe the default exact policy.
