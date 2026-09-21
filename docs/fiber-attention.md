# LH same-fiber attention: sum and tick-repeat bias

Profile `lh-fiber-attention-sum-repeat-v1` is a separate local program from event
GQA/window attention. Python `fiber_attention.py` is the readable query/head
oracle; native `fiber_attention.cpp` uses head-batched matmul for scalar steps.
`fiber-packing.md` describes the independent-batch/event-sequence implementation.
`fiber-pooling.md` adds mean, linear, active-softmax and all-softmax pooling under
separate profile names, keeping the sum profile and its reduction order unchanged.
The first qualified gate (`evidence/lh-attention.md`) only covered scalar state
loops; later evidence must explicitly qualify packed work. No speed claim follows.

## Forward and stored state

Require equal query/KV heads, width divisible by heads, window=0 and sum Aggregate.
Take each complete `Content.sources` row `atom.value * scale`, ordered by local
input slot. Physical scales act before Q/K/V. Do not feed the summary as a token.
Let `D` be width and `d=D/heads`:

- `fiber_qkv` is `[D,3D]`; `fiber_qkv_bias` is `[3D]`.
- Split projected rows into Q/K/V `[rows,heads,d]`; scale Q by `1/sqrt(d)`.
- Subtract scalar `fiber_decay` from every previous log bias **once per elapsed
  logical tick**, in order. Append all new K/V with zero log bias.
- Every current query sees all old keys and **all keys in its own fiber**.
  Softmax scores are `Q K^T + log_bias`. There is no within-fiber triangular mask.
- Concatenate query heads, sum query rows, then apply `fiber_out [D,D]` and
  `fiber_out_bias [D]`. The output bias is applied once after pooling.

Initial last_time=-1, observations=0, value=zero and empty cache. An event at tick
0 first decays an existing initial cache once. A candidate increments observations
once, while adding as many cache rows as real sources. Initial caches and final
caches may have more rows than observations. Numerical zero rows are present;
absent sources are absent. The original inspected `block_size` does not evict;
nonzero Tide windows and GQA are rejected for this profile.

Stored state contains the latest observation's output, last_time, observations,
and three slots: `key`, `value` `[K,heads,d]`, `log_bias [K]`. Unobserved nodes keep
the stored representation. `decode_bias(w,state,cut)` / native `decode_fiber_bias`
subtract through tick cut-1 without mutation or new events. The native decoder
infers heads from the supplied cache; a runtime/checkpoint additionally validates
against the graph's head policy. Current graph/checkpoint schemas: `semantics.md`.

## Training, clearing and continuation

Tide uses ordinary differentiable attention and a learned finite scalar decay
(default .01 in payload dtype). No original LH custom backward is adopted.
All parameters, source scales/values and initial cache slots participate in VJP
checks. Clear returns connected-zero state.value and empty cloned cache slices;
Full still sees the pre-clear comparison. Empty cache structure skips subtraction
and has no new decay-parameter path. A numerical zero bias is still subtracted.
The sum-pooled first event with no prior cache has no decay-parameter gradient.

In-memory cuts preserve the graph; explicit detach and serialized checkpoints
form gradient boundaries. Fixed parameters are required for the interpretation
as an original eager LH tick trajectory. After an optimizer update, deferred
decay uses the current parameter, as declared for lazy Add; no parameter epoch
history is reconstructed. Old-mode Read reads the stored value.

Clear/selection/Next restrictions remain scheduler capability gates for packed
sequence work. Training replay follows `packed-autograd.md`'s first-order
public-root boundary. Cache allocation and long idle-gap cost remain unoptimized.

## Independent anchors and original mapping

For width=heads=1, unit Q/K/V/output weights, zero biases and same-fiber inputs
1 and 3, output is `2+2*(sigmoid(2)+sigmoid(6))`, about 5.756649. This differs
from aggregate-as-token (4) and triangular visibility (about 3.995055).
With Q=K=0, initial V=5, zero bias/decay, current rows 1 and 3, output is 6;
its decay derivative at tick 0 is -4/3. These are explicit independent anchors.

Original LH `c_attn.weight` / `c_proj.weight` are transposed into Tide's layout;
missing original biases map to zero tensors. Local slots follow original incoming
CSC ordering plus appended bridge/token inputs. The bounded oracle links the
untouched source snapshot in STATUS under no-grad, testing LOOP, PACKED,
CACHEDMATMUL, CACHEDPACKED and CACHEDATTENTION with per-sample and multi-sample
projection, initial cache, clearing, idle ticks and capacity growth. CROSSBATCH is
added by the packed gate. The pooling extension covers bounded original Confluence
cases as described in `fiber-pooling.md`; IOCortexNet/Pronounce remains a separate gate.
See immutable evidence for which tests have actually passed.
The oracle independently checks Read against an FP64 norm of Tide's own
proposal. Cross-implementation norms inherit the proposal's payload tolerance:
promoting an FP32 proposal cannot remove its FP32 rounding error. This does not
relax route/selection identity or permit computing the norm in FP32.

FP32 comparison has a conditioning boundary: the initial development fixture
composed small attention outputs with repeated SiLU/RMS and squared-norm losses.
Proposal errors of 2.98e-8 amplified into VJP disagreement beyond the strict
default tolerance; both FP32 paths differed from the same FP32 data/weights run
in FP64. The retained `artifacts/fiber-dev-20260921-1601/` source snapshot and
failure log reproduce this. Acceptance isolates attention scheduling with tanh
Full and separately checks RMS composition with a nonzero output projection
offset and nonuniform directional cotangents. Tolerances are unchanged; no claim
is made that arbitrary ill-conditioned FP32 compositions meet them.
