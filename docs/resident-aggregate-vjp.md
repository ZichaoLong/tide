# Device normalized Aggregate adjoints

`tools/device_online/aggregate_vjp.h` adds first-order FP32 adjoints for mean,
positive weighted mean, active softmax and all-source softmax to the resident
graph reverse loop. Sum keeps its existing path. This document describes the
implementation; [STATUS](STATUS.md) and [ROADMAP](ROADMAP.md) distinguish
development tests from immutable qualification.

The forward owner supplies its actual fiber records, physical source scales,
logical source mapping and frozen coefficient banks. Retained windows copy
these banks with their device tape. No CPU event trace, normalization result
or gradient is supplied to the candidate.

For each reverse stage, device control packs only connected events of the same
Aggregate kind into bounded rows. Each row records the complete logical domain
needed by its formula. Physical scale is applied before normalization. The
packed Jacobian returns message gradients, physical-scale partials and logical
coefficient partials. Device contributor lists reduce repeated node/slot owners
in stable order; the existing parameter registry then accumulates all declared
aliases before one optimizer update and publishes every updated forward alias.

For a scaled message `y_j = s_j*x_j`, coefficient `p_j`, and upstream vector `u`,
let `d_j = dot(u,y_j)` and `c = sum_j p_j*d_j`. Then:

- Mean has no trainable coefficient and sends `u/|A|` to each scaled message.
- Weighted mean uses `m_j = softplus(a_j)` and total present mass `M`;
  its coefficient VJP is `sigmoid(a_j)*(d_j-c)/M` for present sources only.
- Active/all-source softmax uses `p_j*(d_j-c)`. Active softmax connects only
  present logits; all-source softmax also connects absent logits through its
  full denominator, with absent `d_j = 0`.
- Each present message receives `p_j*s_j*u`; its physical scale receives
  `p_j*dot(u,x_j)`. Raw messages are retained, so zero scales require no division.

Missing messages and present zeros remain distinct. A connected zero upstream
still connects every parameter in that formula's logical domain. Poisoned
disconnected owners and absent active-domain coefficients are never evaluated.
Distinct physical alternatives can share one logical source, but two alternatives
in the same event are rejected, including when one message is zero. Physical
message and scale identities remain distinct from coefficient ownership.

The component preallocates row buffers under its tensor budget and advances
their cursor on device. No per-event scalar download drives the loop. Retained
tape storage, graph outputs, Aggregate scratch and serial CANN operator workspace
have separate reservations. A single row that cannot fit is rejected; physical
chunking never changes a logical gradient, update boundary or retained window.
Short/empty replay clears previous values, connections and chunk counts.

The isolated checker compares independent CPU FP32/FP64 autograd for widths
1/7/257, logical domains up to257, physical/logical permutations, None/nonzero/
connected-zero roots, absent and present-zero sources, zero scales, poisoned
unused owners, shorter/empty replay, budget refusal and duplicate-source refusal.
The public trajectory checker adds feedback, self-loops, mixed Aggregate kinds,
cross-node/coefficient/physical-scale aliases, both schedules and optimizers,
retained windows, actual parameter updates and checkpoint continuation. Python
client tests separately cover three graph families and fresh-process disk resume.

The training gate uses explicit AdamW epsilon1e-5, retaining the public default
and ordinary comparison tolerances. It does not claim arbitrary optimizer
trajectory agreement, attention adjoints, HST/SOFTP, FP16, peer progression or
throughput. Those remain separate parts of the execution contract.
