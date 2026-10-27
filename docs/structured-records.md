# Named Read fields, typed state and source classes

Authority: core-5 TD §4–5/PDG §1 and weighted-message architecture §1–3 at the
revision in [upstream.json](upstream.json). This is a representation/interface
increment. The custom tests below do not certify the KV or allocation formulas.

## Read / Describe products

Python `ReadProgram.field_layout=None` retains the scalar descriptor contract.
A mapping `{field_name: shape_tuple}` declares a finite product of floating tensors;
`step` and `batch` then return dictionaries with exactly those names, shapes,
scoring dtype and scoring device. Each field is a separate public gradient root.
The field map passes unchanged through events and `RegionInput.candidates`.
Matching Region programs own its interpretation; an old scalar selector is not
implicitly reinterpreted as a structured selector. Layout/profile bindings must
be versioned by their owning module, just like existing custom program profiles.

C++ `ReadKernel::structured`, `field_layout`, `fields_step`, `fields_batch` implement
the same contract. `Event::descriptor_fields` and `Candidate::fields` retain each
field; the old scalar `descriptor` is undefined for a structured result. The
Python adapter exposes the same dictionary at `event["descriptor"]`. Core execution
is independent of Python. Default fields_batch is a reported scalar fallback;
these interfaces alone do not claim a fused or batched-VJP implementation.

`ReadInput.state=None/nullptr` still denotes the mode-restricted legacy content
interface. A weighted `Read^0` implementation must explicitly bind its mathematical
initial state and call Describe with it. The generic interface never substitutes
None for a mathematical state. The initial/old/proposal descriptor snapshot is
independent of the comparison snapshot read by Full, as tested below.

## State float tensors and int64 metadata

`State.value` retains the existing payload-width tensor contract. A state program
may declare `integer_slots` (Python set / C++ virtual method). Named slots in that
set must be int64 on the state device and nondifferentiable; all other slots retain
payload dtype and their own shapes. The program still validates its exact slot
names, dimensions and clock meaning. Generic clocks/counters remain int64.
A clock wrapper delegates this declaration; only last_time is reindexed by the
wrapper, while program-specific metadata semantics remain the program's concern.

State copy/detach, native conversion and checkpoint encode/decode retain every
slot dtype/value. Existing checkpoint v5 already stores typed tensors, so no schema
change is needed; the distinct program profile determines the legal domain.
Legacy kernels continue rejecting unknown/incorrect slots. Integer metadata is
compared exactly and excluded from the convenience training objective. Numeric
batch replay must also agree exactly on integer values, not only tensor shapes.

## Explicit packing and exact decoding

Python `tensor_fields.pack_fields` / C++ `pack_fields` pack actually computed named
fields into separate buffers per device/dtype, with explicit name, shape, dtype,
bank and offset metadata. Field ordering is canonical by name. Int64 timestamps
never enter a floating buffer. This is a packing primitive, not evidence that the
whole graph's messages or kernels have been batched.

In-memory `PackedFields` / `PackedTensorFields` retain the original computed field
handles solely as independent VJP anchors. Decoding reads the candidate buffers
and binds each field to its own original derivative graph, preserving None versus
connected-zero for unused fields. It never invokes a formula or scheduler to fill
in a result. Omitting the roots starts an explicit new autograd segment, suitable
for value serialization; it does not restore an old graph. Layout/buffer/domain
mismatches fail, and anchored integer fields must match exactly. These records are
not substitutes for a graph's complete checkpoint (state/history/pending/ledger).

## SrcClass versus SourceDomain

Python `SourceClasses` / C++ `tide::SourceClasses` classify physical input ports
and incoming edges. Category labels are nonnegative int64 and may repeat. The
classification never discards, merges or renames an atom; different physical
sources assigned the same class each participate once. `SourceDomain` still means
exclusive logical-source aliases in its existing profiles. The weighted aggregate
will use classes for parameter lookup while retaining physical atom identities.

## Independent checks and evidence boundary

| Test / symbol | What it establishes |
| --- | --- |
| `tests/test_structured_records.py::test_named_describe_integer_state_and_snapshot_vjp` | Literal .5*old+h proposal; bound initial, old and proposal Read give distinct memory fields; Full comparison remains proposal; PDG feedback and nonaligned DAG schedules agree on complete records; field roots preserve None for unused evidence; timestamps above 2^54 survive checkpoint exactly |
| `test_typed_packing_preserves_values_int64_and_independent_field_roots` | Distinct float roots and int64 bank, literal 12.6*a vector-root VJP, absent b gradient, value-only decode and metadata corruption rejection |
| `test_source_class_is_not_source_domain` | Three physical atoms in one class contribute 9*x, with independent input/coefficient derivatives; no class deduplication |
| `cpp/test/structured_records.cpp::check_structured_records` | Independent literal snapshot, gradient and int64/codec/class checks in standalone LibTorch scalar/frontier paths |

CPU reference tolerances remain those in [semantics](semantics.md); integer and
field domains are exact. Read batch training currently uses the existing explicit
semantic replay to preserve per-field roots; that cost must be included in future
profiling. Mixed placement, resident lowering and weighted builtin factory entries
need their own implementations and evidence. Live jobs/qualification status remain
in STATUS, and the complete remaining function/backend scope remains in ROADMAP.
