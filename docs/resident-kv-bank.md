# Resident KV entry bank and snapshot views

The concrete Settle block executor implements fixed KV §§1.3, 3–4 with one
initial/write bank per region. See the [semantic map](semantic-map.md),
[graph domain](resident-block-flow.md) and [public owner](resident-block-session.md).
This is an internal representation and an explicit observation API migration;
the mathematical functions and raw checkpoint schema remain unchanged.

## Values, time and snapshots

`BlockKVBank` holds keys/values/summaries/weights once as `[B*N,C+L,width]`
fields (weights have no last width axis). The first C slots encode the actual
raw initial state; the remaining L slots name potential writes at original
logical times. Bool validity and int64 timestamps remain separate. A present
candidate and a positive KV write are different facts. Missing and zero messages
do not become synthetic writes.

Old and proposal snapshots use int64 `[B*L*N,C]` entry mappings plus their actual
raw clocks, counts and timestamps. Expiration follows original logical time;
it does not compact time into event counts. Read− takes old, Read+ takes proposal,
and Read0 takes a separate bank from the bound initial state. Full takes proposal
and actual regional support. The final `[B*N,...]` raw state is decoded for
continuation/checkpoint. Capacity bounds simultaneous storage, not total time.

`block_kv_bank` owns the representation and actual decoder. `block_kv_read_plan`
derives visibility, original ages, decayed weights and owner indices on device.
`block_kv_memory` projects actual Read keys/values once per bank for qry; sum
reads summaries and has no synthetic query/key/value parameter roles.
`block_kv_full` projects queries and computes head-specific Attention/residuals.
`block_kv_match` uses BMM for ordinary matches and a position kernel where required.
`block_kv_pool` keeps values `[B*N,C+L,width]`, with `[B*N,L,C+L]` weights/logits;
values are not duplicated across queries. Static head/channel loops remain.

## Reverse and observation

Memory/Full VJPs produce bank adjoints and structural root masks. The local
reverse sums their numerical contributions and independent raw snapshot roots,
then invokes the actual write VJP once. Snapshot causal connectivity is distinct
from visible entries and from nonzero numeric gradients: expired entries may
retain connected-zero ancestry. Read0 contributions return to the bound initial
root, not the current continuation. Shared parameters reduce through the existing
canonical owner mapping; private state and initial roots remain private.

Ordinary `advance` publishes `kv-bank-views-v1`: old/proposal values are undefined,
with explicit `shape`, presence and validity. Named cotangents use that schema
without materializing values. `observe` explicitly decodes a retained token's
requested actual raw fields in a bounded device program, returning copies of
decoded values and borrowed metadata. No reference formula supplies missing
results. Token owner/generation, version guards, workspace/tensor budget and
execution failure rules still apply. Complete Python records call this boundary;
outputs/metrics do not. A caller needing mutable metadata must clone it.

Observation changes do not alter block checkpoint schema 2: final raw floating,
int64, initial bindings, optimizer and input ledgers still decode exactly. An
effective-state projection is insufficient for checkpointing. The implementation
currently supports FP32, one device and the block executor's declared KV domain.
It does not qualify general PDG, recurrent memories, half precision or multiple
owners as new bank implementations.

## Independent checks and limits

- `test_resident_kv_bank.py` compares device-kernel CPU adapters with an independent
  list recurrence and roots: empty/nonempty initial, expiry, huge int64 time,
  no-write, Read0 and opaque raw padding.
- `test_resident_kv_pool.py` uses FP64 formulas, directions/roots and a large common
  value offset; `test_resident_kv_read.py` checks actual visibility, position match
  and reverse. These are component checks, not device graph qualification.
- `block_settle_flow_check.cpp` uses an independent scalar formula for ten root
  sets under Read+ and all/none under Read−/Read0; it explicitly decodes actual
  old/proposal fields. It covers duplicate edges/ports, shared parameters,
  missing/zero inputs and huge time with a nonempty, unbounded-window fixture.
  Expiration in a complete device graph is not inferred from these fixtures.
- The standalone public/session checks and installed `block_resident.py` cover
  independent schedules, training and fresh-process raw restoration. The Python
  consumer additionally checks old/proposal decoding and proposal keys/values
  roots. `test_resident_kv_observations.py` checks roots without decoding;
  `test_experiment_block_records.py` rejects incomplete observation material.

Build, device execution, full-graph training and performance evidence are recorded
separately in the [evidence index](evidence/README.md). A shared bank and fewer
copies alone do not establish wall-time improvement or complete P1–P6 delivery.
