# Bounded canonical owner transfer

Implementation: `7e375f5abeea76c2c5406ef91d1919facba56470`.
[Audited records](resident-owner-stream-20261002.json). All eight fixed-source
jobs passed; no skipped cases, tolerance relaxation or implicit dtype fallback.

Canonical gradients now consume contributions in reverse-window/alias ordinal
order, grouping independent owners by device pair within each ordinal. CANN
loops reuse bounded FP32 packets for reduction and master publication, including
strided aliases, FP16 rounding and incomplete final packets. Neither reference
results nor host per-packet decisions feed execution. Each packet endpoint is
capped at64MiB and shrinks within its tensor reservation.

Optimizer identity metadata no longer retains an obsolete initial gradient bank;
public training layout geometry shares its master storage. Existing independent
CPU references and optimizer/checkpoint semantics remain the acceptance anchors.

| Fixed-source gate | Result |
| --- | --- |
| Standalone, Python-owned backend, installed consumer | Three builds passed; affected host objects/dependencies verified, fresh links and loader closure |
| Dedicated transfer and canonical optimizer | 20 transfer cases/100 replays; two order-sensitive reduction replays; FP32/FP16 each4 optimizer trajectories/32 updates |
| Public standalone training | 32 trajectories,512 continued windows,128 updates; FP32/FP16,SGD/AdamW,both schedules,explicit Full/state maps and2→3-card restore |
| Actual Python-owned consumer plus projection training | 17 cases passed |
| Actual standalone consumer | 14 cases passed |
| Separate actual FP16 Attention training profile | Two cards,four continued windows,two AdamW updates,diagnostics off |

Actual consumer checks contain24 whole-model FP32/FP16 training trajectories
across PDG,TimedDAG and SettleGraph plus four forced head-splitting trajectories.
They compare loss,discrete state/routes,gradients/None and updated parameters with
independent CPU execution. Three additional Python-owned projection cases cover
shared-owner gradients,masters,slots and checkpoint restore.

## Allocation calibration

A4,194,307-element FP32 source (16MiB plus12 bytes) was published over65 device
packet iterations with capacity65,521 elements. Source and destination existed
before peak reset. Both endpoint packets plus metadata reserve528,402 bytes.
Measured TorchNPU allocator peak deltas were344,576 and347,136 bytes on the two
logical devices; the two CANN programs separately report77,312 bytes of operator
workspace each. These counters have different scopes and must not be blindly
summed. The transfer did not allocate complete numerical send/receive banks.

This calibration excludes preexisting caller tensors and untracked vendor/driver
HBM. It is not a proof of total-model admission. Public reverse counters report
planned packet iterations and tensor reservations,not observed active rows or
allocator peaks. Full gradients,forward/retained banks and optimizer slots remain
real storage requirements.

## Independent profile and retained failure

The actual D32 Attention consumer trace contains18,449 AI_VECTOR_CORE,704 AI_CORE
and258 MIX_AIV operator records. `tide_owner_stream` executes on device; no AiCPU
operator was observed. Construction is included; this trace is not throughput
measurement or evidence that NPU outperforms CPU. Each update records eight
canonical reduction packet iterations and891,940 reserved stream bytes.

Development build `build-owner-stream-dev01` failed on ambiguous C++ empty Tensor
assignment. Explicit `Tensor{}` fixed the compilation error; its failed record
and log remain unchanged. All fixed-source jobs succeeded.

Reproduce the local evidence audit with
`python TASK/launchers/owner_stream_evidence.py 7e375f5abeea76c2c5406ef91d1919facba56470`.
Artifacts use source `owner-stream-clean01`, backends `owner-stream-clean01` and
`owner-stream-python-clean01`, consumer `owner-stream-consumer-clean01`, and the
 eight build/component/native/session/LibTorch/profile run records in the JSON.
Total per-card memory admission,representative screening and full-size F6 remain
pending; this increment closes bounded canonical transmission only.
