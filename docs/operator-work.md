# Inference operator-work accounting

The optional `--work-count 1` scale flag measures forward operator shapes in
the LH Attention instance. It is default-off, process-wide, CPU-only, and
requires no_grad and row Emit. It does not change tide-core-3 semantics.
Only one measured executor may run per process. Reset and snapshot happen
with workers quiescent; worker increments are relaxed atomic integers.

`op/*` counters use FMA=2 FLOPs. Matrix work excludes bias, normalization,
softmax, activation, pooling, indexing, cache copying, allocation and control.
It is an arithmetic inventory, not hardware counters or a total-cost model.

| Counter | Definition |
| --- | --- |
| qkv_rows/flops | Each source row projected D→3D, including Pronounce |
| out_rows/flops | Pooled node/sample event projected D→D, including Pronounce |
| emit_edge_rows/flops | Projected logical edge/sample blocks, D→D each |
| emit_rows/calls | Rows/calls of dense Emit projections; LH splits bridge/intra while PDG combines them |
| head_rows/flops | Vocabulary projection D→V (partitioning does not multiply work) |
| valid_score_elements | Heads × sum(query rows × available KV rows) |
| executed_score_elements | Actual score matrix shapes, including padding/masked future keys |
| valid/executed_attention_flops | QK and AV, 4 × D × corresponding query/key pairs |
| pending_edge_rows | End-of-token projected/awaiting-projection body edge blocks |
| body_candidates/selected | Node/sample/tick events, excluding Pronounce |
| aggregate_*_elements | PDG's generic source scale and summary-add element counts; LH zero means absent from this path |

Streaming PDG buckets samples by exact KV/query lengths; sequence prefill may
still compute masked future keys. LH CROSSBATCH gathers each query's own sample
cache, padded to the participating samples' maximum KV length. It never implies
cross-sample attention. Counts include only this original multi-batch CROSSBATCH
path; other LH profiles/modes and backward are outside this contract.

LH computes Emit at the following body tick. PDG computes it when sending.
For a window, LH generated work is `executed + pending_end - pending_start`.
PDG delayed-consumption work is `executed - pending_end + pending_start`.
Either adjustment aligns this boundary only; independently initialized models
can still choose different routes. Do not infer functional parity from counts.

`build_lh_original.py --accounting` instruments a new project-owned a10fdb1
copy using unique source anchors. All changed/added files and binaries are
hashed. Tensor expressions and scheduling remain original. The test sets an
explicit seed and fixed `(3*sample+7*token)%vocab` IDs to match PDG's input
policy; it does not match weights. `benchmark_lh_original.py --work-count 1`
merges per-token JSON sidebands with original Think timers; counters reset
before Think, JSON/checksums print after Think. Actual runtime pools are logged.
The accounting-enabled test supports a bounded small complete-logit audit.

Validation: `tests/test_operator_work.py`, native fiber-policy analytic prefill
and clear cases (via `tests/test_cli.py`), and `scripts/check_lh_work.py` on a
small prepared LH fixture. Scale `--check 1 --work-count 1` compares complete
trace/cache/pending/output continuations against an uncounted scalar schedule.
Large counts and timings are evidence for the measured window only.

Measured wide comparison and exact validation scope: [evidence](evidence/lh-pdg-operator-work.md).
