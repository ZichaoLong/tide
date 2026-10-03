# Bounded consumer owner placement

Qualified clean source `c38b72ebf1e66804e09dcf72f083ee4e6c1e5850` on2026-10-03.
[Reviewed record](consumer-memory-balance-20261003.json) pins builds,inputs,
terminal jobs,independent comparisons,allocator observations and profile CSVs.
[Contract](../consumer-capacity.md).

Aggressive automatic placement now tries bounded joint Full/state owner moves
when every operator maximum has reached one and the initial placement still
exceeds a card's complete envelope. Each move strictly improves integer memory
excess/peak scores; shape-equivalent trials share cost evaluation,with static
physical-edge locality and node/device order resolving ties. No owner set becomes
empty. At most2×nodes moves and4096 trials; requested operator maxima are retried
on the new map. Nonempty explicit Python maps and conservative placement stay
fixed; empty tuples/lists mean automatic placement. Canonical optimizer ownership,
logical work,API capacities and all safety margins are unchanged. No numerical
prepass,CPU events,routes or gradients enter planning or candidate execution.

Seven clean jobs passed/exit0 with empty control groups and released leases:
three builds,CPU29,NPU36,D512 allocation calibration and separate FP16 profile.
NPU36 contains32 actual independently CPU-referenced candidates across
Add/Attention,FP32/FP16,Python/native and standalone,operator/owner pressure,
automatic and ragged sample chunks,complete updates and continuation. Three
additional cases check pre-allocation or KV-journal refusal; one repeats a CPU
boolean-interface check. No skips. Core/CANN dependencies and unchanged archive
members were byte-verified; two affected host members use source/header/options-
verified development objects and fresh links. Internal failure accessors add no
object-layout change. This is not a from-scratch vendor build claim.

D512/B8/T4/V257 Attention,128 body nodes,physicalB2×4,two connected windows,
one FP32 AdamW update: the bounded budget forces one owner move in six trials.

| Bytes | Logical0 | Logical1 |
| --- | --- | --- |
| Admitted peak estimate | 21159582340 | 13365092892 |
| Observed allocator growth | 6347777024 | 5545201152 |

Full16/emission1/aggregate8/attention2/key128/reverse1/head128 are the final
operator maxima. Loss7.532632350921631 matches prior7.532631874084473 within the
existing FP32 tolerance; events3163,stages104,output counts,cut and connected
window/update boundaries agree. Per-card allocation and continuation-pool gates
pass. Changed placement and operator cuts explain why this is calibration,
not an isolated placement-speed or allocator-saving comparison.

The separate two-card FP16 training profile also forces an owner move and keeps
three physical sample groups/two complete updates. It contains88089 operator
rows (83500 AI_VECTOR_CORE,1049 MIX_AIV,3540 AI_CORE); no AiCPU observed.
Profile time is not formal throughput. Its KV journal uses a static all-node
bound3024 for the declared twelve token positions,independent of actual selection.

Retained failures:

- CPU dev01's old weak-card refusal assertion became invalid after legal moves;
  dedicated owner-move and all-card sample-pressure cases replaced that assumption.
- NPU dev03 retained23 passes/10 failures. Empty automatic tuples incorrectly
  disabled Python fallback; the fix passed four targeted native cases and the
  clean gate. B17 also exposed an underprovisioned2048-row KV journal.
- Dev04/dev05 diagnostic reproductions remain failed. At refusal the committed
  general journals had170–238/2048 rows and KV journals952/2048 before an
  over-capacity append. Failure-only counters run after device completion and do
  not return a successful continuation. Tiny complete B17 tests now use the
  static all-body bound17×12×(8²−6²)=5712; two tests still exercise the original
 2048-row refusal. Original full-size experiment capacities did not change.
- Standalone clean01's build helper could not find an original compile command
  through recursive object reuse. Clean02 used the same source/header/options
  with original dev05 compile records. The failed job remains failed.

An initial audit helper expected a nonexistent profiler success string. Hash and
consumer result checks passed; the marker was corrected to the actual serialized
consumer success record. The initial helper is preserved; no run was rewritten.

Raw source `TASK/sources/memory-balance-clean01`; builds
`memory-balance-standalone-clean02`, `memory-balance-python-clean01`,
`memory-balance-consumer-clean01`. Runs share the `memory-balance-*-clean01`
prefix except the standalone build above. `TASK=/mi/data2T/zlong/tide-execution-flows`.
Audit: `python TASK/launchers/memory_balance_evidence.py c38b72ebf1e66804e09dcf72f083ee4e6c1e5850`.

This qualifies bounded placement and diagnostics. Static fitting does not prove
original Attention B512 complete training or throughput. Full-size CPU/mixed/
resident comparisons,mixed multi-card integration and F7 remain pending;
CUDA requires target-machine validation.
