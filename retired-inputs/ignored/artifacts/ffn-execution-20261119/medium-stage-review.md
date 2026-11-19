# Medium synchronized-stage review

Tool `a5853660`; production tree `89f2707e71d989340c12de310797a4c7f4c99f64`. Four jobs completed with outer status passed. Verified 1,940 frozen tracked files and preserved raw fingerprints. Read-only analysis; no model/device work.

FP32, HF32 off, physical NPU 9, CPU 224–227/4 threads. Tide d192/KV32/FFN768 has 129,025,464 parameters; Transformer32 d496/FFN2048 has 129,039,856. B16/L128/chunk32, two warmup updates; each diagnostic restores its own first-measured-update anchor, state 256→384, AdamW 2→3. This is medium backbone training with a synthetic mean-square objective, not 9B or task-quality evidence. Transformer is an auxiliary near-total reference, not A–D.

**Three passes:** ordinary timing, nested synchronized scopes, and individually synchronized operators. Do not add them or divide diagnostic times by normal latency. The three-repeat throughput result remains [the timing review](medium-npu-timing-review.md); each job here has only one normal witness.

| Pass / seconds | baseline | W | P | Transformer32 |
|---|---:|---:|---:|---:|
| One normal complete update | 2.115853 | 2.132701 | 2.394711 | 0.424362 |
| Stage replay forward | 1.519113 | 1.535032 | 1.797426 | 0.337601 |
| Stage replay backward | 0.982358 | 1.057407 | 1.102379 | 0.219955 |
| Stage replay training root | 2.521265 | 2.612456 | 2.919754 | 0.571134 |
| Operator replay elapsed | 13.956623 | 14.093143 | 15.049887 | 3.002219 |
| Sum of synchronized operator intervals | 6.338494 | 6.253187 | 6.771061 | 1.284829 |

The following FFN rows are an **exclusive partition of the stage replay**, including normalization and row preparation. Each scope row is assigned once. Parent remainder includes hook/dispatch and entry waits for unwrapped prior work; it is not pure Python or a hardware kernel duration.

| FFN forward exclusive seconds | baseline | W | P | Transformer32 |
|---|---:|---:|---:|---:|
| ffn.caller.remainder | 0.012335 | 0.014213 | 0.032236 | 0.018722 |
| ffn.grouped.matmul | 0.000000 | 0.018838 | 0.000000 | 0.000000 |
| ffn.grouped.plan | 0.000000 | 0.021323 | 0.000000 | 0.000000 |
| ffn.grouped.projection.remainder | 0.000000 | 0.028708 | 0.000000 | 0.000000 |
| ffn.node_linear.remainder | 0.048451 | 0.010677 | 0.080411 | 0.000000 |
| ffn.normalization | 0.006870 | 0.006741 | 0.022527 | 0.016492 |
| ffn.rows.apply.remainder | 0.001931 | 0.001898 | 0.050298 | 0.000000 |
| ffn.rows.bucket | 0.000000 | 0.000000 | 0.088129 | 0.000000 |
| ffn.transformer.linear | 0.000000 | 0.000000 | 0.000000 | 0.025318 |
| Total FFN / row preparation | 0.069587 | 0.102398 | 0.273599 | 0.060531 |

W builds 48 owner plans and reuses each for its three projections: 144 grouped forward calls and 288 native adjoints. The plan is not rebuilt per projection. W removes the forward float-bank gather, but activation sorting/inverse selection and grouped-plan construction remain. Its FFN row-apply scope rises 0.069587→0.102398 s.

P keeps attention rectangles unchanged. KV FFN selected/executed rows are 118,509/164,244 versus 118,515/335,104 baseline; KV FFN calls 28→131, with SSD unchanged at 81,920 rows/20 calls. Its row-apply scope 0.273599 s contains bucket construction 0.088129 s, row-apply remainder 0.050298 s and FFN body 0.135173 s. The last body is already split among normalization/projections/caller remainder above; do not add it again. `z=norm(x); FFN(x)=x+Wd(SiLU(Wg z)*Wu z)` has dense forward cost `3 R d f`; reduced R does not remove the extra calls and indexing.

| Separate operator pass, FFN forward | baseline | W | P |
|---|---:|---:|---:|
| Float-bank gather calls | 144 | 0 | 453 |
| Float-bank gather seconds | 0.035518 | 0 | 0.051945 |
| Float-bank logical output bytes | 1642070016 | 0 | 1642070016 |
| Activation index_select calls | 0 | 288 | 0 |
| Activation index_select seconds | 0 | 0.020654 | 0 |
| Forward matrix calls | 144 | 144 | 453 |
| Forward matrix seconds | 0.014544 | 0.018561 | 0.026238 |

P retains the same 1,642,070,016 logical weight-gather output bytes but spreads them over 453 calls instead of 144. W activation index_select produces 4,804,116,480 logical output bytes across 288 calls. These are repeated logical output events, not bus traffic, peak storage or a basis for cross-pass timing subtraction.

| Allocation forward exclusive seconds | baseline | W | P |
|---|---:|---:|---:|
| allocation.control.remainder | 0.020060 | 0.019703 | 0.019590 |
| allocation.validate_solve.remainder | 0.089821 | 0.089569 | 0.087799 |
| allocation.sorted | 0.106340 | 0.106393 | 0.112355 |
| allocation.bisect | 0.182795 | 0.181075 | 0.170889 |
| allocation.save_face | 0.011198 | 0.011618 | 0.010969 |
| observer.metadata | 0.008194 | 0.007876 | 0.008030 |

All three paths execute 48 sorted solves over 90,112 rows and 28 fallback calls over 49,152 rows. Only 324/323/337 fallback rows are requested; each call still evaluates the entire width 8 rectangle for 40 bisections plus 3 correction iterations. Sorted invalid rows 375/380/395 include inactive rows and are not fallback counts. Saving the face records slope/boundary masks for VJP. Any fallback compaction or solver change requires separate numerical/gradient qualification.

**Observer reads are separate:** each Tide replay has 76 `observation.metadata` scalar reads, 1,402 scalar reads inside the training path, and 1 extra diagnostic loss read outside training. Raw `scalar_host_reads=1403` excludes metadata but includes that final reporting read. These counts are neither all device synchronizations nor production wall-time costs; observer scope synchronizations and dynamic-size operations are separate.

| Whole backward operator pass | baseline seconds / calls | W seconds / calls | P seconds / calls |
|---|---:|---:|---:|
| aten.index_put.default | 0.316084 / 1660 | 0.280958 / 1516 | 0.416856 / 2259 |
| aten.index_add.default | 0.000000 / 0 | 0.041454 / 288 | 0.000000 / 0 |
| aten.bmm.default | 0.156771 / 1824 | 0.127595 / 1536 | 0.181136 / 2442 |
| aten.slice_backward.default | 0.088925 / 883 | 0.102550 / 1027 | 0.087215 / 883 |
| aten.new_zeros.default | 0.117895 / 1660 | 0.126350 / 1804 | 0.163814 / 2259 |
| npu.npu_grouped_matmul.default | 0.000000 / 0 | 0.035947 / 288 | 0.000000 / 0 |

Native grouped adjoints have explicit FFN scopes. Generic backward indexing/scatter/zeroing rows do not retain forward module ownership; their changes are whole-model observations, even where counts and shapes fit the expected mechanism. They cannot prove that every added backward millisecond belongs to FFN packing or weight assembly.

All 8 diagnostic replays pass original 1e-6/0.002 comparisons against that candidate’s own normal update. Readout and complete private-state comparisons are exact; Tide gradients/parameters/AdamW are finite and within threshold but nonexact. This is in-process anchor replay, not new-process checkpoint qualification. Raw output strengths/times are not retained by ordinary run_stream. Cross-candidate warmed anchors and activity differ, so these remain finite execution comparisons, not strict same-trajectory speedups; P’s existing cross-path qualification failures remain.

The evidence supports the earlier throughput decision: retain baseline for this same-B medium training case; W/P save recorded memory but did not improve the three-repeat normal throughput. Allocation fallback rectangles, packing dispatch and generic reverse assembly are bounded future opportunities, not demonstrated recoverable gains. Full exclusive cuts, source/input identities, replay checks, raw references and exact values are in [medium-stage-review.json](medium-stage-review.json).
