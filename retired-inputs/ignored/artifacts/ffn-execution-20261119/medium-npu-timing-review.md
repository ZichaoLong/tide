# Medium NPU normal-training timing review

Frozen tool `2b8851a7`; production `05e8338b`, src/tide tree `89f2707e71d989340c12de310797a4c7f4c99f64`. All 15 result hashes and 484 frozen tracked files verified. Suite completed, outer status passed, cooperative queue completed; final live descendant audit belongs to the main report.

Single Ascend NPU physical 9, CPU 224–227 / 4 threads, FP32 with HF32 disabled. B16, L128, chunk32; two real warmup updates then five continuing updates in each of three serial paired repeats. Tide d192/KV32/FFN768 has 129,025,464 parameters; auxiliary Transformer32 d496/FFN2048/31×16 heads has 129,039,856. Transformer is not A–D. This is synthetic backbone mean-square training, with no task head or quality result.

Throughput per repeat is 10,240 input positions divided by the sum of five complete update times. The paired ratio is candidate throughput / baseline for the same repeat. Medians do not imply confidence intervals.

| Candidate | r0 / r1 / r2 positions/s | Paired ratio median [min,max] | Max allocated / reserved GiB |
|---|---:|---:|---:|
| baseline | 961.595 / 883.923 / 883.702 | 1.0000 [1.0000, 1.0000] | 22.739 / 23.777 |
| W | 942.152 / 882.153 / 875.783 | 0.9910 [0.9798, 0.9980] | 21.517 / 22.520 |
| P | 821.560 / 747.421 / 761.036 | 0.8544 [0.8456, 0.8612] | 20.598 / 21.494 |
| WP | 788.020 / 718.749 / 707.665 | 0.8131 [0.8008, 0.8195] | 19.273 / 20.291 |
| Transformer32 | 4593.017 / 4501.196 / 4448.307 | 5.0337 [4.7765, 5.0923] | 6.644 / 7.670 |

W means grouped direct weight-bank access; P means power2 owner-row packing; WP combines them. W has no measured throughput gain; P/WP are slower in all three repeats while reducing the recorded allocator peak. Retain baseline for this medium same-B throughput. Material repeat drift remains: baseline r0 is about 8.8% faster than r1/r2. This does not establish a 9B or best-batch recommendation.

**Memory boundary correction:** raw `boundary.memory` says peaks are sampled “before checks”, but `continuous.run_stream` runs its untimed finite checks before returning, and `execution_candidates.ordinary` samples the allocator afterward. These peaks include those checks and exclude the later continuation audit. Complete update timing ends before the checks and is unchanged. Raw evidence was preserved. RSS is process lifetime.

Only r0 has separate untimed work and storage diagnostics, each restored to that candidate’s own warmed anchor at position 256. They are not activity logs of all timed samples. All four Tide initial parameter/input hashes match; all four warmed parameter, AdamW, gradient and private-state hashes differ. Full counts are 200435 / 200441 / 200429 / 200441 for baseline/W/P/WP. W/WP equal aggregate counts do not establish equal support or trajectory. Every Tide final parameter hash also differs across repeats; no rounding or other causal explanation is inferred.

| r0 separate work audit | baseline | W | P | WP |
|---|---:|---:|---:|---:|
| KV FFN selected rows | 118515 | 118521 | 118509 | 118521 |
| KV FFN executed rows | 335104 | 335104 | 164244 | 163812 |
| KV FFN calls | 28 | 28 | 131 | 132 |
| All FFN executed rows | 417024 | 417024 | 246164 | 245732 |
| All FFN calls | 48 | 48 | 151 | 152 |
| FFN forward multiply-add pairs, billion | 184.478 | 184.478 | 108.895 | 108.704 |

KV attention query rectangles remain 335,104 rows; SSD stays at 81,920 selected/executed rows and 20 calls. Packing reduces KV FFN rows by about 51%, but increases calls 28→131/132. For `FFN(x)=x+Wdown(SiLU(Wgate norm(x)) * Wup norm(x))`, forward dense cost is `3 R d f` multiply-add pairs. Packing gathers valid rows into per-owner capacities 1,2,4,… capped at B×Q, shares the row map across the three projections, then scatters back. Grouped weight access sorts activation owners and reads a bank view via native grouped matmul. Independent owners remain independent. Across r0 KV-down dispatches, both W/WP visit 768 nonempty owner groups; the summed bank-view spans are 832/3,920, including empty groups. These span counts are not copied-weight bytes.

The separate ledger observes FP32 native grouped matmul: 432 calls for W and 1,368 for WP over forward/backward/optimizer. BMM counts are 2,640 baseline and 3,567 P. `index_put` logical output events total 21.872 GB baseline, 58.064 GB P, 10.944 GB W, 12.144 GB WP. Those repeated bytes are not bus traffic, bandwidth or peak storage. More calls/indexing/gradient assembly can offset saved arithmetic, but this ledger does not time-decompose the slowdown.

| r0 separate ledger, GiB | baseline | W | P | WP | Transformer32 |
|---|---:|---:|---:|---:|---:|
| Forward exclusive saved storage | 19.984 | 18.753 | 17.792 | 16.433 | 4.428 |
| Peak unique saved storage | 19.987 | 19.138 | 17.795 | 16.819 | 4.913 |
| Repeated saved payload events | 25.376 | 25.482 | 22.941 | 28.129 | 6.599 |
| Forward unclassified live allocator | 0.645 | 0.604 | 0.646 | 0.626 | 0.273 |

Tide forward ownership is 0.481 GiB weights, 0.961 GiB AdamW and 0.195 GiB private state; gradients are 0 at forward and 0.481 GiB after backward. Input/output are each 0.001465 GiB. W/WP saved views can alias parameters: unique saved storage and exclusive saved storage differ. Do not add their peaks to allocator/parameter peaks. Unclassified allocator remainder includes unseen tensors/backend effects, not only workspace. Ledger phase peaks are cumulative after reset, not additive phase increments; boundary includes prior warmup work. The ledger uses summed per-chunk readout squares, a different reduction order from the normal joined-readout objective.

Normal runs record finite output/parameters/gradients/optimizer/private state and detached continuation for 75 measured updates; this is not medium cross-path numerical qualification or new-process recovery. Existing small NPU qualification passes baseline/W at the original 1e-6 + 0.002|reference| threshold; P and WP each retain 11 failed field comparisons over continued updates, with no recorded nonfinite, support, dtype, metadata or original-time failure. Each candidate’s own new-process restore update passes at that small scope. CPU combined retains 2 failed field comparisons / 3 scalar coordinates. None is erased by this finite timing suite.

Reproduction inputs, exact commands, 75 raw elapsed samples, paired ratios, memory phases, work/operator summaries, fingerprints and qualification references are in [medium-npu-timing-review.json](medium-npu-timing-review.json). Raw suite: `artifacts/ffn-medium-npu-timing-v1/results/suite.json`. No model or device work was executed for this review.
