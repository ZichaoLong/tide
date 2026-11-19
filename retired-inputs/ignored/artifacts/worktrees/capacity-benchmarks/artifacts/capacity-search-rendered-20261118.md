# Bounded throughput search

These are finite observed candidates, not proven maximum capacity or mathematical/quality equivalence.
Times include complete selected-mode work; memory and intrusive audits retain their declared boundaries.
The common-shape table and independent fixed-model capacity table answer different questions.

## fc-k / bf16 / train / capacity

Source `f486a56ed7b6f68e2aff2e7100673bf2a2fe20be`; search state `completed`; children `{"completed": 24}`.
[Raw search](</mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-fck-budget-bf16-v2/results/search.json>) SHA256 `bc15c8a00304f2fbd37769048729130db91143fa8d8befeae8c9274f45e7aa82`.
Device budget 48.0 GiB; sampled child RSS budget 112.0 GiB; axis `batch` values `[32, 64, 128, 256]`.

Frozen calibration SHA256 `ad86861d928a93b415b33e5b4faf69f6e3f4ba4ea75878fb9aea65eaebe41fb5`. L remains 128.

| Subject | Best stable B | Seconds | Input positions/s | Timed peak GiB | Initial Tide active minus calibration | Stop |
|---|---:|---:|---:|---:|---:|---|
| tide | 256 | 0.737 | 44446.54 | 23.738 | 8708.00 | candidate_limit |
| transformer_A | 256 | 0.384 | 85225.15 | 8.518 | 8708.00 | candidate_limit |
| transformer_B | 256 | 0.420 | 78010.44 | 5.940 | 8708.00 | candidate_limit |
| transformer_C | 256 | 0.182 | 180171.28 | 3.530 | 8708.00 | candidate_limit |
| transformer_D | 256 | 0.106 | 308147.82 | 2.473 | 8708.00 | candidate_limit |

Different best B values are independent load optima under the declared budget, not same-workload speedups.
A missing active difference means Tide was not audited at that B; it does not mean zero activity change.

Stop reasons: candidate_limit; candidate_limit; candidate_limit; candidate_limit; candidate_limit.

## fc-k / fp32 / train / common

Source `f486a56ed7b6f68e2aff2e7100673bf2a2fe20be`; search state `completed`; children `{"completed": 24}`.
[Raw search](</mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-fck-length-fp32-v2/results/search.json>) SHA256 `6190371bbd7d0b2522142eb483f1d7f2282ba867e85f0efe20088e6afc865507`.
Device budget 48.0 GiB; sampled child RSS budget 112.0 GiB; axis `length` values `[128, 256, 512, 1024]`.

| B/L | Tide seconds | Input positions/s | Timed peak GiB | A time ratio | B | C | D | Stable subjects / admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 8/128 | 0.131 | 7803.21 | 1.129 | 1.272 | 1.192 | 9.331 | 7.815 | 5/5 |
| 8/256 | 0.168 | 12214.85 | 2.106 | 1.223 | 1.237 | 8.147 | 9.280 | 3/5 |
| 8/512 | 0.250 | 16352.99 | 4.067 | 1.304 | 1.319 | 6.590 | 8.933 | 5/5 |
| 8/1024 | 0.445 | 18400.58 | 7.976 | 1.526 | 1.539 | 6.614 | 9.776 | 5/5 |

A–D values are Tide/Transformer elapsed-time ratios at the same B/L; greater than one means Tide takes longer.

Stop reasons: candidate_limit.

## fc-s / bf16 / train / common

Source `f486a56ed7b6f68e2aff2e7100673bf2a2fe20be`; search state `completed`; children `{"completed": 18}`.
[Raw search](</mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-fcs-length-bf16-v2/results/search.json>) SHA256 `0628e30fdfb4067aece9aca588d361499efee0d5a6b5902ff7b485f39ea07003`.
Device budget 48.0 GiB; sampled child RSS budget 112.0 GiB; axis `length` values `[128, 256, 512]`.

| B/L | Tide seconds | Input positions/s | Timed peak GiB | A time ratio | B | C | D | Stable subjects / admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 8/128 | 0.263 | 3888.42 | 1.367 | 2.243 | 2.227 | 16.647 | 13.669 | 5/5 |
| 8/256 | 0.435 | 4704.55 | 2.637 | 3.105 | 3.182 | 20.700 | 18.010 | 5/5 |
| 8/512 | 0.780 | 5250.95 | 5.161 | 3.945 | 3.813 | 23.650 | 29.318 | 4/5 |

A–D values are Tide/Transformer elapsed-time ratios at the same B/L; greater than one means Tide takes longer.

Stop reasons: candidate_limit.

## hb-1b / fp32 / train / common

Source `f486a56ed7b6f68e2aff2e7100673bf2a2fe20be`; search state `completed`; children `{"completed": 18}`.
[Raw search](</mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-hb1b-length-fp32-v2/results/search.json>) SHA256 `f67d0150b71abd66f06740c4c030f0c243ad909fcb2e0f71a1a89cd618e67330`.
Device budget 48.0 GiB; sampled child RSS budget 112.0 GiB; axis `length` values `[32, 64, 128]`.

| B/L | Tide seconds | Input positions/s | Timed peak GiB | A time ratio | B | C | D | Stable subjects / admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1/32 | 1.516 | 21.11 | 19.461 | 0.534 | 0.548 | 13.454 | 38.746 | 5/5 |
| 1/64 | 1.566 | 40.88 | 21.338 | 0.551 | 0.508 | 13.418 | 38.544 | 5/5 |
| 1/128 | 1.763 | 72.59 | 24.501 | 0.639 | 0.646 | 13.926 | 40.017 | 5/5 |

A–D values are Tide/Transformer elapsed-time ratios at the same B/L; greater than one means Tide takes longer.

Stop reasons: candidate_limit.

## hb-multiscale-x-settle / fp32 / train / common

Source `f486a56ed7b6f68e2aff2e7100673bf2a2fe20be`; search state `completed`; children `{"completed": 18}`.
[Raw search](</mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-settle-length-fp32-v2/results/search.json>) SHA256 `83112b7c7f64fcc0993c5113280edb9947a166a92e8e115aa14fab8cd98887c3`.
Device budget 48.0 GiB; sampled child RSS budget 112.0 GiB; axis `length` values `[128, 256, 512]`.

| B/L | Tide seconds | Input positions/s | Timed peak GiB | A time ratio | B | C | D | Stable subjects / admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 8/128 | 1.326 | 772.00 | 7.799 | 2.176 | 2.148 | 38.163 | 37.864 | 4/5 |
| 8/256 | 1.923 | 1064.73 | 15.322 | 2.522 | 2.390 | 47.292 | 49.808 | 5/5 |
| 8/512 | 3.256 | 1257.82 | 30.515 | 3.033 | 2.979 | 48.054 | 52.305 | 4/5 |

A–D values are Tide/Transformer elapsed-time ratios at the same B/L; greater than one means Tide takes longer.

Stop reasons: candidate_limit.

## hb-multiscale-x-timed / bf16 / train / common

Source `f486a56ed7b6f68e2aff2e7100673bf2a2fe20be`; search state `completed`; children `{"completed": 12}`.
[Raw search](</mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-timed-length-bf16-v2/results/search.json>) SHA256 `7273f3fe4a4556ed81c14960f75beb52036208bf73f3c86c93d300e7a253c777`.
Device budget 48.0 GiB; sampled child RSS budget 112.0 GiB; axis `length` values `[128, 256]`.

| B/L | Tide seconds | Input positions/s | Timed peak GiB | A time ratio | B | C | D | Stable subjects / admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 4/128 | 4.109 | 124.62 | 13.543 | 6.195 | 5.973 | 127.472 | 122.681 | 4/5 |
| 4/256 | 6.673 | 153.46 | 27.386 | 7.806 | 7.798 | 150.829 | 145.187 | 5/5 |

A–D values are Tide/Transformer elapsed-time ratios at the same B/L; greater than one means Tide takes longer.

Stop reasons: candidate_limit.

