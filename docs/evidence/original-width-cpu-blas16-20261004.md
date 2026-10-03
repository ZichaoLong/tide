# Original-width continuous CPU calibration with explicit BLAS16

Four LibTorch/TimedDAG/prefill FP32 cases passed on clean `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`, 2026-10-04. These retain the original 480 body nodes, 2208 body edges, D2048/T12/V50304 and 9,468,053,696 Add or 17,521,117,376 Attention parameters, with reduced logical batch. **They are cost/memory pilots, not actual B512 timings or CPU/NPU comparisons.** [Audited results and exact commands](original-width-cpu-blas16-20261004.json).

Each fresh process executes one continued warmup and one measured step, two connected windows per step. Training includes forward/loss, backward, finite checks and SGD. Physical splitting preserves the logical batch/update boundary. ATen16/workers1 is unchanged; startup uses `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=16`, `MKL_NUM_THREADS=1`, `OMP_THREAD_LIMIT=32`, following the [separate BLAS diagnosis](cpu-blas-policy-20261004.md).

| Case | Logical/physical batch | Warmup s | Measured s | B512 warmup/measured forecast s | Peak RSS growth GiB |
| --- | --- | ---: | ---: | --- | ---: |
| Add training | 64/32 | 105.561763 | 115.645236 | 939.407377/1039.361385 | 115.290 |
| Attention training | 32/16 | 536.410864 | 535.885729 | 9544.565449/9692.312292 | 311.844 |
| Add inference | 64/32 | 26.999169 | 28.504013 | 248.392358/262.236922 | 40.933 |
| Attention inference | 64/32 | 163.179488 | 210.936170 | 1501.251293/1940.612768 | 79.435 |

Forecasts are `(sample_work_seconds × 512 / pilot_batch + optimizer_seconds) × 1.15`. Optimizer time is not multiplied by the sample ratio. They include the safety coefficient and support launch admission only; batch-dependent routes, overhead and machine load can change actual time. All pilots used phase instrumentation and overlapped the separate NPU profiling job. One process per cell is insufficient for a formal recommendation.

All four checks reached final cut816 and returned24outputs per logical sample, retained their requested two physical chunks and stayed within the declared memory estimates. Full reported event counts and losses are retained in the JSON; equal input shape does not imply equal CPU/NPU work. CPU peak measurement is process-lifetime RSS growth, not allocator accounting. Add training construction took48.238962s; construction is separate from the table's step times.

Attention training **still refuses the original3000s guard**: warmup9544.565449s/measured9692.312292s forecasts. The separately declared, unexecuted full-size budget permits12000s per update and24400s per process, including400s construction allowance. The original refusal and all BLAS1 history remain intact. The other three cases fit3000s forecasts; their actual B512 timings remain pending.

Audit checked all frozen source inventory hashes, source-matching unchanged CPU consumer/core identities, plan/input/launcher/result hashes, continuous step/output boundaries, finite scalars, phase accounting, capacity observations and terminal exit0/empty cgroup. It does not add full-state cross-backend equivalence evidence. Re-audit: `python TASK/launchers/audit_cpu_blas_continued.py`; raw job `TASK/runs/cpu-blas-continued01`.
