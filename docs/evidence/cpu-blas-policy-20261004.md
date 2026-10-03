# CPU BLAS startup policy diagnosis

Clean source `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`, 2026-10-04. The installed OpenBLAS remained at one thread after ATen was set to16. Explicit BLAS/OpenMP startup16 produced **3.432× descriptive throughput** in a bounded complete-training diagnostic with three fresh processes per policy. This is reduced-topology evidence, not an original-size speedup. [Audited raw samples](cpu-blas-policy-20261004.json).

The original CPU consumer loads the same Torch2.10/OpenBLAS library as the isolated probe. OpenBLAS0.3.30 reports `USE_OPENMP`; `torch.set_num_threads(16)` changed ATen but left `openblas_get_num_threads()` at1. A read-only one-second sample of the old CPU worker observed about1.45CPU-core equivalents. This establishes a separately controlled BLAS pool, not the time distribution of the complete old run.

Both tested policies keep ATen16/inter-op1 and graph workers1. Fresh processes use `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1` or16, `MKL_NUM_THREADS=1`, and `OMP_THREAD_LIMIT=32`. The shape probe checks actual BLAS thread counts. No setting was changed inside the old running process.

| FP32 forward/backward matrix shape | BLAS1 median seconds | BLAS16 median seconds | Ratio |
| --- | ---: | ---: | ---: |
| 32×2048×2048 | 0.016934 | 0.002052 | 8.250× |
| 32×2048×6144 | 0.052291 | 0.005674 | 9.216× |
| 384×2048×50304 | 4.048926 | 0.299997 | 13.497× |

Each shape uses two warmups and five samples, without a graph or optimizer. Those operator ratios cannot be substituted for end-to-end throughput.

The complete-graph check uses a four-body-node Attention graph,84,946,950 parameters,D2048/B8/T4/V257,physicalB4×2,TimedDAG/LibTorch/CPU/prefill/FP32. Each process performs one continued warmup plus three measured complete SGD updates,two connected windows each. Policy order alternates across repeats. All six processes passed finite checks,capacity calibration and continuation.

| BLAS threads | Three process medians,seconds/update |
| --- | --- |
| 1 | 6.178496, 6.737636, 6.178125 |
| 16 | 1.800303, 1.752266, 1.804052 |

All three updated losses match exactly across the six processes:32.4087181091,26.1608428955,23.7334518433. All reported work counters match,including384candidate events and64outputs per measured update. This is a complete-run consistency observation;it does not replace full state/route/gradient semantic qualification. Other explicitly non-formal CPU/profile jobs overlapped,so the3.432× ratio remains a diagnostic result.

After these two finite diagnostic rounds,the supplemental `wide-eager-cpu-attention-b512-extended01` was deliberately cancelled to free its memory for required CPU calibration with the new startup policy. It has **no complete B512 result** and is neither a passed timing nor a failed mathematical update. Its receipt is cancelled/exit143,the cgroup is empty,and its partial assessment plus cancellation reason and hashes remain retained. The old3000s refusal is unchanged. The separately protected historical stopped CPU process was untouched.

Original-width continuous cost/memory calibration now uses an explicit BLAS16 startup while preserving ATen16/workers1,model/dtype,capacity margins and geometry. Actual originalB512 performance still requires execution. There is no broader worker sweep or claim that this setting is best for every model,matrix shape,CPU architecture or BLAS implementation.

Re-audit: `python TASK/launchers/audit_cpu_blas.py`. Raw jobs: `cpu-blas-scope-probe01`, `cpu-blas-graph-probe01`. Both are terminal/exit0 with empty cgroups. The source-matching CPU binary is unchanged;this evidence changes the declared experiment environment,not the graph implementation.
