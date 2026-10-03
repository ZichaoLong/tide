# First unprofiled original-B512 CPU measurement

The first formal original-size process passed on clean `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`,2026-10-04. LibTorch/TimedDAG/prefill/CPU/FP32 Add inference took **226.376559s per measured step**,or **54.281238 input tokens/s**. This is one independent process,not a three-process recommendation or a CPU/NPU comparison. [Audited result and exact command](formal-b512-first-cpu-20261004.json).

The workload retains D2048/B512/T12/V50304,480body nodes/2208edges and9,468,053,696 parameters. One continued warmup precedes one measured step,with two connected windows and12,288input tokens per step. PhysicalB32×16 preserves the logical batch and continuation. ATen16/workers1 and explicit BLAS/OpenMP startup16 are recorded;phase instrumentation,diagnostic exports and profiler are disabled. Own heavy jobs were serial;external shared-server load remains uncontrolled.

| Observation | Actual value |
| --- | ---: |
| Construction | 46.319800s |
| Continued warmup | 207.571919s |
| Measured step | 226.376559s |
| Outer consumer process | 481.412491s |
| Outputs per measured step | 12,288 |
| Final cut after warmup+measurement | 816 |
| Candidate events | 1,188,500 |
| Body candidate events | 1,163,924 |
| Selected events | 208,896 |
| Peak RSS growth | 42.769GiB |

All physical groups,output/cut boundaries and finite values passed. Observed RSS growth is below the114.559GiB estimate;CPU RSS is a process-peak proxy,not allocator accounting. Actual warmup/measured times satisfy the separately declared600s/update operating budget and original3000s threshold. The preceding248.392358/262.236922s phase forecasts remain forecasts;the table contains completed measurements.

Audit verified frozen source inventory,matching CPU core/consumer bytes,packet/plan/budget identities,complete result and terminal exit0/empty cgroup. No CPU reference execution supplied this candidate's routes or decisions. The accepted original-scale CPU/NPU near-tie strict failure remains separate;future NPU comparisons must report their own actual work counts.

Raw job:`TASK/runs/formal-first-cpu-blas01`;child:`assessment/cell-0-repeat-1`. Re-audit:`python TASK/launchers/audit_formal_first_cpu.py`. The remaining matrix,repeats,training and accelerator profiles are not certified by this first result.
