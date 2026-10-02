# Representative PDG complete-flow comparison

Fixed source `80dae6e14d41614d0cdb1056bb39b57ca10d07ed`, before the optional
training-diagnostics change. Both LibTorch schedules passed:40 screening and72
confirmation processes. [Audit and distributions](representative-pdg-matrix-20261002.json)
include source/binary/qualification/result/log hashes, all samples, memory peaks,
construction, warmup and outer-process times. These are representative results,
not original wide/full-size execution or measurements of later implementations.

The common rank-aligned v2 packets have128body nodes/544edges,D128/B8/T4/V257,
clear=true,8,995,632Add or17,384,240Attention parameters. Each candidate independently
consumes the same parameter/input rules and initial state. CPU reference execution
is outside candidate timing. No precomputed reference events/routes are replayed.

Five FP32 presets are screened with one continued warmup and one measured step.
Per workload, CPU, selected mixed and resident are confirmed in three fresh
processes, each with one continued warmup and three measured steps. A step spans
two windows and64input tokens; complete training includes loss, VJP, finite checks
and AdamW. CPU uses16native workers with packed sources/batch-next; mixed uses4.
ATen/BLAS threads remain1. Own heavy measurements are serial, on a shared server;
order rotates across confirmations. Queue120s, child900s and cell120s are finite.

The table reports median of the three process medians, in seconds per complete
step. Relative throughput is CPU time divided by resident time; values above1
mean resident is faster.

| Schedule | Memory/work | CPU | Selected mixed | Resident | Resident/CPU throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| prefill | add-inference | 0.132957 | mixed-b 1.226391 | 0.159859 | 0.832× |
| prefill | add-training | 0.548210 | mixed-b 2.824072 | 0.362354 | 1.513× |
| prefill | attention-inference | 0.250326 | mixed-a 2.405648 | 0.273881 | 0.914× |
| prefill | attention-training | 1.180975 | mixed-a 4.173503 | 1.018009 | 1.160× |
| streaming | add-inference | 0.139970 | mixed-a 1.119666 | 0.419918 | 0.333× |
| streaming | add-training | 0.629859 | mixed-b 3.214827 | 0.718168 | 0.877× |
| streaming | attention-inference | 0.262756 | mixed-c 2.713564 | 0.540818 | 0.486× |
| streaming | attention-training | 1.204491 | mixed-a 5.121948 | 1.408153 | 0.855× |

Resident prefill wins both training groups. CPU wins both prefill inference
comparisons and all four streaming comparisons. CPU also wins every short-process
comparison that includes construction/startup/exit. Mixed paths are slower in this
scope; the one-step pilot selects a candidate but does not establish a global
optimum, and some mixed process medians vary substantially.

Observed event counts, output counts and final cuts match exactly; losses differ
by less than1e-5 from the independent matched reference. Resident prefill uses26
stages per step, versus80for streaming, with the same semantic work. This supports
batching as a useful mechanism, but stage count alone is not a kernel-time profile.
Prior TimedDAG profiling identified host submission/synchronization costs and no
observed AiCPU; it must not be relabeled as a fresh PDG profile.

Raw records: `TASK/runs/matrix-pdg-libtorch-{prefill,streaming}-{screen01,confirm01,confirm02,confirm03}`,
where `TASK=/mi/data2T/zlong/tide-execution-flows`. Family-specific stress topologies,
original wide size, FP16 matrix comparisons, CUDA and other CANN versions remain
outside this evidence. Other family/client/schedule cells are tracked by F6.
