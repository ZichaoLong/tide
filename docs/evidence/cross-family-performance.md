# Cross-family streaming and prefill performance assessment

This is the terminal E4 assessment for the authorized cross-family extension.
It tests clean source `eaa15c6f1d58291d818f8872ee20987abbb3d63b` from
`/var/tmp/zlong-graph-execution-foundation/qualification/cross-family-20260925`
with driver SHA256
`8b8cbedf3e976c4f5745d0d339e34c43faf9d13589181e3076e3bb9155bbb3ee`.
The durable unit was `tide-cross-family-performance-20260925-a` in
`background.slice`, Nice 10. Its wrapper exited 0 and is inactive/dead; the
individual failed run below remains failed.

The CPU environment is aarch64, Torch/LibTorch 2.10.0+cpu, FP32, explicit CPU,
ATen/OpenMP/BLAS1, with one inter-op thread. Resource discovery recorded 320
effective CPUs and a combined budget of 160 CPUs; effective memory was about
1.60 TiB and the task budget about 0.80 TiB. Medium used four workers and one
thread with two reset warmups and a 180-second per-run bound. Large used 32
workers and four threads with one repeat and a 900-second bound including setup,
warmups, measurement and profiling. No formal timing overlapped qualification.

Raw records are under
`artifacts/cross-family-performance-20260925-a/`: `assessment.json`, the
`medium/` and `large/` suites, their summaries, per-run process audits and local
Trackio-compatible records. Trackio was best-effort and degraded because the
workload environment has no `trackio` module; the project-owned records are the
authoritative evidence. This does not invalidate the runs.

The medium tier covered all 60 logical v2 variants with three independent
processes each: **180/180 completed, 0 failed, 0 bounded stops**. Every one of
the 120 summary rows (including phase rows for transition workloads) has three
completed repeats. The records include separate prefill and streaming timings
where the workload permits them, and separate grad-forward, backward, optimizer
and full train-step timings for TR01/TR02. Examples of whole-workload medians
from the optimized variants are:

| Workload / variant | Whole or train-step median (s) | Separate observations |
| --- | ---: | --- |
| T01 native-frontier-optimized | 0.30087 | prefill/streaming paths are recorded; native max full batch 896 |
| S01 native-settle-optimized | 0.78911 | prefill 0.61688 s, streaming 0.17223 s |
| S02 native-settle-optimized | 0.55965 | legal whole-sequence path; native max full batch 1024 |
| TR01 native-frontier-optimized | 2.99926 train-step | grad-forward 0.61873 s, backward 2.32516 s, optimizer 0.02058 s |
| TR02 native-settle-optimized | 13.15938 train-step | grad-forward 4.08112 s, backward 8.92881 s, optimizer 0.04494 s |

The full three-repeat table, dispersion and all phase rows are in
`artifacts/cross-family-performance-20260925-a/medium-summary.md` and
`medium-summary.json`. These are finite observations, not automatic speedup or
default recommendations. Python frontier/Settle records expose causal fallback
counters where the contract requires them; native optimized records expose
actual sequence, batch and replay counters. For example, T01 native frontier
records four prefill state-sequence calls with max full batch 896, while the
Python frontier path records its declared state-prefill fallback events.

The large tier launched 12 fixed stages: two graph families for each of the
wide-add, wide-attention and narrow presets, each at a resource stage and its
parameter-count target. Eleven completed; one target timed out. Only one repeat
was budgeted, so these values have no variance or speedup claim.

| Preset / graph / nodes | Parameters | Status | Whole forward (s) | Prefill / streaming (s) | Peak combined RSS (GiB) |
| --- | ---: | --- | ---: | --- | ---: |
| wide-add / TimedDAG / 128 | 3,758,889,152 | completed | 16.31290 | 8.83063 / 7.48228 | 25.697 |
| wide-add / TimedDAG / 320 | 9,397,213,568 | completed | 16.15222 | 8.41092 / 7.74130 | 49.393 |
| wide-add / Settle / 128 | 3,758,889,152 | completed | 15.65862 | 8.96419 / 6.69443 | 25.413 |
| wide-add / Settle / 320 | 9,397,213,568 | completed | 16.07247 | 9.20541 / 6.86707 | 50.002 |
| wide-attention / TimedDAG / 128 | 5,907,421,376 | completed | 38.70207 | 14.96877 / 23.73330 | 63.907 |
| wide-attention / TimedDAG / 384 | 17,722,251,712 | completed | 37.65473 | 14.50582 / 23.14891 | 114.427 |
| wide-attention / Settle / 128 | 5,907,421,376 | completed | 37.75817 | 14.79281 / 22.96536 | 65.454 |
| wide-attention / Settle / 384 | 17,722,251,712 | completed | 35.42773 | 13.62266 / 21.80507 | 114.601 |
| narrow / TimedDAG / 256 | 46,391,680 | completed | 43.75946 | 18.75525 / 25.00421 | 6.960 |
| narrow / TimedDAG / 46,912 | 8,496,773,056 | completed | 54.11131 | 20.27014 / 33.84117 | 51.862 |
| narrow / Settle / 256 | 46,391,680 | completed | 41.41152 | 16.36882 / 25.04270 | 6.966 |
| narrow / Settle / 46,912 | 8,496,773,056 | **failed: bounded timeout** | unmeasured | no measured phases | 51.829 |

The final narrow Settle target constructed the model and then exceeded the
900-second bound (`901.732593` seconds, worker exit `-15`). Its record has no
timing metrics, and the process audit reports no unreaped child. The assessment
wrapper's exit 0 means the bounded assessment reached its terminal state; it
does not convert this individual failure into a pass. The target TimedDAG run
did complete at 8.497B parameters, while target-scale Settle remains unverified
by this fixed budget.

The measured records also show the actual path boundary. Large TimedDAG
prefill uses a causal frontier and then native streaming positions; Settle uses
its native encoded frontend. Native counters record state sequence calls, full
batch sizes, packed source batches, region waves and zero semantic replay for
the no-grad paths. Training medium records retain semantic State/Read replay as
part of the declared correctness contract. No conclusion is drawn from a
variant name or a requested option alone.

The assessment is therefore closed with all planned measurements terminal,
the one resource-limited target retained as a failure, and no blanket claim of
large-scale acceleration, arbitrary-model support or target success for every
graph family.

## Extended-timeout follow-up

The original failed record above is retained unchanged. A separately
authorized follow-up reran only the narrow Settle target from the same clean
source and matching build, in a new output directory, with the durable user
service `tide-cross-family-timeout-20260925-b` in `background.slice`, Nice 10.
The command used the same 32 workers and four Torch threads, but increased the
per-run bound from 900 to 1800 seconds. The service reached a terminal state
and was collected; its persistent output is
`artifacts/cross-family-performance-20260925-b/`.

Both staged runs completed: the 256-node resource stage took 240.704912 s and
the 46,912-node target took 1034.865380 s. The target constructed
8,496,773,056 parameters, completed prefill in 70.643869 s and streaming in
59.924672 s, and reported a whole no-grad forward time of 130.568541 s.
Peak combined RSS was 55,644,233,728 bytes (about 51.82 GiB); no child was
left unreaped. The individual follow-up records are
`large/narrow-settle-n256-native-settle-optimized-r0/summary.json` and
`large/narrow-settle-n46912-native-settle-optimized-r0/summary.json`.

This establishes that the earlier 900-second boundary was insufficient for
this exact CPU run. It does not change the original failure record, or imply
that a longer bound alone is a general performance or NPU qualification.
