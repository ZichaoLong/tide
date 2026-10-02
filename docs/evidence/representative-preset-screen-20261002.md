# Representative five-preset screen and CANN profiles

Tested source: `0b1a5aa68061f37aef1f4387cf28abb0c77a464f` (clean immutable snapshot).
All three screening rounds and four independent profile jobs passed.
This qualifies one representative performance submatrix, not full-size F6.
[Audited raw samples, configurations, hashes and profiles](representative-preset-screen-20261002.json).

## Scope and method

- TimedDAG, standalone LibTorch, general online greedy prefill; Add and Attention, inference and complete AdamW training.
- 128 reachable body nodes/544 edges, D128/B8/T4/V257, clear=true; 8,995,632 Add or 17,384,240 Attention parameters.
- Five FP32 presets and resident FP16 separately. Every cell has three fresh processes; each uses one continued warmup step plus three measured steps, two continued windows per step (64 input tokens). There is no reference execution inside a candidate.
- CPU node workers and ATen intra-op threads are both one, inter-op/BLAS/OpenMP one. The aarch64 host exposes 320 CPUs; this is not a tuned multi-core CPU baseline. Every accelerator cell uses one leased logical Ascend910_9392 NPU, with LibTorch2.10/CANN9.0.0 and the qualified standalone SDK.
- Preset order rotates between repeats. Own heavy measurements run serially with a nonblocking timing lock. Cooperative device leases do not exclude unrelated host or device activity; no hardware-limit or statistical-significance claim.
- Step timers include input preparation/upload, online scheduling, graph, head/loss, necessary synchronization and, for training, backward/finite checks/optimizer. Construction, continued warmup and complete child-process wall time are also retained. Counter extraction and phase allocator sampling are outside step timers.
- Resident uses aggressive splitting, queue/arrivals4096, outputs128, trace8192, KVtrace65536, forward8GiB, retained8GiB, backward128GiB, optimizer4GiB, head256MiB. These local budgets are ceilings, not sums of HBM allocations. No admission-driven physical reduction was required.

## Steady measurements

Seconds per complete step: median of the three process medians; brackets give their minimum–maximum. Every raw step is retained in the JSON. A throughput ratio above1 means faster; it is CPU seconds divided by candidate seconds.

| Preset/dtype | Add inference | Add training | Attention inference | Attention training |
| --- | ---: | ---: | ---: | ---: |
| cpu/float32 | 0.230 [0.229–0.232] | 0.799 [0.789–0.800] | 0.797 [0.788–0.802] | 2.307 [2.286–2.399] |
| mixed-a/float32 | 1.274 [1.266–1.309] | 3.419 [3.332–3.465] | 3.472 [3.404–3.595] | 7.023 [6.817–7.384] |
| mixed-b/float32 | 1.377 [1.308–1.395] | 3.667 [3.498–3.879] | 3.727 [3.718–3.775] | 7.262 [7.262–7.380] |
| mixed-c/float32 | 1.584 [1.545–1.620] | 3.891 [3.791–3.963] | 3.880 [3.754–4.094] | 7.951 [7.554–7.967] |
| resident/float32 | 0.158 [0.158–0.159] | 0.364 [0.361–0.366] | 0.294 [0.293–0.297] | 1.050 [1.043–1.057] |
| resident/float16 | 0.159 [0.157–0.160] | 0.368 [0.366–0.369] | 0.277 [0.275–0.282] | 1.062 [1.045–1.063] |

Resident FP32 throughput relative to this CPU baseline is **1.462×/2.198×/2.709×/2.198×**, respectively. Mixed-a is the lowest-median mixed candidate in all four groups, while every mixed candidate is slower than this CPU baseline. Keep all presets configurable; this screening result does not select a universal configuration for other machines, scales, schedules or graph families.

FP32 losses agree with the independent CPU trajectory within1e-5, and measured candidate-event counts, output counts and final cut agree exactly in all three rounds. Resident executes26 device stages and36 Full chunks per measured step; CPU greedy reports20 stages with varying per-node batches. These stage definitions differ. Earlier full-observable/gradient/update gates remain the semantic qualification; scalar benchmark checks do not replace them.

## Cold process and finite-run costs

Each child process performs four real steps including warmup,256 input tokens. Its wall time also includes runtime startup and teardown, so it is broader than construction plus step timers. For this short lifetime most resident configurations are slower despite their better steady rates.

| Work | CPU process seconds | Resident FP32 process seconds | Resident/CPU throughput |
| --- | ---: | ---: | ---: |
| add-inference | 1.789 | 5.436 | 0.329× |
| add-training | 4.137 | 6.449 | 0.641× |
| attention-inference | 4.271 | 5.814 | 0.735× |
| attention-training | 10.535 | 9.789 | 1.076× |

Median construction/warmup seconds (CPU→resident FP32): Add inference0.551/0.197→1.051/0.281; Add training0.561/0.760→1.144/0.587; Attention inference0.986/0.589→1.479/0.448; Attention training0.985/2.124→1.625/1.198. Do not extrapolate a crossover lifetime from a single finite trajectory.

## FP16 and memory

FP16 has FP32 loss, adjoints and optimizer masters. It is about6.5% faster than resident FP32 for Attention inference here, with no observed gain in the other three cells. Low precision changes event counts and loss: maximum absolute loss differences from FP32 CPU are0.11189/0.27863/0.01166/0.11548. These are separate precision trajectories, not a same-work FP32 speedup or a large-model numerical-parity pass.

Peak allocated HBM in GiB, maximum across phases/repeats (excludes untracked vendor/driver allocation):

| Work | Resident FP32 | Resident FP16 |
| --- | ---: | ---: |
| add-inference | 0.128 | 0.092 |
| add-training | 0.779 | 0.733 |
| attention-inference | 0.427 | 0.249 |
| attention-training | 2.386 | 2.032 |

Every resident observed allocator peak stayed below its declared shape estimate. FP16 reduces inference storage more strongly than training storage, consistent with retained FP32 adjoints/masters. This calibration does not prove full-size admission.

## Separate representative profiles

Four msprof runs use the same Attention packet, dtype, physical capacities and schedule as their screening cells: construction, one continued warmup and one measured step (four windows total). Their losses and work counters match the accepted first measured screening step. Collection enables runtime APIs, task timing and AiCPU; exports and input CSV hashes are audited. Instrumented durations are excluded from the performance table.

| Profile | Device operators | Host kernel-launch API calls | Sync memcpy API calls | Async memcpy API calls | Device task sum, seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixed-infer | 248812 | 259552 | 27030 | 39046 | 0.606 |
| resident-infer | 34244 | 369 | 104 | 41 | 0.544 |
| mixed-train | 268252 | 271132 | 15873 | 26146 | 0.662 |
| resident-train | 80631 | 1632 | 263 | 260 | 1.815 |

No AiCPU operator appears in any of these four traces. Resident uses4 device-program execute calls for inference and8 for training; runtime device loops execute many kernels without a corresponding host launch for every kernel. The mixed path has many small vector tasks and host transfers. This supports host submission/transfer granularity as a major limitation of these mixed flows, rather than an AiCPU explanation.

Resident training actually has a higher summed device-task duration than mixed training while achieving a lower uninstrumented step time. Removing host stalls can outweigh additional bounded/padded device work. Task times may overlap; API levels nest; neither sum is end-to-end wall time. Construction is included in these traces, so these are whole-process counts, not counts per steady event. No inference about another shape or dtype is implied.

## Reproduction, retained records and remaining work

Generate the unchanged packets with `prepare_execution_flow.py --preset representative --memory add|attention`. Use the public `run_execution_flow.py` with `--implementation libtorch --family timed-dag --schedule prefill --threads 1 --steps 3 --warmup 1 --windows-per-step 2 --optimizer adamw`, the explicit preset/device/dtype and matching qualified binary. Training adds `--training`; resident uses the capacities above. Exact argv arrays and input text hashes remain in each raw cell. Run three processes serially with separate output directories.

Task-local artifacts are under `TASK/runs/representative-screen01`, `02`, `03`, and `representative-profile-{mixed,resident}-{infer,train}01`; `TASK=/mi/data2T/zlong/tide-execution-flows`. Status/exit/queue/log files all certify terminal success. The JSON links every accepted result and CSV through hashes. The intentionally suspended historical CPU task is unchanged.

Remaining F6 work includes other required families, languages and schedules; tuned CPU/mixed host execution; generic multi-card mixed placement and scalable resident queue/cache/retained storage; original full-size dimensions and matched comparisons. Full-size parameters remain9,468,053,696/17,521,117,376 at D2048/B512/T12/V50304. No reduction of logical batch, offline CPU-produced event schedule, new CUDA result or other-CANN qualification is claimed.
