# Existing NPU dispatch profile interpretation

This is a new analysis of the retained b4f26b3 FP32 trace, not a new performance
experiment. The two-device CANN9.0/TorchNPU2.10 launch used Attention D32/B4/T3,
FP32 payload and model-device Read/controls/ranking/event keys. The entire tiny
process includes initialization/warmup and is not the D2048/B512 full-size workload.
See the adjacent JSON for exact CSV hashes and aggregation.

| Device engine | Operator tasks | Summed task time (us) | Fraction of task sum |
| --- | ---: | ---: | ---: |
| AiVectorCore | 6247 | 17538.98 | 76.19% |
| Mixed AiVector | 300 | 972.92 | 4.23% |
| AiCPU | 55 | 3365.90 | 14.62% |
| AiCore | 410 | 1142.26 | 4.96% |

All 55 AiCPU tasks are int64 Sort. FP32 Sort, LpNormV2, SoftmaxV2, int64
ReduceMin, NonZero and Gather use AiCore/vector variants in this trace. AiCPU
is a processor on the accelerator, not the host CPU. This is evidence of partial
AiCPU execution, not evidence that most model arithmetic uses AiCPU.

Task sums can overlap across devices/streams and are not end-to-end fractions.
The host API table also contains nested levels and must not be added into a wall
time total. It reports 7012 launches, 2031 SetDevice calls, tiling/loading and
synchronization/copy activity; cold setup contributes. These observations plus
the source explain plausible overheads but do not rank full-size bottlenecks.

The current scheduler keeps histories, payload handles and C++ dispatch on the
host. Ranking uploads int64 histories and returns selected indices; event keys
are processed on device, then minimum time/indices return to the host. Descriptor
finiteness checks, scalar extraction, phase barriers, many small operators and
Read VJP replay remain in the timed execution. Resident message/state payloads
remain on assigned NPUs and use device-to-device copies across shards.

Existing full-size screens (see accelerator-performance-20260928.md) found Add2
17.110 versus34.933 ms/sample-token and Attention4 56.671 versus67.420 for CPU64
Read/CPU scheduling versus all32 model-device control. Those labels do not put
the model on CPU. Screens had shared-host contention and are not causal proofs
of which overhead dominates. Three matched CPU64/CPU32 pairs did not establish a
consistent precision advantage. Training choices and cold/warm timings remain
separately documented there.

CANN `msprof --task-time=on --runtime-api=on --aicpu=on BINARY ...` is usable with
the standalone C++ client and already produced retained traces on four CANN
runtimes. `scripts/profile_accelerator.py` uses `torch_npu.profiler` for Python
and native-adapter forward/backward/update, with Chrome traces and hardware event
checks. Its new `--dtype` selects FP32/FP16. MindStudio/compatible trace viewers
can inspect timelines; a dashboard is not needed to retain/analyze the evidence.
`scripts/summarize_ascend_profile.py --csv-dir CSV_DIR --output-dir NEW` reproduces
this engine/API aggregation without summing nested APIs into a wall-time claim.

The FP16 profiler entry point was additionally exercised on Python and native
mixed-TimedDAG forward/backward/AdamW at cb58110 (fixture and master copies ordered
before the nondefault worker stream). Both succeeded on TorchNPU2.10/CANN9.0:

| Path | Hardware kernel events | AiVector | Mixed AiVector | AiCore | Host scalar extractions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python | 1644 | 1214 | 266 | 164 | 242 |
| Native adapter | 1638 | 1222 | 252 | 164 | 211 |

These two tiny FP16 traces report no AiCPU kernel type and no named CPU-fallback
event. They use the public Python/native scheduler, a different path from the
older standalone all-device dispatch trace. Their absence of AiCPU does not
show that changing dtype removes int64 Sort from AiCPU. Host scalar extractions
and copies remain present. This does not prove
absence of all host work or certify full-size bottlenecks. Exact commands,
runtime manifests, result/trace hashes and retained artifact paths are in the JSON.
The analysis tool itself is committed at fc2a76f; it only reads exported CSVs.
