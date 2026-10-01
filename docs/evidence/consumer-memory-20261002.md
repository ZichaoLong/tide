# Complete-consumer phase memory

Source `fdfc74802934cf145c052eb3ae966ff94c323e0a`, 2026-10-02.
All five immutable-source jobs passed; [audit record](consumer-memory-20261002.json).
CPU and installed standalone NPU consumers build/link successfully. The NPU
client reuses source/header/options-verified objects and has a fresh link/loader
check; the byte-identical qualified owner-stream backend/core is not rebuilt.

CPU four cases and NPU seven cases passed without skips. They exercise complete
inference/training through pure Python, the native Python client and standalone
LibTorch, mixed A/B/C, one/two-device resident, and resident FP32/FP16. Independent
CPU losses, output counts and continuation cuts agree within existing dtype
tolerances. A released 16MiB temporary remains visible in peak allocation, and
resetting the phase clears that historical peak. Earlier full semantic gates
remain the evidence for schedules, routing and gradients.

`memory` records initial, construction, optional warmup and measured phases for
every logical accelerator. Counters are sampled outside step timers; initial
setup is charged to construction. CPU peak RSS is process-lifetime, not a
resettable per-phase counter. No event-loop instrumentation is introduced.

A bounded actual Attention consumer used 128 reachable body nodes, 544 body
edges, D64/B2/T2/V257, 4,367,024 learned parameters, two NPUs, online prefill,
FP32, AdamW, four continued windows and two updates (one warmup, one measured).
Diagnostics were disabled. The complete graph, embedding/head, loss, VJP and
optimizer ran independently of a CPU execution.

| Phase | npu:0 peak allocated | npu:1 peak allocated | Peak reserved (0 / 1) |
| --- | ---: | ---: | ---: |
| Construction | 139.67 MiB | 117.99 MiB | 144 / 126 MiB |
| Warmup training | 388.25 MiB | 374.19 MiB | 406 / 412 MiB |
| Measured training | 389.15 MiB | 374.37 MiB | 440 / 412 MiB |

These are process allocator observations, excluding untracked CANN/driver HBM.
The calibration has one measured sample and is not a throughput recommendation
or a full-size result. CUDA counter source has not been compiled/executed on a
CUDA stack. This increment does not implement total per-device memory admission.

Retained failures: the new test initially collided with the global `dtype`
fixture, then a Python test incorrectly supplied a native-library option. Both
were corrected without production or tolerance changes. The first larger
calibration was explicitly refused by reverse-packet budgeting; trace1024 then
exposed a separate cache-adjoint nested budget refusal. Neither was OOM. Reviewing
the subdivisions led to a 32GiB backward envelope, with unchanged cache capacity;
the successful allocator peaks above show why local envelopes must not be summed
as actual HBM. Total planning remains the next scale-enablement task.

Artifacts use `consumer-memory-clean01`, builds `consumer-memory-{cpu,npu}-clean01`,
and jobs `build-consumer-memory-{cpu,npu}-clean01`,
`consumer-memory-{cpu,npu,calibration}-clean01` under the existing execution-flows
artifact root. Their manifests, exact commands, input identity, raw results,
queue mappings, loader/binary hashes and failures are audited in the JSON.
