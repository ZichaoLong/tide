# Configurable Read/control placement qualification

Implementation `a12ee0c676faa80096db4034595b581db11a5f22` was built and tested from clean frozen `perf-a3`.
The standalone consumer links the previously qualified installed core; no public
core or Python-package implementation changed. See [contract](../accelerator-scale.md)
and [machine-readable evidence](accelerator-scoring-20260928.json).

## Verified configurations

| Gates | Cells | Scope |
| --- | ---: | --- |
| CPU historical FP64 | 8 | Add/Attention, resident/host, memory/locality |
| CPU FP32 against historical FP64 | 8 | Same combinations, explicit precision comparison |
| Two-NPU historical FP64 Read | 8 | Same combinations, default unchanged |
| Two-NPU FP32 Read and controls on model devices | 4 | Add/Attention, resident, memory/locality |
| Mixed placement and real topology | 6 | Two tiny mixed Read/control placements;465-node Add/Attention,D8/B1,seeds0/7 |

All34 cells passed both no-grad and autograd comparisons of complete state,
traces,routes,history,pending messages,logits,owners and residency. Isolated
nonzero/zero VJPs now include descriptor and control roots. Tiny cases retain
strict rtol1e-5/atol1e-6; four real-topology cells explicitly use the previously
specified basis-conditioned policy. CPU FP32 and one mixed NPU case explicitly
compare with historical FP64. Discrete routes remain exact; passing these cases
is not a promise that arbitrary near-tie decisions are precision-invariant.

Both standalone builds passed the analytic L2-norm value/gradient/zero-vector
and configuration-rejection CTest,plus existing VJP negative tests. Every
qualification process exited normally. Known root-owned-file and repeated
finalization warnings remain; no unsupported-double diagnostic occurred in the
final parity logs.

## Device execution and limits

Tested aarch64 Ascend910_9392,TorchNPU2.10.0,CANN9.0.0,driver25.3.rc1,
TASK_QUEUE_ENABLE=0. The immutable two-device msprof capture observed
41 FP32 LpNormV2 tasks on MIX_AIV and 349 SoftmaxV2 tasks on
AI_VECTOR_CORE. Both physical devices executed these operators. Softmax counts
include model/attention work; they are not solely region-control counts.
This is operator-placement evidence,not a throughput result or proof that CPU
coordination was eliminated. Node ranking extracts scalars to CPU; count
histories and event scheduling remain CPU. The new mode is not yet qualified
on eight devices,other SDK/driver tuples,or CUDA. No full backward/optimizer
performance claim is made.

The accepted follow-up treats NPU node selection and event scheduling as separate
candidates,each requiring independent correctness gates and inference/training
measurements before choosing an implementation.

## Retained failures and full-size work

Development dev09 failed for a missing C++ header. Dev10 passed same-precision
checks but its explicit mixed FP64 comparison tried to convert a descriptor on
NPU. Dev11 fixed only the comparison view by moving it to CPU before conversion;
old failures remain failed. The final immutable checks passed.

Earlier bb0ecc6 full-size pilots retained FP32 payload and CPU FP64 Read:

| Workload | Devices | Mode | Mean ms/sample-token |
| --- | ---: | --- | ---: |
| Add9,468,020,899 parameters | 4 | no-grad | 16.2893 |
| Attention17,269,426,339 parameters | 4 | no-grad | 56.0128 |
| Add9,468,020,899 parameters | 2 | grad-forward | 48.0210 |

Each passed12 growing-context tokens,4 warmup+8 measured,exit0 and record
validation. These are individual pilots with different modes/resources,not
repeated scaling comparisons. Two independent multi-device pilots did overlap
during construction. One four-device run failed during placement and another
was cancelled after other processes entered allocated devices; no clean capacity
conclusion follows. Raw contention/terminal records are retained under the task's
runs/parallel-pilots-a2c-{results,device-contention}.json.

Current code initializes every acquired NPU before CPU weight construction,
making all reserved contexts visible. Advisory queue locks cannot prevent an
unrelated process from using those devices. Concurrent tests must still record
shared-resource interference.

Four new full-size pilots,Add/Attention×no-grad/grad-forward,were submitted from
this qualified source using FP32 model-device Read/controls,two NPUs per case,
resident/locality,D2048/B512/V50304,seed7,12 tokens,1800-second native limit and
512GiB RSS cap. Their results are pending. Exact commands/budgets/prerequisites
are in task-root/runs/scoring-pilots-a3.json; unit/run names are
pilot-{add,attention}-g{0,1}-n2-a3. See current STATUS for actual allocation.
