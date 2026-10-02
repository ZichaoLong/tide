# Physical sample chunks in complete eager consumers

Tested source `e6cc52b07d33e5be33054f43bc938dbe1f46a0da`, clean immutable
snapshot. Both consumer builds, CPU60, NPU38 and the separate four-process memory
observation passed; no skipped tests. [Audit and raw-record hashes](consumer-sample-chunks-20261002.json).

## Execution contract

`--sample-chunk-rows N` limits samples in one physical group for Python, native
adapter and standalone LibTorch eager CPU/mixed consumers. Zero keeps the entire
logical batch. Groups share one parameter generation and optimizer. Each group
executes all requested connected windows before backward, then releases that
autograd graph. Parameter gradients accumulate over all groups, preserving None
and connected-zero semantics. A single finite-gradient agreement and one optimizer
update follow the whole logical batch. Per-group state/history/pending/KV carries
into the next update. Input and target formulas use original global sample IDs;
loss normalization and reported token counts use the original logical batch.

This is a general sample-independence transformation of the declared graph
semantics, not an event replay or a topology/input-specific schedule. Every
candidate independently executes its online schedule. Sample iteration is a
physical execution boundary; no new per-message host loop is added to graph
scheduling. The final physical group can be shorter, without padding or dropping
samples. User-visible `batch_execution` records requested/effective rows and group
count. Split diagnostic windows declare their sample range and global IDs, so
partial continuations are not mistaken for a whole-batch result.

Persistent state for all samples, parameters and optimizer state still remain
live. The setting is an explicit maximum, not yet automatic eager admission.
The resident consumer explicitly rejects a nonzero request until its corresponding
state-switching and VJP accumulation path exists. Mixed execution still has one
payload device. No core, backend library, checkpoint ABI or CANN kernel changed.

## Affected validation

The new directed matrix covers Python/native/LibTorch; all three graph families;
streaming/prefill; Add/Attention; clear/carry and delayed arrivals; SGD/AdamW;
continued inference and complete training. Logical B5/physical2 includes the
one-sample tail. CPU checks cover FP32 and FP64; mixed A/B/C checks cover FP32.
Each partial diagnostic window is compared against the same sample range of an
independent unsplit CPU streaming reference without sorting away stable-order
errors. All gradient entries (including None), parameters after multiple updates,
full continuation fields, loss, outputs and cuts are checked.

CPU60 comprises48 directed sample cases, two shared CLI cases, one refusal/whole
batch case and nine existing host-control regressions. NPU38 comprises24 mixed
sample cases, eight host-control regressions, two resident CLI/failure cases and
four unchanged resident Attention training regressions (FP32/FP16, two clients).
Development passed the same affected scope. Qualification uses byte-verified
unchanged core/resident libraries, exact source/header/compiler-option checks for
consumer object reuse, fresh links and loader checks. The unchanged full CPU and
CANN suites were not rerun. CUDA and other environment versions remain untested.

## Separate memory observation

Four fresh standalone processes use one TimedDAG/prefill Attention packet:
128 body nodes/544 edges, D128/B16/T8/V257, 17,384,240 parameters, clear=true,
FP32, two continued windows, one complete AdamW step, no warmup. Every run consumes
256 logical input tokens and preserves the same output count/cut/event count;
loss differences stay below1e-5. CPU uses16 native workers, mixed A uses4; both
use packed sources/batch-next and ATen/BLAS1. One NPU is leased for mixed A.

| Flow | Physical samples | Process peak RSS, bytes | NPU peak allocated, bytes |
| --- | ---: | ---: | ---: |
| cpu | 16 | 1,043,226,624 | 0 |
| cpu | 4 | 632,844,288 | 0 |
| mixed-a | 16 | 2,135,420,928 | 292,252,160 |
| mixed-a | 4 | 1,748,725,760 | 292,252,160 |

CPU process peak RSS decreases39.3%; mixed A host RSS decreases18.1%. The mixed
NPU peak is unchanged, so this case shows a host activation-lifetime saving,
not a reduction of its fixed device storage. RSS includes the C++ process's
libraries and allocator behavior; NPU counters cover the Torch allocator, not
all vendor/driver allocations. These observations do not prove a memory upper
bound for arbitrary shapes or modules.

Each configuration has only one process; no throughput recommendation is made.
This is a medium capacity observation, not the original D2048/B512 wide workload.
F6 still needs full-size execution, mixed multi-device placement and resident
sample/state storage work. Logs and receipts remain under
`TASK/runs/sample-chunks-{cpu,npu,memory}-clean01`, with
`TASK=/mi/data2T/zlong/tide-execution-flows`. Memory cells have180s bounds and the
parent800s; failure stops the recipe rather than triggering repeated retries.
