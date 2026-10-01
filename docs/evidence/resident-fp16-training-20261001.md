# Public resident FP16 complete training

Implementation: `00950483711a835027b9d570b08042f141dd1838`.
All seven immutable-source tasks passed with exit0. [Machine-readable audit](resident-fp16-training-20261001.json)
authenticates source, runtime-specific builds, reused archives, four rebuilt
public objects per runtime, loaders, device kernels, logs and trace CSVs.

## Behavior and independent checks

The single-NPU public C++ owner and Python client accept actual FP16 payloads
with FP32 cotangents, accumulation, optimizer masters and slots. Named checkpoint
parameters remain FP16; restore requires exact correspondence with rounded
masters. Sub-half updates survive publication and resume. FP16 roots are
explicitly refused; there is no hidden loss scale. See [contract](../resident-training.md).

| Fixed-source check | Completed scope |
| --- | --- |
| Basic/extended half training | 43 trajectories,688 windows,172 updates |
| Event/fiber/mixed half training | 40 trajectories,640 windows,160 updates |
| FP32 control training regression | 98 trajectories,1568 windows,392 updates |
| FP32 event regression | 66 root cases,8 trajectories |
| FP32 fiber regression | 172 root cases,20 trajectories |
| Python client | 138 tests passed,no skips |

Each half trajectory performs four updates over four retained windows per
update:nonzero,connected-zero,disconnected,None-to-nonzero continuation.
Independent CPU schedules keep actual half forward rounding with FP32/FP64
adjoints; a separate CPU FP32 NamedOptimizer updates its own masters. No oracle
output,route or gradient feeds the NPU candidate. VJP tolerances remain
rtol2e-3/atol2e-5; diagnostic roots are explicitly scaled by256,not implicitly
unscaled by the runtime. FP32 regression tolerances are unchanged.

Coverage includes SGD/AdamW,master low bits,aliases,retained state and cache
roots,feedback,normalized Aggregate,LH/SwiGLU,GQA,fiber pools,mixed profiles,
HARD/HST/SOFTP,both schedules,checkpoint continuation with schedule changes,
wrong dtype/master mismatch refusal and copy-isolated exports. Nonfinite roots
refuse an update without changing optimizer state or generation. Inactive cache
padding is poisoned to verify its exclusion.

Python adds36 lifecycle cells across three families,two schedules,two optimizers
and three state profiles;three fresh-process checkpoint suffixes;two independent
CPU scalar autograd/master trajectories;97 affected prior precision/event/fiber
cases. Python resident remains a client of C++/CANN,not an independent PyTorch
device scheduler. These fixtures validate training mechanisms,not convergence.

## Build and device placement

Local aarch64 Ascend910_9392,CANN9.0,Torch/TorchNPU2.10;standalone SDK and
Python-owned runtimes are isolated. Only four affected public C++ objects and
training checkers were rebuilt. Unchanged core,CANN kernels and fixture/support
archive members were byte-verified against successful terminal parent builds.
This is an affected-path qualification,not a from-scratch vendor build or a
repeat of the unchanged8,954-check CPU gate.

A separate half mixed-cache SOFTP AdamW profile completed one trajectory,
16 windows and4 updates. Exported trace counts:35,415 AI_VECTOR_CORE,
2,076 AI_CORE,298 MIX_AIV. No AiCPU engine or logged CPU fallback was observed.
The trace includes construction,CPU assertions and exports; it establishes
observed placement only,not full-size throughput or absence of host boundary work.

## Reproduction and retained failures

The frozen snapshot is `TASK/sources/low-precision-training-clean01`,where
`TASK=/mi/data2T/zlong/tide-execution-flows`. Both builds and all five device jobs
use the implementation revision above. Job names,commands,terminal status and
raw-log hashes are recorded in the audit. Queue waits were bounded at120s,
builds900s,runs600s;profile raw storage512MiB. Final audit:

```bash
python "$TASK/launchers/precision_training_evidence.py" 00950483711a835027b9d570b08042f141dd1838
```

Original development failures remain:dev01 builder omitted a checker object;
Python dev02 omitted the fresh process's explicit native library;dev03/dev04
used unsupported ordinary identity memory/Full in the scalar fixture. The final
fixture uses the supported public identity boundary. These were build/test
corrections,no tolerance relaxation. No failed build's production artifact was
used. Seven qualification jobs are terminal;no peer or performance-matrix claim
follows. Cross-card progression,consumer scale,full-size comparisons and
additional target environments remain under ROADMAP F1–F7.
