# Complete NPU resident qualification — 2026-10-11

The exact composite accepts **133 registered component/dtype invocations**,the
installed standalone consumer,and **547 public cases with zero skips**. The
[audit](resident-composite-20261011.json) verifies every component marker/log hash,
unique inventory coverage,public JUnit,source/build identities and terminal
service/guard closure. This is correctness evidence,not new performance evidence.

| Accepted part | Source | Invocations |
| --- | --- | ---: |
| Original resident10 prefix | 9c3ffd3acad1a6be31f37e349e6505a14ffc4e1b | 33 |
| Corrected State/Read checker | e3c86d70795c594eafbc587388d9e984c3f92612 | 2 |
| Exact unstarted component completion | 9c3ffd3acad1a6be31f37e349e6505a14ffc4e1b | 98 |

The original resident10 remains failed at its State/Read repeat-bound assertion.
[The test-only stream-boundary repair](precision-and-stream-gates-20261011.md)
was separately qualified;the original failure was not deterministically
reproduced. No accepted prefix case was repeated by the completion task.
The resident production libraries remain byte-identical to their a926703 build;
only the checker was recompiled. The core digest is
`0e2d0baa8fb2e3db1aed807c20c4ef78e53838922295dd4f60fb7578313266f0`.

Component and public coverage includes PDG positive-delay feedback,TimedDAG,
SettleGraph,streaming/prefill,FP32/FP16,HARD/HST/SOFTP,State/Read/Full/Aggregate,
event/fiber Attention,VJPs,SGD/AdamW,continued windows,checkpoint,capacity and
1/2/3-owner placement. It includes sharded state/Full/projection,gradient
accumulation,compact contexts/journals and independent CPU references.
Candidate events,routes and VJPs are computed independently.

Full training explicitly selected its existing score-conditioned softmax policy;
all discrete decisions and other tensor/VJP/update policies remain unchanged.
The original strict width257 Full and near-tie trajectory witnesses remain.
Finite qualification does not promise strict trajectory equality for all
cross-device numerically sensitive inputs.

`tide-g4-npu-resident-completion01.service` used physical3,7,9→logical0,1,2
under occupied-card authorization. New completion budget:8CPU,24GiB,5400s
workload/6000s service. Observed guard elapsed 2998.92s,
peak summed RSS 10962575360B. All constituent accepted tasks
are terminal with zero exits,empty cgroups and no guarded survivors. Summed RSS
counts shared mappings multiple times;it is not a kernel allocation guarantee.

The separate clean00b1775 CUDA scale-consumer follow-up freshly compiled/linked
and passed all5 CPU CTests after the control-precision fix. Its manifest and
terminal guard are included in the audit. The first site launch failed before
build due to a wrong core-path assumption;that failed receipt is preserved.
**Actual GPU correctness,residency,multicard,memory and performance remain pending.**

The next bounded six-component/four-complete-profile tranche is separate. No new
speedup,unprofiled throughput or complete execution timeline is claimed here.
