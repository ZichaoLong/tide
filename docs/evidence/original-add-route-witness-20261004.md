# Original Add: a strict route mismatch at a floating near tie

**The strict discrete comparison failed.** A bounded diagnostic located a
specific CPU/resident disagreement;it did not turn the earlier full-size
performance runs into equivalence passes.
[Audited witness](original-add-route-witness-20261004.json).

The two public sessions independently execute the original9.468B Add model and
inputs,at matched physicalB2,two connected windows and FP32. They use the clean
historical`26176de888013fda5eccfe039fa504e87c2e7e95` public package,CPU and three
NPUs. Comparison happens after both finish each window;no reference events or
numerical results drive the resident candidate. The first17 windows have equal
event identities/active flags;window18 first differs at sample17,time280,region7.

| Candidate | CPU FP32 score | Resident FP32 score | CPU selected | Resident selected |
| --- | ---: | ---: | --- | --- |
| Node245 | 4.005112171173096 | 4.005106449127197 | No | Yes |
| Node246 | 4.005112648010254 | 4.005106449127197 | Yes | No |

Both nodes have zero prior selections. CPU scores differ by one FP32 ULP
(`2^-21`). The resident scores tie,so its specified stable node-ID ordering
selects245. Offline FP64 norms of the exported resident FP32 proposals are
4.005106345184017 and4.005106571542164;**both correctly round to the same FP32
score**. Thus a more accurate norm reduction on these same payloads would not
restore the ordering after the required scalar FP32 rounding. FP64 diagnostic
norms rank246 above245 on both backends,but this is not an NPU FP64 implementation.

The maximum observed proposal absolute difference before the first selection
split is7.703900337219238e-6. The divergent window contains2325 CPU events and
2327 resident events. Small floating differences can therefore change later
discrete work even when each selector follows its own declared rounded scores.
No capacity loss or CPU schedule prepass is needed to explain this witness.

This explains a concrete mechanism behind the earlier event/loss discrepancy.
It does not establish that every difference in the full512 samples comes from
this one decision:the diagnostic stops at the first mismatch,uses three rather
than nine cards,and its CPU source predates the later eager transport changes.
It does show that differing physical sample sizes are not required to trigger
the problem. Complete gradients were not compared by this diagnostic.

The exact route checks,stable tie policy,model and packet remain unchanged.
The original full-size pair must not be labelled strictly equivalent or used
as an equivalence-certified speedup recommendation. Existing smaller qualified
fixtures retain their original scope. Further numerical policy choices must
be explicit;rounding scores or changing a fixture to hide this failure would
not qualify the original case.

`route-witness02` is terminal,exit0,with its three-card lease released. Exit0
means forensic collection succeeded;its record explicitly sets
`equivalence_qualified=false`. The first diagnostic remains failed:its small
case passed,but original session construction rejected a mistakenly copied
512MiB workspace ceiling. The second restored the historical512GiB whole-program
ceiling while retaining the60GiB per-card admission and queue/trace limits.

Raw:`TASK/runs/route-witness02/assessment/original/observation`.
Audit:`python TASK/launchers/route_witness_evidence.py`,where
`TASK=/mi/data2T/zlong/tide-execution-flows`.
