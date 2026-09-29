# Matched CPU/NPU baselines and full-size NPU profiles

The same-source FP32 comparison found lower CPU inference latency for both full-size
models, while NPU Add complete training was faster. CPU Attention training exceeded
the fixed process time limit after a complete warmup, so its steady training ratio
is unavailable. All four full-size NPU trace windows were collected and their
operator/API exports reviewed; all allocated devices are represented. These are
descriptive shared-host observations, not hardware limits or convergence results.
The bounded scheduler is a separate implementation; none of these baseline timings
demonstrates full-size captured scheduling.

## Source and workload

Both standalone consumers use clean source
`951031e1f7f8b40365ae6ce7f87e7d162a6d5ab2`, identical consumer hash
`7b47afb5a2b8f96893ab3584d1b2dff6d59e1f7cf9712d6f2bc71e4a613f3589`, and
C++ core hash `5e342902e64abbc384099c3903317955e2eac2c9bfa599cd1a9ddf9eb9fbf439`.
The separate CPU and NPU binaries use Torch2.10.0+cpu, GCC10.3.1/C++17/ABI1;
the NPU build additionally uses the qualified standalone TorchNPU2.10/CANN9.0 SDK.
The `+cpu` base Torch tag does not imply that the NPU binary ran its model on CPU.
The CPU binary is separately built for the CPU backend. Before timing,20 CPU
prerequisite cases passed (FP32 payload with FP64/FP32 Read forward/isolated VJPs,
and complete training), as did12 two-device NPU FP32 forward/VJP/training cases.
Those finite correctness gates remain distinct from full-size timing acceptance.

The aarch64 host has320 usable CPUs and8 NUMA nodes. Accelerators are
Ascend910_9392/A3 compute chips,64GiB each; chip counts are not board counts.
Both workloads use465 nodes,4418 physical wires,D2048/B512/V50304/T12,seed7,
FP32 payload/Read/control, HARD signaling and the same initialization/input policy.
Add contains9,468,020,899 parameters (historical binary-unit8.8B label);
Attention contains17,269,426,339. CPU/NPU here denotes model placement.
Read, softmax controls, node ranking and event scheduling use CPU in all timing
cells. Resident model state/KV/message payloads follow their assigned backend.

CPU node/head workers are56 for Add and160 for Attention, with forward
ATen/BLAS1; complete backward/optimizer phases use16, with effective counts
recorded. NPU host workers16/head1/ATen1; inference uses2/4 chips and training4/9.
Affinities span NUMA domains and are recorded. One read-only CPU Add training
sample observed16 live threads and anonymous pages on all8 NUMA nodes; it does
not measure sustained utilization or establish optimal NUMA placement.

Inference advances12 growing-context tokens, first4 warmup and8 measured.
Synchronized timing includes embedding, graph, head, transfers and executor
checks/barriers. Construction, input-ID creation and terminal recording are excluded.
Training has2 complete12-token AdamW updates; the first is warmup. Graph state
resets per update, while parameters/master/slots continue. The synthetic objective
and AdamW policy match the earlier consumer contract (lr1e-4,eps1e-5,decay0.01).
The full update timer includes zero_grad, forward/loss, backward, finite guards,
optimizer and payload updates. Construction and prior-graph disposal are excluded.

## Repeated uninstrumented observations

The fixed plan had24 timing cells.22 were launched:21 completed,1 CPU Attention
training process timed out at7200s. Its later2 repeats were skipped by the declared
stop rule. Each successful group has3 fresh processes; each entry below is the
median and range of measured process means in **ms/sample-token**. This divides
batch-token time by512; training divides a whole update by512*12. It is not
single-request latency. Token positions within one process are not independent repeats.

| Workload | CPU median [range] | NPU median [range] | Descriptive comparison |
| --- | ---: | ---: | --- |
| Add inference |7.387522 [7.185662,7.709558]|17.858098 [17.707540,17.969441],2 chips|NPU/CPU time2.4173|
| Attention inference |19.531847 [18.742095,19.808823]|56.012343 [53.369812,60.457528],4 chips|NPU/CPU time2.8677|
| Add complete training |78.793172 [71.323919,81.003428]|47.932888 [46.626633,50.259633],4 chips|CPU/NPU time1.6438|
| Attention complete training |no measured update|128.275286 [126.192269,134.359332],9 chips|no valid CPU ratio|

Add phase medians explain the observed complete-update difference:

| Seconds/update | CPU | NPU4 |
| --- | ---: | ---: |
| Whole update |484.105249|294.499664|
| Forward |83.226887|229.232860|
| Backward |385.751454|66.704810|
| Optimizer |13.663453|1.420827|

Each phase median is computed independently; summing the phase medians does not
reconstruct one observed update. NPU forward is slower in this configuration,
while backward and optimizer are faster; backward accounts for most of the
observed training advantage. This does not predict other graphs or thread settings.

CPU Attention completed its first, warmup update in4018.027822s
(forward1005.783888,backward2974.257023,optimizer37.986502), then exceeded
the process bound before a measured update was published. That warmup is excluded
from the table and all speed ratios. Its aggregate event/edge/state and owner
counts match all3 completed NPU warmups. Paired measured counts are also checked;
count equality does not establish full-size tensor or gradient equivalence.

## Full-size profiles

All inference windows capture token4, the first token after4 warmup tokens.
The training window captures only backward of update1, after one complete warmup.
Every profile uses full dimensions; no profile timing enters the throughput table.

| Captured NPU workload | Chips | Operator tasks | AiCPU tasks | Runtime launches / copies / sync-category calls |
| --- | ---: | ---: | ---: | --- |
| Add token, CPU32 controls |2|187,854|0|187,854 /74,210 /10,525|
| Attention token, CPU32 controls |4|828,056|0|828,056 /517,443 /86,017|
| Attention token, all32 model controls |4|1,004,756|197|1,004,756 /541,811 /210,468|
| Attention backward, CPU32 controls |9|4,085,290|0|4,085,290 /235,756 /209,080|

The CPU32 Add and Attention exports contain no AiCPU task type. Vector tasks
account for91.31% and85.89% of their summed operator-task durations respectively;
these are task sums, not fractions of end-to-end execution. Mul/Pack dominate
Add's task sums; ConcatD/Pack/Mul/Gather are prominent in Attention. The all32
window adds176,700 operator tasks and124,451 runtime synchronization-category
calls versus CPU32. Arithmetic placement changes without removing host dispatch.
Copy categories include multiple transfer directions and are not just payload
trips to CPU. These profile runs had different device allocations and shared load;
no causal speed ratio is inferred from their instrumented durations.

Collection explicitly enables CANN task time, runtime/ACL APIs and AiCPU events;
AiCore hardware occupancy counters are not collected. The9-chip backward export
contains4,085,290 tasks and no AiCPU task type.
Its44.055256s summed task duration consists mainly of vector work (63.54%) and
AI_CORE work (35.50%); Mul and MatMulV2 each contribute about14.2s of summed task
duration. These per-task sums overlap across devices/streams and cannot be added
to host API time. Per-device operator-interval coverage ranges0.85–10.79% of the
exported operator span, with the same visibility caveats below.

All32 Attention token4 has1,004,756 operator tasks:967,713 AI_VECTOR_CORE,19,262 MIX_AIV,17,569 AI_CORE,
15 MIX_AIC and197 AI_CPU. All197 AiCPU tasks are INT64 Sort; their60.447944ms is
2.0189% of the2.994075s summed operator-task duration across4 chips. These sums
are not a wall-time decomposition. Runtime-level records show1,004,756 launches,
541,811 copies and210,468 synchronization-category API calls. Levels nest and
calls may overlap, so their durations cannot be added into total runtime.

On each all32 device, the union of exported operator intervals covers about
0.97–2.03% of its46.17–48.37s operator span. This is trace coverage, not measured
hardware utilization. DMA, firmware, unrecorded work and profiling overhead can
occupy gaps. It cannot prove that all remaining time is host dispatch. The source
and the many small operations/API calls support reducing launch/synchronization
and packing overhead as an engineering direction; they do not quantify the
speedup a new scheduler would achieve. CPU32 and all32 token4 aggregate work
counts match; complete full-size numerical parity was not checked here.

The all32 original export exceeded its8GiB disk-output bound. A separate offline
export from1,571 hash-checked raw files (1.432432GiB) succeeded. Its first analysis
failed because CANN split1,004,756 rows into two CSVs. Reporting-only commit
`381d44a` adds consecutive-slice streaming and rejects gaps/mixed exports;10
CPU-only directed tests passed. A test-import-only follow-up at`d0c23ad`
also passed all10 tests with the repository Python path, without an extra
scripts-path setting. A separate reanalysis of the preserved export
then succeeded and verified exact4-device coverage. Original failures, successful
export and successful reanalysis retain separate statuses and hashes.

The backward original export also reached the8GiB disk bound. A separate
4,235-file raw copy (3.962986GiB) was re-exported with a48GiB disk bound;
export completed at15:21:50UTC, analysis at15:22:46UTC. Its five consecutive
operator CSV slices and exact9-device coverage passed review; the resulting
profile directory is about19.985GiB. Original collection, failed export and
successful recovery remain distinct records.

Vendor warnings include “Cluster Tuning did not complete!” and failure to find
host/device paths at the parent profile directory. The child PROF directories
contain the operator/API exports with all required devices; accepted coverage
concerns those data, not all tuner features. Both failed original exports and
the all32 single-file-analyzer failure are preserved. No NPU workload was rerun.

## Interpretation and limits

The source retains per-descriptor finite checks/scalar reads, explicit phase
barriers, host C++ dispatch and Read VJP replay. The all32 profiling configuration
moves Read/control/ranking/event-key arithmetic onto NPU, but still returns chosen
indices/minimum times to C++ and uses dynamic event tensors. It is not the bounded
captured backend. Neither device placement nor capture alone proves that result-
dependent host branches were migrated.

Full-size [bounded capacity evidence](bounded-scheduler-capacity-20260929.md)
remains separate: Add eager inference passed on8 chips, whereas12-token captured
prerequisites hit notification limits; Attention inference and Add training had
OOMs. The12-chip Attention FP16 failure had same-device external contention, so
uncontended fit is unknown. Static expansion/device predication passed independent
finite full-observable/VJP/optimizer gates, but a general device event queue and
full-size usable captured replay have not been delivered. Whole reset-window
bounded timings cannot be compared as ratios against growing-context token timing.

The prior [FP32/FP16 comparison](accelerator-fp16-performance-20260929.md) remains
tied to its own source, placement and Read policy. This new repeated baseline is
FP32 only; combining the two reports does not create a matched CPU/FP16 comparison.

Heavy project timings were serialized. Small builds/gates/observers shared host
memory/fabric, and external load remained uncontrolled. Resource sampling starts
at10:38UTC and cannot establish earlier exclusivity or continuous idleness. In
particular, the third NPU Attention training allocation had sampled external
process overlap on physical9. Changing allocations and three-process variance
remain part of the result. No unrelated process was stopped.

## Evidence review

The terminal audit covers11 jobs,67 experiment records (53 completed,14 failed),
and118 passing prerequisite cases in18 gate groups. Source/build/binary/topology
identities, portable records, terminal child cleanup and all67 local Trackio
projections were checked; no audit errors remain. Audit validity does not turn
failed experiments into successful workloads. Best-effort tracking uses local
Trackio0.35.0; no dashboard dependency was introduced.

The [adjacent JSON](accelerator-cpu-npu-comparison-20260929.json) is the reviewed
`reports/delivery-evidence03/evidence.json` pack. It includes all timing samples,
phase medians, commands, source/build identities, failures/skips, profile and
recovery provenance, actual coverage and resource-overlap observations. Reporting
fix381d44a is separate from the measured consumer revision951031e.

Raw records remain under the ignored `artifacts/device-scheduler-JOB` links.
The reviewed summary and audit are `reports/matched-analysis-final01/analysis.json`
and `reports/audit-completed-final01/audit.json`; every report, command and accepted
profile CSV has provenance in the adjacent JSON. Resource observations are copied
to immutable hashed snapshots after the observers stop.
