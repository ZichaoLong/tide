# Add assembly and complete-profile follow-up, 2026-10-10

The Add assembly change is CPU-qualified on clean
`e565f962da63a7e5ceb5ffdbd0ec6699db95c296`. Matching NPU and CUDA rebuilds passed
on clean `a926703e9a41ec6cd41bcf8f2f83665f36bdb999`; host preparation then passed
at e565f96. This report adds no device execution or performance result.
[Reviewed sources, counts and artifact hashes](add-batch-followup-20261010.json).

## Implementation and evidence scope

Add's private structural VJP graph uses row views and reuses the complete
packed content group. Public numeric states and bound outputs still own their
storage. Tick arithmetic/rounding, input version checks, partial cotangents,
None versus zero gradients, clocks and independent references remain intact.
The C++ virtual `batch_graph` boundary defaults to the existing batch behavior;
clock wrappers forward it and only built-in Add changes internal storage.
This interface change required rebuilding both accelerator cores and consumers.

`scripts/profile_execution_flow.py` prepares and collects complete standalone
LibTorch mixed-a/b/c or resident execution through the ordinary consumer.
It checks source, core, resident, compiled-consumer and loader identities;
requires full native results, real operators and memory calibration; and bounds
collection plus export time, process-group RSS and output size. Resident cases
also require queue activity. A kernel list alone does not prove residency;
review the timeline and host APIs. `--prepare-only` performs no workload, and
profile collection is diagnostic rather than throughput evidence. CANN's local
minimum storage limit is 200MB; the planned cases use 512MB.

The a926703 to e565f96 delta is exactly five files: the execution contract,
profile storage validation and three test files. Compiled core, resident and
consumer sources are unchanged. Common core digest:
`0e2d0baa8fb2e3db1aed807c20c4ef78e53838922295dd4f60fb7578313266f0`.
Resident component digest:
`53dfba264427b829ead61e59524e28d490488f1ce854c767be43b00b1ccf058b`.
The CPU build was constructed from the frozen dirty Add development snapshot
based on 8523353, then reused only after exact source/binary checks. Its dirty
construction provenance is retained; this is a clean-source qualification,
not a claim that the reused CPU binary was built from a clean checkout.

## Qualification

| Gate | Fixed source | Result and boundary |
| --- | --- | --- |
| Complete CPU FP64/FP32 inventory | e565f96 | 9648 passed, 654 optional skips, 2 resource-admission failures; original run remains failed |
| Scoped CPU resource repair | same e565f96 | Exactly those 2 tests passed with 8 visible CPUs; composite accepts 9650 unique cases |
| Complete topology suite | e565f96 | All 22 cells passed, including positive-delay PDG feedback and duplicate edges |
| Installed library consumers | e565f96 | All 10 Python/native/C++ checks passed |
| Standalone CPU checks | byte-identical core, e565f96 launch | All 12 CTests passed |
| CUDA core/resident/consumers | a926703 | Fresh standalone/Python and installed/online builds; 7 host CTests |
| CUDA semantic and host checks | a926703 | 102 semantic sources, 1 CPU CTest; 271 CPU tests passed, 84 optional device skips |
| NPU core/resident/consumers | a926703 | Fresh separate Python/standalone cores and owners; installed/online/scale builds; 7 resident and 5 scale host CTests |
| Dynamic imports and absent-CUDA refusal | e565f96, exact a926703 binaries | Both backends import without device initialization; explicit CUDA request correctly fails at preflight |
| Complete-profile preparation | e565f96 | Four actual Add/Attention × mixed-c/resident commands return prepared, with no workload executions |
| Synthetic profile lifecycle | clean adbd51e | 15 passed without vendor initialization; later storage-bound/CLI changes passed 18 directed checks and the e565f96 full inventory |

The two resource failures occurred before any workload: each CLI smoke needs
2 workers plus 2 coordinator slots, while its launcher allows half the visible
CPU set. The newly enforced 6-CPU affinity supplied only 3 slots. Both nested
records have an empty run list and the exact admission error. The repair used
8 visible CPUs on the same source and reran only those two tests; it passed
in 48.37s. `g1-cpu-completion01` audits the exact full failure inventory and
repair JUnit identities, then runs all topology/installed/CTest stages.
Do not describe this composite as one entirely passing pytest invocation.

The audited passing services are inactive/success with MainPID0 and empty
cgroups. The CPU repair/completion guards also report no surviving child
process-group members. Qualification04 remains cancelled after diagnosing an
obsolete owner-map test stub; qualification05 remains failed on an obsolete
backend-name assertion. Their reproducers and subsequent fixes remain retained.
No failed receipt was relabeled, and no floating/discrete tolerance was relaxed.

## Environment, resources and reproduction

This is aarch64, GCC10.3.1, C++17 and Python3.11.15. CPU Torch is 2.10.0;
CUDA uses Torch2.10.0+cu128/toolkit12.8.1; NPU uses Torch/TorchNPU2.10/CANN9.0.0
through the public SDK with separate standalone and wheel owners. Shared
installations were not changed. NPU imports emitted owner warnings; no device
operation was run, so successful import says nothing about device health.

All long work uses frozen sources and separate background.slice services.
Legacy cgroup v1 does not establish enforcement of requested CPUQuota,
MemoryMax or TasksMax. Builds use two workers and single-threaded pools;
the later CPU gates and pending device children additionally use the existing
process-group affinity/RSS/deadline guard. RSS is sampled rather than a kernel
allocation cap. Launch/affinity and deliberate timeout/reap checks are retained;
the intentional timeout record remains failed.

Raw artifacts are under `TASK` in [STATUS](../STATUS.md):
`runs/g1-cpu-{qualification06,resource-repair01,completion01}`,
`runs/g3-{cuda,npu}-clean02`, `runs/g3-followup-host01` and
`runs/g2-profile-entry-clean01`. Corresponding `plans/NAME.sh` and JSON receipts
retain exact commands. `plans/audit-g1-g3-followup01.py` produced the reviewed
`runs/g1-g3-followup-audit01.json` without rerunning workloads; it refuses overwrite.
CPU entry points remain `scripts/qualify_library.py --reuse-build`,
`scripts/library_complex.py`, `scripts/library_consumer.py` and CTest, using a
matching core and new output directory. Allocate at least 8 visible CPUs for
the complete CPU CLI suite. Target build/gate commands are in
[device-control](../device-control.md).

## Prepared device batch and remaining work

`plans/g-stage-device-batch02.{py,json}` passed its complete `--check-only`
preflight at 2026-10-10T13:21:50Z. It is prepared and **not submitted**.
The final read-only inventory at 13:39:53Z again reported all 16 NPUs unavailable;
no physical device was selected and no device process was started.

The finite serial chain is: named FP32/FP16 module gates, eight positive-delay
PDG full-training cells, the complete three-device resident gate, six component
measurements, then four Add/Attention mixed-c/resident traces. Every allocation
wait is at most 60s, and the first failure leaves successors unstarted. Child
service ceilings total 15900s and the controller has 16200s. No automatic retry,
timeout increase or full-size tranche is declared. Trackio remains off.

The complete traces use D64/B8/T4/V257, 32 body nodes/112 edges, TimedDAG prefill,
two connected windows, one warmup and one collected training step. Add uses
SGD and Attention AdamW. Offline estimates fit an 8GiB/card admission budget
with physical B8; real peak-memory calibration remains required. Profile data
will select any next implementation and limited timing comparison.

Custom/nondefault replay, causal depth, output connectivity groups, public
storage isolation and small-tensor organization remain batching costs. Resident
already uses explicit device VJPs. CUDA's one-thread semantic adapter, serial
reductions/scatter and uncalibrated peaks remain performance gaps. Actual GPU
correctness, residency, multicard behavior and performance require a GPU host.

The earlier [CPU selection results](batched-selection-cpu-20261010.md) remain
source-scoped: component improvements did not yield an observed complete Add
gain, and Attention's refused case produced no timing. No speedup is claimed
for this follow-up. Preserve the [strict near-tie witness](original-add-route-witness-20261004.md)
and all old qualification/performance scopes. The stage remains open for device
qualification, profile-led optimization and finite route-selection evidence.
