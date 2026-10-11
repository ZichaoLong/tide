# NPU batched component observations, 2026-10-11

All six planned component measurements are retained exactly once, with forward,
first-order VJP and None-connectivity parity before timing. These are pure
PyTorch FP32 components on one shared NPU,32 rows ×128 features,one warmup and
one synchronized measurement per independent process. They are not complete
LibTorch graphs or resident explicit-VJP performance. [Audited records](npu-batched-components-20261011.json).

| Component | Replay ms | Batched ms | Observed ratio | Kernel events, replay → batch |
| --- | ---: | ---: | ---: | ---: |
| read | 10.9906 | 1.9430 | 5.66× | 235 → 21 |
| state-add | 14.7226 | 5.6592 | 2.60× | 526 → 33 |
| state-attention | 65.2333 | 28.6773 | 2.27× | 2393 → 484 |

Kernel counts and stage durations come from a separate profiled invocation,
not from the timed operation. The Read replay trace has64 finite checks and64
scalar item operations; the batched path groups validation and uses its explicit
VJP. Add profile host attachment/backward stages fall from7.197/12.401ms to
3.085/3.270ms. Attention falls from30.450/61.881ms to14.924/21.855ms, while
its unchanged numeric packing stage remains9.2ms. Batched Attention kernel
durations sum to1.528ms across a43.927ms kernel span. Stage/timeline evidence
supports reducing host graph/dispatch work for these components; it does not
prove kernel fusion or isolate external contention. Nested host timings and
overlapping device tasks must not be added into a wall-time fraction. No CPU
fallback markers were observed in the six traces.

All three tranches selected physical9→logical0,without exclusive access.
Single samples give no variance estimate or universal speedup guarantee.
Keep built-in batch VJPs enabled where supported. Preserve custom/nondefault
replay,causal depth,per-slot connectivity,public storage isolation and fine
controls. Whole-graph mixed/resident selection needs its own evidence.

Read replay was measured at clean00b1775;Read batched at clean1bba497;the four
State cases at clean9fd5263. Python/core sources and the AST of make_case and
operation are identical across these commits;only profiler lifecycle/summary
code changes. No test of an older commit is relabeled a later-source pass.

The first inline exporter exceeded8GiB RSS after preserving timing/raw data.
Offline attempt01 used an unavailable package-level API;attempt02 exposed the
one-worker manager exiting at its NON_BLOCKING timeline helper. Attempt03
explicitly exported text with a blocking timeline dependency and one worker,
finding235 kernels in the retained raw. Its peak was2,892,038,144B. The first
deferred Read batched export completed all18 parser tasks but failed on decimal
string timestamps. Summary-only repair accepted its existing21-kernel trace;
neither Read measurement was repeated. Original failed receipts remain failed.

The reusable deferred exporter was committed as1bba497 and qualified with49
CPU lifecycle tests. Decimal-time support was committed as9fd5263 and qualified
with55 related CPU tests,zero skips. Four actual deferred State exports each
completed all18 parser tasks under the same per-case180s/8GiB/2GiB bounds,
with original capture/timing hashes unchanged. The single-session adapter is
explicitly restricted to TorchNPU2.10.0 and changes no installed vendor files.
It records parser hashes and rejects incomplete task inventories,empty device
traces,fallback markers and mutated inputs.

Artifacts use TASK from STATUS: runs/g2-npu-components10,11,12,
runs/g2-read-replay-export01,02,03 and runs/g2-profile-summary-cpu-clean01.
Audit: plans/audit-g2-components-20261011.py. Passed and failed services are
all terminal with empty cgroups;guards report no surviving child groups.
Affinity/RSS/deadline are sampled by the project guard;legacy cgroup-v1
properties are requested,not proven kernel-enforced resource isolation.
