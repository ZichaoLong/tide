# Eager payload owners and connected training

Qualified clean source `55c396073afa2a78376ab84d3b29e5e192850f7e` on 2026-10-03.
The [reviewed record](eager-payload-owners-20261003.json) pins source, binaries,
build reuse receipts, terminal jobs, device traces and their raw hashes.

Public Python/native and independent C++ now support owner-local node payloads,
state and KV; complete-region selection across owners; differentiable message,
control and shared-scale copies; and canonical parameter aliases. Delivered and
pending messages live at their destination node. The implementation preserves
message/edge identity, None versus connected-zero gradients, connected windows
and ordinary optimizer updates. Each candidate independently runs common inputs
and initial parameters; no reference events/routes/gradients drive it.

Eight clean jobs passed with exit 0, empty cgroups and completed NPU leases:

| Gate | Verified scope |
| --- | --- |
| Three builds | CPU, NPU Python adapter, NPU standalone; source/header/options-verified reuse, changed region object rebuilt, fresh links and loader checks |
| CPU pytest | 310 passed, no skips; FP64/FP32 owner tests, placement/public-library/checkpoint/region continuation and affected FP16 single-device regressions |
| NPU pytest | 87 passed, no skips; actual two-NPU Python/native owner matrix and configuration guards |
| Standalone CPU | Each FP64/FP32: 12 configurations, 24 updates, 48 connected windows |
| Standalone NPU | FP32: 36 configurations, 72 updates, 144 connected windows; both devices use nondefault streams and two node workers |
| Separate profile | One two-NPU mixed-C Attention/HST greedy configuration, two updates/four connected windows; includes CPU reference and assertions |

Owner training covers Add/Attention, PDG/TimedDAG/Settle, streaming/greedy and
mixed A/B/C. Tests compare complete observables, input/parameter gradients,
optimizer slots/counters and updated parameters, including shared leaves and
independent roots with zero-valued VJPs. Python checkpoint v5 additionally changes
the device map and schedule between updates; restored state/history/pending and
optimizer slots follow their new owners. Standalone continuation is in-memory;
this does not certify disk graph-continuation serialization in C++.

The profile contains 6,786 device operators: 5,174 AI_VECTOR_CORE, 1,017 MIX_AIV,
564 AI_CORE and 31 AI_CPU. The latter are 23 BOOL/INT64 ScatterElements and eight
INT64 Sort tasks. Both leased devices appear in raw CSVs; the diagnostic observed
no host CPU fallback. AiCPU is accelerator execution. Exact int64 sorting remains
required. Initialization/reference/assertion work is included, so API counts and
overlapping task-time sums are not candidate throughput or a scale forecast.

The development CPU run `mixed-owners-cpu-dev01` remains failed: 304 passed/four
error-text assertion failures, although malformed owners were rejected. The
corrected messages and original custom Region.initial vector/value/VJP contract
are covered by the clean gate; no old failure was relabeled.

The first operationally successful profile `mixed-owners-profile-clean01` saw
untracked runtime output `fusion_result.json` in its source checkout and was not
accepted as clean qualification. All 1,499 frozen source files retained their
hashes. After every job ended, that generated file was preserved under the original
run with `source-artifact-audit.json`; tracked sources and old job records were
unchanged. Replacement `mixed-owners-profile-clean02` ran from its output directory
and passed clean. The initial audit rejection/helper is retained.

Raw source: `TASK/sources/mixed-owners-clean01`. Builds:
`TASK/builds/mixed-owners-{cpu,npu-python,npu-standalone}-clean01`.
Jobs: `build-mixed-owners-{cpu,npu-python,npu-standalone}-clean01`,
`mixed-owners-{cpu,cpp-cpu,npu,cpp-npu}-clean01`, `mixed-owners-profile-clean02`.
`TASK=/mi/data2T/zlong/tide-execution-flows`; each run retains `status.json`,
`task.log`, launcher command and applicable queue/profile records.
Audit: `python TASK/launchers/mixed_owners_evidence.py 55c396073afa2a78376ab84d3b29e5e192850f7e`.

This is the directed eager-owner qualification, not the entire historical gate.
Cross-device messages currently copy individually. Actual large-model consumer
integration, per-card constant caching, capacity/locality planning, packed
transfers, eager FP16 owner training and formal throughput remain open under F5/F6.
CUDA source portability is not CUDA hardware evidence. Attention B512 and final
F7 integration remain required; this increment does not close the overall goal.
