# Original-width continued accelerator calibration

All ten planned cases passed on clean `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`,2026-10-04. This closes four LibTorch resident model/mode envelopes,two LibTorch mixed inference envelopes and four Python mixed training gaps. **These are original-width,reduced-batch calibration results;they do not complete the originalB512 performance matrix.** [Audited samples,commands,work counts,peaks and identities](original-width-accelerator-calibration-20261004.json).

All use the unchanged480-body-node/2208-edge graph,D2048/T12/V50304 and9,468,053,696 Add or17,521,117,376 Attention parameters. Each process uses eleven freshly leased NPUs,FP32,one continued warmup and one measured step,two connected windows each. Training includes loss,backward,finite checks and SGD. Cases are serial;phase instrumentation is enabled. The Python CPU/mixed scheduler is independent Python code;the LibTorch binary hash applies to the C++ cells. No CPU reference routes,events,outputs or gradients are input to the candidates.

| Cell | Client/schedule/flow | Model/work | Logical/physical batch | Warmup s | Measured s | Larger B512 phase forecast s,including1.15 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 2 | libtorch/prefill/resident | add infer | 8/4 | 5.265 | 5.148 | 387.531 |
| 5 | libtorch/prefill/resident | add train | 4/2 | 22.729 | 19.825 | 3117.543 |
| 8 | libtorch/prefill/resident | attention infer | 8/4 | 6.458 | 6.310 | 475.334 |
| 11 | libtorch/prefill/resident | attention train | 2/1 | 27.637 | 24.795 | 7398.533 |
| 1 | libtorch/prefill/mixed-a | add infer | 64/32 | 51.433 | 62.056 | 570.913 |
| 7 | libtorch/prefill/mixed-a | attention infer | 32/16 | 61.855 | 81.839 | 1505.836 |
| 76 | python/prefill/mixed-a | add train | 32/32 | 279.218 | 336.290 | 6167.096 |
| 82 | python/prefill/mixed-b | attention train | 8/4 | 147.516 | 181.254 | 13258.712 |
| 88 | python/streaming/mixed-a | add train | 32/32 | 292.447 | 347.113 | 6365.153 |
| 94 | python/streaming/mixed-a | attention train | 8/4 | 142.461 | 176.357 | 12894.608 |

The forecast is `(sample_work_seconds × 512 / pilot_batch + optimizer_seconds) × 1.15`. Fixed optimizer time is not multiplied by the sample ratio. Each source phase is measured,while theB512 forecast is only a launch-budget basis. The record preserves both warmup and measured forecasts. Original3000s refusals remain for resident Add/Attention training and all four Python training cases;separately declared longer budgets do not turn those refusals into passes.

All outputs equal24per logical sample,and all final cuts equal816. Actual event counts and losses are recorded per process. Each requested physical split was retained and every measured allocator peak was inside its declared device estimate. Resident owner validation compares both Full and state owner maps;eager validation compares node owners. These observed peaks do not prove safety for arbitrary future inputs or changed capacities;originalB512 must still pass its own admission and runtime checks.

Python Add uses oneB32 physical group. This bounded follow-up preserves physical rows after the [earlierB64/900s timeout](original-width-continued-mixed-20261004.md),which stays failed. One-group completion does not by itself certify full-batch multi-group persistence. The other eight cases use two physical groups. The earlier source-matching LibTorch mixed training pilots are separate evidence and were not repeated here.

The parent `remaining-accelerator-calibration02` is terminal passed/exit0 at2026-10-03T20:45:55.803368UTC,with an empty cgroup and all leases released. Each queue allowed120s;children allowed900s except Python Add1800s. The whole bound was28200s,including the preceding diagnostic dependency wait. No automatic retry occurred. External server load remains uncontrolled;these phase pilots are not formal speed ratios or recommendations.

Re-audit: `python TASK/launchers/audit_accelerator_calibration02.py`. The first offline audit adapter assumed the eager owner field for resident records and raisedKeyError;its failed adapter copy is retained under `audit-attempt-01`. The corrected audit strictly compares each schema's actual owner fields. Runtime records and production code were unchanged.

Remaining work includes the first actual unprofiledB512 result,necessary original-width Python CPU/mixed-inference/resident calibration,the other full-size family/client/schedule cells,three fresh processes for recommendations,and separately scoped FP16/profiling evidence. The accepted near-tie strict CPU/NPU failure remains a numerical limitation;these independent runs do not relabel it equivalent.
