# Consumer private-bank liveness accounting

Qualified clean source `219719dfb15cf9e2c30f7e7facac5feb1a5c19cb` on2026-10-03.
[Reviewed record](consumer-private-bank-capacity-20261003.json) pins source,
consumer/build/loader/runtime hashes,terminal jobs,tests and allocator results.
Runtime bytes remain those of qualified [b5e6345](resident-attention-borrow-20261003.md).

Aggressive multi-device declared Add/Attention consumers already borrow private
frozen projection and complete ordered fiber Attention banks. Forward parameters
include those tensors; the complete-consumer envelope now removes their second
retained-copy charge. It reports independent `retained_parameter_copies` and
included `borrowed_parameter_banks` footprints. Conservative and legacy single-
device consumers retain their copy allowance. Dynamic state/KV/journals/messages,
Full snapshots,gradient/optimizer/accumulation costs,API limits and all per-card
safety margins are unchanged. This deduction does not apply to arbitrary
mixed-head/subset library tapes. [Contract](../consumer-capacity.md).

Four clean jobs passed/exit0 with empty control groups and released leases:
CPU26 directed static/C++ parity and actual model-inventory tests;
installed consumer build;NPU25 cases;D512 actual allocator calibration.
The NPU gate executes24 independent CPU-referenced training/sample-slicing
candidates and checks one pre-allocation refusal. No skips. Only affected
consumer objects were rebuilt or reused after source/header/options checks;
backend/core/CANN artifacts were byte-verified,not rebuilt or profiled again.

D512/B8/T4/V257 Attention,physicalB2×4,two connected windows,one FP32 AdamW update:

| Per-card bytes | Previous estimate | Revised estimate | Observed allocator growth |
| --- | --- | --- | --- |
| Logical0 | 24575275428 | 24006886288 | 7233247232 |
| Logical1 | 16702870820 | 16134481680 | 6386085888 |

Each estimate decreases568389140bytes. Actual allocator growth is exactly the
prior qualified observation; loss,all statistics,output count,cut,physical groups
and effective operator chunks match. This changes accounting,not allocation or
throughput. It is not a same-lease speed experiment. Both new estimates and
continuation pools remain valid under the unchanged margins.

Raw source: `TASK/sources/private-bank-capacity-clean01`; consumer:
`TASK/builds/private-bank-capacity-consumer-clean01`; runs:
`private-bank-capacity-{cpu,npu,calibration}-clean01` and
`build-private-bank-capacity-consumer-clean01`.
`TASK=/mi/data2T/zlong/tide-execution-flows`.
Audit: `python TASK/launchers/private_bank_capacity_evidence.py 219719dfb15cf9e2c30f7e7facac5feb1a5c19cb`.
The initial audit helper pointed to an older dependency manifest and was rejected
by its identity assertion. Actual runtime bytes matched b5e6345; the reference
was corrected,with the original helper retained as
`TASK/launchers/private_bank_capacity_evidence_initial_reproducer.py`.
No run result or failed historical record was rewritten.

Original Attention full-size training remains pending: the initial coordinator
placement still exceeds its envelope. Static accounting alone does not finish
F6; bounded memory-aware placement,real scale execution,formal comparisons and
F7 integration remain separate work.
