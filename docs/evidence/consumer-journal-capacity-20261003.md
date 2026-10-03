# Sharded consumer KV-journal lifetime accounting

Qualified clean source `af137df13198b4e0bbd5713249369c691204e8e2` on 2026-10-03.
The [reviewed record](consumer-journal-capacity-20261003.json) pins source,
consumer/runtime/loader hashes, terminal jobs, test results and calibration.
The qualified c38b72e resident/core/CANN execution bytes are unchanged.

The declared aggressive multi-device Attention training consumer has one live
FP32 KV journal and one reusable proposal bank per state owner. Retention clones
one journal per connected window, with aliases deduplicated. The revised Python
and C++ envelope charges these lifetimes explicitly, including int64 metadata,
FP32 values for either payload dtype, count/alignment allowance and separate
packing scratch. Conservative, single-device and inference bounds remain.
No queue, KV, journal or API capacity, safety margin, or dynamic state/gradient/
optimizer allowance is reduced. See the [contract](../consumer-capacity.md).

Four clean jobs passed with exit0, empty control groups and released leases:

- CPU38: affected static planning, storage inventory and C++/Python parity;
  includes FP32/FP16 payloads, one/two/four retained windows and legacy policies.
- Installed standalone consumer build: source/header/options-verified object
  reuse and fresh link; actual native and standalone execution follows below.
- NPU17: sixteen complete two-device candidates, covering native/standalone,
  Add/Attention, FP32/FP16 and operator/owner pressure. Each independently
  compares two complete updates with two connected windows against CPU;
  one additional case refuses before parameter construction. No skips.
- D512 fixed-layout allocation calibration described below.

D512/B8/T4/V257 Attention, 128 body nodes, physicalB2×4, two connected windows,
one FP32 AdamW update. Replay the actual joint owner map and effective chunks
from [c38b72e calibration](consumer-memory-balance-20261003.md), preserving the
packet, capacities, budget and runtime. No reference trajectory drives execution.

| Per-card bytes | Previous estimate | Revised estimate | Observed allocator growth |
| --- | --- | --- | --- |
| Logical0 | 21159582340 | 17375287940 | 6347777024 |
| Logical1 | 13365092892 | 9580798492 | 5545201152 |

The estimate decreases 3784294400 bytes per card. Actual allocator growth is
identical to the prior observation. Loss `7.532632350921631`, output count64,
cut80, events3163, stages104, all other statistics and saved-continuation records
match exactly. The revised envelopes cover observed peaks; unchanged continuation
pool budgets also pass. This corrects accounting, with no allocation or throughput
improvement claimed. No new profile is needed for unchanged runtime bytes.

Raw source: `TASK/sources/journal-capacity-clean01`; consumer:
`TASK/builds/journal-capacity-consumer-clean01`; runs:
`journal-capacity-{cpu,npu,calibration}-clean01` and
`build-journal-capacity-consumer-clean01`. `TASK=/mi/data2T/zlong/tide-execution-flows`.
Audit: `python TASK/launchers/journal_capacity_evidence.py af137df13198b4e0bbd5713249369c691204e8e2`.
The initial calibration helper's static preflight used an invalid `owners`
keyword. It failed before any device work; the helper and failure receipt remain
under `TASK/launchers/journal_capacity_calibration_preflight_*`. The corrected
helper, copied into the successful run, uses the explicit Full/state maps.
No failed runtime or historical record was rewritten.

Original Attention B512 complete training, eager mixed multi-card execution,
full-size formal performance comparisons and F7 integration remain open.
This evidence does not qualify CUDA or another accelerator/runtime stack.
