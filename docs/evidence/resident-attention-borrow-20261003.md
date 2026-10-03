# Private Attention-bank retention

Qualified clean source `b5e634588405c47320ff03692bbf6cdb5af1d356` on 2026-10-03.
[Reviewed records](resident-attention-borrow-20261003.json) pin all source/build,
fixture/result, loader, reused archive/member and profile hashes. No public ABI,
core schedule, formula, device kernel or retained API capacity changed.

Aggressive private sharded training owners may retain references to their frozen
event QKV/output matrices and complete ordered fiber parameter banks. Whole-bank
node order, sentinel geometry, dtype/device/contiguity and source identity/version
are checked; publication with live windows remains prohibited. The internal tape
field order has an explicit mapping to forward-bank fields. Subset/mixed-head
gathers, default/conservative/legacy snapshots and dynamic state/KV/message
records keep independent ownership. The new `borrowed_attention_bytes` is an
included footprint, not an allocator saving. Consumer estimates remain unchanged.
See [retention contract](../resident-retained.md).

Eight terminal jobs passed with exit0, empty control groups and released leases:
three standalone/Python/installed-consumer builds; native, Python and actual
consumer correctness; a same-lease allocator comparison; a separate FP16 profile.
Affected host objects were rebuilt during development and reused only after
source/header/options/hash checks in the immutable build, then freshly linked.
All eight private-layout owner users and three retention archive members were
covered. Unchanged core/CANN artifacts were byte-verified; this is not a complete
from-scratch kernel rebuild.

- Six native FP32/FP16 cells:160 independent CPU-referenced trajectories,
  2560 windows and640 updates, including aliases, multiple owners, retained
  state/cache/message adjoints and continuation/restore.
- Python retention/snapshot/restore:16 passed. Actual complete-training/sample-
  sliced consumers:32 passed. No skips; unrelated unchanged CPU suites omitted.
- D512/B8/T4/V257 Attention,128 body nodes,physicalB2×4,two connected windows,
  one FP32 AdamW update,old/new independent processes on the same two-card lease:
  peaks `[7508509696,6661348352]` → `[7233247232,6386085888]` bytes.
  Both cards decrease275262464bytes (262.5MiB). Loss remains exactly
  `7.532631874084473`; all prior statistics, effective chunks and admission gates
  match. The new borrowed Attention footprint totals546327592bytes across owners.
- Separate two-card FP16 actual-consumer profile:53176 observed operator rows,
  comprising50628 AI_VECTOR_CORE,723 MIX_AIV and1825 AI_CORE. No observed AiCPU
  rows or fallback diagnostics. This small trace is not a universal kernel-path
  or performance claim; profiler timings are excluded from throughput evidence.

Original `attention-borrow-native-dev01` remains failed/exit1. Its new ownership
assertion caught a field-order mismatch before graph trajectories; explicit
mapping fixed it in dev02. The frozen reproducer, failed log and their hashes are
retained in the reviewed record, never relabelled passed.

Raw source: `TASK/sources/attention-borrow-clean01`; builds:
`TASK/builds/attention-borrow-{standalone,python,consumer}-clean01`; run names use
`attention-borrow-{native,python,consumer,memory,profile}-clean01` and the three
`build-attention-borrow-...` variants. `TASK=/mi/data2T/zlong/tide-execution-flows`.
Audit: `python TASK/launchers/attention_borrow_evidence.py b5e634588405c47320ff03692bbf6cdb5af1d356`.

The allocator comparison/profile ran after original Add B512 terminated. No
full-size Attention training or formal speed recommendation is inferred from this
increment. Calibrated consumer admission, memory-aware placement, full-size
Attention training and the remaining F1–F7 matrix continue separately.
