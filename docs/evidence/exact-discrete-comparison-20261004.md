# Exact integer and boolean tensor comparison

Implementation `e69b3bd`, 2026-10-04. [Audited clean qualification](exact-discrete-comparison-20261004.json).

The generic comparator previously passed floating tolerances to every tensor.
It could accept distinct int64 values `2**53` and `2**53+1`, or different integer
and boolean values when callers supplied a large payload tolerance. Integer and
boolean tensors now always use zero absolute and relative tolerance. Float
tolerances, dtype/shape checks, None versus zero and candidate execution are unchanged.

The clean affected CPU gate passed **121 checks in 118.18 seconds**: exact
large/negative int64 boundaries, nested discrete records, masks, shape/dtype and
gradient presence, the three-family qualification suite, accelerator boundary
and continuous flow semantics. The two-device eager FP16 regression passed
**49 checks in 213.85 seconds**, no skips. Both services exited0, are inactive,
have empty cgroups and released the device lease.

Source inventories and matching CPU/Python-NPU core binary hashes were verified.
The NPU regression explicitly reuses the qualified `7b1fae5` external consumer;
this Python comparator correction requires no native rebuild. Later CPU RSS
accounting changes are not attributed to that earlier compiled NPU consumer.
Recorded warnings concern shared installation owners, TorchScript deprecation
and base-format allocation; no host tensor-compute fallback warning appeared.

The existing full-size near-tie route failure remains failed. This correction
does not turn it into a pass or establish full-size/GPU performance. Initial
development collection used the repository-reserved fixture name `dtype`;
renaming the local parameter to `storage_dtype` fixed collection before the
48-check development run. Neither development failure nor historical evidence
is relabelled by the clean gate.

Raw: `TASK/runs/compare-discrete-{cpu,npu}-clean01`; frozen source
`TASK/sources/compare-discrete-clean01`. Re-audit using
`python TASK/launchers/compare_discrete_evidence.py`. Exact commands and hashes
are in the machine report.
