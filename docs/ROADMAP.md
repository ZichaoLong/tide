# Remaining current delivery

The new FC, Lattice, Exchange and Multiscale Settle/Timed implementations are
present, including K/S/X, four reception profiles, three allocations and three
sharing profiles. CPU formula/integration evidence and the first FC comparison
are linked from docs/evidence/instances-first-cpu-20261117.md. The scope and API
are in semantics.md and instances.md; this list contains remaining work.

1. Qualify the next immutable implementation containing independent long KV
   window checks and the zero-reception continuation-gradient fix.
2. Complete real NPU FP32 Attention/model/training/continuation checks. The first
   attempt failed shared admission before device execution; existing locks
   require explicitly bound shared leases. Preserve that failed attempt.
3. Run finite FC/HB comparisons against parameter-matched and dimension/layer
   matched Transformer/SSD, with complete training and actual candidate/Full
   work. Resolve major measured costs with bounded diagnostics; do not invent
   throughput or universal scaling conclusions from smoke tests.
4. Record the actual backend/device limits, observations and reusable findings;
   leave a clean installed consumer example and concise navigation.

Execution/performance is FP32 only. CPU FP64 may validate formulas. FP16/BF16,
old teaching bindings, LH-DAG/PDG, arbitrary graph executors, historical APIs,
resident/mixed matrices and exhaustive action/dependency records are excluded.
No need to restore or complete old P1–P6. Native/LibTorch kernels remain options
for a demonstrated benefit, not compulsory interfaces for this delivery.
