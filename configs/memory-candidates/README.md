# Finite memory-binding configurations

These are explicit experimental configurations, not replacement defaults or a
numerical/performance pass. Formulas and ownership: [memory candidates](../../docs/memory-candidates.md).

All use independent non-memory parameters, source_residual, Settle, quad, rank8,
region_batch_size4, gather, SSD prefix summaries/doubling. Small/medium enable
reception recomputation for training; 9b disables it for inference.

| Prefix | Graph | d/summary/Full FFN | Ordinary/Hub/regions |
|---|---|---|---|
| small | compact-5-dd4, n2/h1 | 192/48/768 | 12/5/11 |
| medium | 5-dd4, n8/h2 | 192/48/768 | 192/40/44 |
| 9b | 5-dd4, n32/h4 | 768/192/3072 | 768/80/44 |

Suffixes kv384/kv256/kv128 name the 9b internal widths. Small and medium use the
same fractions d/2, d/3, d/6: 96/64/32. This changes all Q/K/V head counts together;
head dimension stays32. It is not GQA. All-SSD has no KV width at all.

`9b-near-kv128` additionally uses Full FFN3840 (8,768,921,040 parameters),
`9b-near-memory-shared` uses FFN4016 (8,780,695,024). These approximate the original
X total8,778,358,224 but do not preserve module ratios. `9b-memory-shared-batch1`
means region_batch_size1, not independent input batch1; it isolates the base-MM
reuse opportunity against region_batch_size4. Actual B is a benchmark argument.

Run via the frozen durable launcher. Qualification and result links belong in
STATUS/evidence; constructor counts alone do not establish runtime capacity.
