# LH–PDG inference operator-work comparison

Measured 2026-09-23 (Asia/Shanghai), implementation source
`f0c31bef864af0ccdfa82afc1889c686890fbca6`. Authority remains tide-core-3;
no graph, selection, state/clear or training semantic change. Optional counters
are defined in [operator-work.md](../operator-work.md).

## Result

This wide configuration has essentially equal matrix arithmetic: PDG executes
0.00846% more counted matrix FLOPs; its four linear projection classes total
0.06748% more. It is a substantive comparable-compute workload. It is not an
exact whole-model function comparison: weights are independently initialized.

CPU FP32/no_grad, D2048, B512, V50304, four-head fiber Attention/all-softmax,
SiLU/RMS, clear, two body ticks/token, 12 growing-context tokens. Both use seed7
and external IDs `(3*sample+7*token)%V`. Means below use indices4–11 and divide
by512 samples. A multiply-add counts as2 FLOPs. This inventory excludes
non-matrix arithmetic, memory movement, indexing, allocation and scheduling.

| Per sample-token | Original LH + counters | Optimized PDG + counters | PDG/LH change |
| --- | ---: | ---: | ---: |
| Body candidate events | 145.141846 | 145.205322 | +0.0437% |
| Body selected events | 32 | 32 | 0% |
| QKV source rows | 164.933350 | 165.007813 | +0.0451% |
| QKV GFLOPs | 4.150684 | 4.152558 | +0.0451% |
| Attention output projection GFLOPs | 1.225927 | 1.226459 | +0.0434% |
| Emit GFLOPs (executed in window) | 1.358395 | 1.360673 | +0.1677% |
| Vocabulary head GFLOPs | 0.206045 | 0.206045 | 0% |
| All four linear projections, GFLOPs | 6.941051 | 6.945735 | +0.0675% |
| QK + AV executed GFLOPs | 0.009153 | 0.005057 | −44.75% |
| All counted matrix GFLOPs | 6.950204 | 6.950792 | +0.00846% |
| ms/sample-token | 24.69385 | 28.44610 | +15.1951% |
| Sample-tokens/s | 40.49592 | 35.15421 | |
| Whole-process peak RSS, GiB | 148.81489 | 109.74986 | |

Each column is one repetition on a shared host. Token ranges are23.14648–25.63086ms
for LH and26.03201–29.35530ms for PDG; population standard deviations0.75351 and
1.00155ms. Raw observations are retained. The new28.45ms versus the earlier
29.66ms is not a newly established optimization gain: this increment adds
accounting, and host/timing variation is not isolated. The observed15.2% gap
cannot be attributed to substantially more matrix arithmetic. It includes
local operator implementations and all excluded work, not merely graph dispatch.

## Both cortex networks and the Emit boundary

All four CSR blocks are hash-identical in the two input preparations. The
232 static nodes per cortex comprise224 leaves and8 hubs/leads. inet and onet
have984 edges each and are transposes, with232 iobridge and8 oibridge edges.
Parameters/state remain independent across the two cortexes. PDG has464 body
owners plus one Pronounce node and4418 physical phase edges encoding2208 logical
body edges. Phase aliases do not duplicate parameter owners or force duplicate
edge projections. Both actual parameter counts are17,269,426,339.

LH projects prior activations on receipt in the following body step. PDG
projects when sending. LH executes161.93335 edge/sample blocks per sample-token
in this window; adding `pending_end - pending_start` gives162.18335 generated
blocks. PDG generates162.20483. The resulting difference is only0.01325%.
The boundary correction itself is0.25 blocks/sample-token for LH and0.19702
for PDG. Both12-token records obey the independent flow identity:

`QKV source rows = consumed body edge rows + embedding rows + Pronounce sources`.

Pronounce initially has fewer than two observations per sample; the measured
window has two. No missing warmup observation is invented for accounting.

## Padding, call count and remaining performance questions

LH averages2471.417 valid and4469.187 executed attention score elements per
sample-token (including heads), a1.80835× padding ratio. PDG averages2469.329
valid/executed elements, ratio1.0. LH CROSSBATCH uses each query's own sample
cache, padded to the participating samples' maximum length; it does not compute
cross-sample attention scores. PDG buckets by exact query/KV shapes.

That exact bucketing creates5769.25 attention group calls per batch token,
versus921.875 in LH:6.258× more. QKV calls are921.625 versus921.875, so this is
attention subdivision, not six times more learned projection work. PDG's
combined row Emit takes896.375 projection calls versus1322.25 in LH, with nearly
the same projected elements. QK/AV accounts for only0.132% of LH's counted
matrix FLOPs and0.0728% of PDG's. FLOPs saved through exact attention shapes
therefore need not offset tensor creation, small calls or indexing costs.
This is an optimization hypothesis, not an isolated operator timing result.

PDG's measured batch-token phases: update7.05349s, Full/Emit5.96256s,
selection0.27917s, events0.30694s, commit0.43871s, cleanup0.42976s,
head0.06656s; total14.56440s. Construction227.34508s is excluded.
Generic PDG Aggregate additionally handles337936 source-scale elements and
38507.5 summary-add elements per sample-token. These are separately counted;
LH does not perform this generic summary path. They cannot be priced by
subtracting matrix FLOPs.

Next bounded investigation should measure call/allocation/layout costs inside
update and Full, including exact versus coarser attention buckets and QKV/output
weight layout. Keep the simple reference, complete-state tests and fixed work
inventories. No new kernel optimization or long-context/training claim is made.

## Validation and execution record

- Development:81 directed tests/47.55s, CPU FP64/FP32; frozen source repeat:
  81 tests/48.62s. Paths: test_operator_work.py, test_pdg_scale.py, test_cli.py,
  test_fiber_packing.py, test_lh_original_records.py. This is a directed gate,
  not a repeat of the previous6459-test complete CPU regression.
- Native analytic counts cover ragged two-event prefill, initial cache and
  clear; counted/uncounted values and KV state match exactly. Small scale checks
  compare complete trace, state, history, pending messages, routes and outputs
  against the uncounted scalar path. Serial/parallel counts agree.
- Fresh original-LH D16/B4/V257, six tokens: full logits match exactly with
  counters off/on; serial/4-thread values pass FP32 tolerance and all counters
  match. Original tensor expressions and schedules are retained by explicit
  source transformations in a new owned a10fdb1 copy. Original repos are read-only.
- All12 large PDG steps have the same model/work counters and output checksums
  as the previous uncounted optimized case. This is checksum evidence at scale,
  not a complete large-state equivalence proof or an LH–PDG equality proof.
- Both wide cases exited0 after12 tokens; all five portable run records validate.
  Source inventories, LH tracked/added/binary/graph hashes, exported topology,
  matrix-shape formulas and window flow identities pass the terminal audit.

Immutable source:
`/var/tmp/zlong-graph-execution-foundation/qualification/operator-work-20260923-0300`.
PDG uses copied development build outputs verified against the complete C++
source hash; the original build manifest retains its development base. It is
not a new clean compilation. Both instrumented LH targets were freshly built.

Unit `tide-operator-work-20260923-0300` passed/exit0, inactive/dead, MainPID0;
six driver stages passed. CPU affinity160–319, background.slice, Nice10,
service bound4200s. Wide subprocess bounds:1200s and1280GiB address space each.
LH actual ATen/OpenMP160, OpenBLAS128, default inter-op320 (no inter-op tasks);
PDG node/head workers160 in separate phases, ATen/OpenMP/BLAS1, inter-op1.
Reported pool sizes do not imply simultaneous use of their product.

Records: `artifacts/operator-work-20260923-0300/`: status.json, pipeline.json,
directed-cpu.log, small-prepared/, lh-small-parity/, wide-prepared/,
lh-wide/, pdg-wide/, analyze.py, analysis.json and post-run-audit.json.
Fixed launcher: `artifacts/operator-work-runner-20260923-0300.py`. Commands and
hashes are in the manifests; large weights are not saved. Tracking was
best-effort/degraded (Trackio unavailable); local records are complete.

Limits: one repetition, independent weights, short growing context, original LH
inner timer printing versus PDG phase clocks. Wide counter overhead is not
isolated by a same-source on/off timing pair. PDG startup load is recorded;
its wrapper does not record terminal load, retained as null in the analysis.
The first development build's test-brace error and two repaired analyzer
preflight assumptions (manifest key domains and absent terminal load) are
retained in their failure records. No benchmark failure is relabeled as passed.
