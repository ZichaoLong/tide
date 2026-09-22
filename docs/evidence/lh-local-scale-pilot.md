# Original LH local CPU scale pilot

Completed 2026-09-22. Historical 8.8B/8.5B observations are reference points,
not performance gates. Eight original-LH cases and all 36 measured forward
events completed successfully. This is an exploratory original-C++ measurement;
large Tide/LH parity and a Tide/LH speed ratio remain unmeasured.

## Source, runtime and workload

Native harness/build and first four runs:
`a7edf44eb7046a307c9a53085ea2420e76e644b1`.
Follow-up wrapper with explicit BLAS control:
`9548be6ecb6635c2beed2aee0b62a1eb5cb05282`.
Both frozen checkouts are clean; all 365 tracked files in each matched its Git
archive after execution. Both use the same native binary, SHA256
`a04a2c8ca712d042fdfb1943ef6902b4a15de77320ff08e553319a3ab54ea647`.
The unchanged 69-file LH C++ snapshot is `artifacts/lh-source-20260921-1428`,
identity `ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`.
Its manifest records original HEAD and actual dirty-source hashes. LH was read-only.

CPU aarch64 HiSilicon (model name unavailable), 320 physical cores on the host;
each large process pinned to CPUs 160–215, 56 cores in socket 2 / NUMA 4–5.
Default first-touch memory, shared host, no exclusive CPU or memory reservation.
Python 3.11.15, Torch/LibTorch 2.10.0+cpu, C++11 ABI, CPU SVE256.
Release build, original `ENABLE_PARALLEL_FOR` OpenMP path, runtime assertions
off, no CUDA. ATen/OpenMP threads 56, inter-op 1; OpenBLAS 56 initially, then
explicitly 1 in two follow-up cases. Placement caps the process to the same
56 physical CPUs even though separate OpenMP and pthread BLAS pools coexist.
All jobs used Nice=10 / background.slice, sequential large cases, a 600-second
and 256-GiB address-space limit per case. No limit was reached.

| Candidate | Hidden | Leaves + hubs | Static nodes per cortex | Actual parameters | Nominal budget |
| --- | ---: | ---: | ---: | ---: | --- |
| wide-add | 2048 | 224 + 8 | 232 | 9,468,020,899 | 1/32: localnum32, selectnum1 |
| narrow-add | 128 | 57,344 + 512 | 57,856 | 9,024,921,853 | 1/64: localnum128, selectnum2 |

Inet and onet have separate states and weights. Static counts are not PDG
counts. Adjacency sizes (input/output/io/oi) are 984/984/232/8 and
245,373/245,373/57,856/512. Both use batch512, FP32, seed7, vocabulary50304,
two layers, bias-free edge Linear, SiLU/RMSNorm, Add memory, allsoftmax
Confluence, clear=true and Add Pronounce. These are **not attention workloads**.
Model JSON came from historical LH revisions `ed4ba40` and `81e702a`; graph
generation uses snapshotted original Graph.py/BaseUtils.py, not the Python
interpreter. The historical experiment's exact graph/configuration remains
unknown. The two actual parameter totals differ by about 5%.

`IOCortexNet::think` timing includes embedding, body, readout, vocabulary head
and lightweight selector counters. It excludes construction, token-ID generation,
finite checks, metric logging and post-forward graph disposal. State updates
inside think remain timed. Tokens are pre-generated random IDs, not output
sampling or a trained language-model quality benchmark. Each run has one
repetition from reset state; warmup advances state. Grad-forward retains the
state graph through warmup and the measured window, with no backward/optimizer.
Both ATen and LH hidden-state grad flags are set, and output requires_grad is
checked. LH grad/nograd also select different hidden-state implementations;
the measured overhead is not an isolated ATen graph-recording cost.

## Observations

Token steps are zero-based. Means divide summed forward time by batch times
measured steps; aggregate throughput divides total sample-tokens by summed time.
Do not use the arithmetic mean of per-step throughput from summary.json.
Ranges describe consecutive token ages, not independent repeated trials.
RSS is the maximum observed process high-water mark, including construction and
warmup, sampled after each measured forward; it is not isolated weight memory.

| Case | Steps | BLAS | Mean ms/sample-token | Range | Aggregate tokens/s | RSS GiB |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| wide-nograd | 4–11 | 56 | 6.096 | 5.065–6.512 | 164.03 | 50.64 |
| narrow-nograd | 4–11 | 56 | 16.109 | 15.108–17.229 | 62.08 | 87.36 |
| wide-grad-forward | 1–3 | 56 | 5.967 | 5.205–6.558 | 167.59 | 51.95 |
| narrow-grad-forward | 1–3 | 56 | 79.962 | 61.504–100.239 | 12.51 | 178.21 |
| wide-short (nograd) | 1–3 | 56 | 5.567 | 4.536–6.395 | 179.64 | 47.28 |
| narrow-short (nograd) | 1–3 | 56 | 15.777 | 12.389–18.322 | 63.38 | 87.30 |
| wide-blas1 (nograd) | 4–7 | 1 | 5.927 | 5.119–6.414 | 168.71 | 50.29 |
| narrow-blas1 (nograd) | 4–7 | 1 | 13.928 | 12.880–14.899 | 71.80 | 87.63 |

Aligned comparisons, always using the same token ages:

| Comparison | Hidden2048 | Hidden128 |
| --- | --- | --- |
| Grad-forward / nograd, steps1–3 | 5.967 / 5.567 = 1.072x time | 79.962 / 15.777 = 5.068x time |
| Nograd BLAS1 / BLAS56, steps4–7 | 5.927 / 5.877 = 1.009x time | 13.928 / 15.570 = 0.895x time |

All four pairs have exactly equal per-step recorded candidate counts, selected
row counts, selected node-job counts and FP64-reduced logit sums. This checks
aggregate workload and a numerical diagnostic; no complete large-model route,
state, tensor or gradient trace was retained, so it is not full equivalence proof.
The 7% wide grad difference and the approximately 10.5% narrow BLAS reduction
need repeated same-window measurements before treating them as stable effects.

The wide nograd case is 2.64x faster than narrow over steps4–11 on this machine.
At these ages, selected fractions including mandatory hubs are about 1/29 and
1/60.3, not exactly the nominal 1/32 and 1/64. Global batch512 also does not
mean 512 rows in each node call: selected rows per selected node job fall from
22.76 to 17.75 (wide) and 21.42 to 8.72 (narrow) over these steps. Counts combine
both cortices and two layers. Narrow selected node jobs rise from 91,800 to
225,550 while selected rows remain near 1.966 million per token step.
Packing shape and active-node coverage are still changing despite stable Full
counts. The result supports further profiling of many small calls, indexing,
state allocation and autograd objects; it does not isolate their costs or
establish an optimal hidden dimension/activation ratio.

## Validation and retained records

Before scale execution, five width64/batch4 harness checks passed: FP32 serial,
FP64, grad-forward, backward and eight-thread parallel. All instantiate
15,514,787 parameters. Counts match exactly; logit sums match the FP32 serial
case at atol/rtol1e-5 (largest FP64 absolute sum difference about 0.000106).
This is harness bring-up, not a whole-model gradient parity gate. Large runs
all checked finite logits, output AD mode and exact native/preflight parameter
counts. Original-LH/Tide correctness evidence remains independently scoped in
[lh-single-graph](lh-single-graph.md).

Completed durable jobs and project-owned records:

- `tide-lh-local-build-20260922-0720` and
  `tide-lh-local-{wide,narrow}-input-20260922-0720`: passed; matching artifacts.
- `tide-lh-smoke-fp32-20260922-0725` and
  `tide-lh-smoke-{fp64,grad,backward,parallel}-20260922-0727`: passed;
  matching artifacts contain each run/summary and raw metrics.
- `tide-lh-scale-pilot-20260922-0730`: passed/exit0,
  2026-09-22T07:25:47Z–07:35:40Z; `artifacts/lh-scale-pilot-20260922-0730/`.
- `tide-lh-window-threads-20260922-0740`: passed/exit0,
  2026-09-22T07:37:00Z–07:41:15Z; `artifacts/lh-window-threads-20260922-0740/`.

Each outer status.json retains the exact command/cwd/source, pilot.json all case
exit codes, and task.log the launch/progress output. Each case has run.json,
summary.json, metrics.jsonl and native/model.json. The follow-up directory also
contains analyze.py, comparison.json, record-validation.json and
post-run-audit.json. All eight large run records validate; native and copied
metrics are byte-identical. Post-run audit checks both frozen trees, native
binary, original snapshot, generated graph files, historical model bytes and
graph-generator files. Initial host/placement is in the first pilot's host.json.
Trackio best-effort is explicitly degraded (ModuleNotFoundError); complete local
records remain available. No dashboard or package installation was required.

Analysis can be repeated without running a model:

```sh
/home/zlong/anaconda3/bin/python artifacts/lh-window-threads-20260922-0740/analyze.py
```

Inputs are `artifacts/lh-local-{wide,narrow}-input-20260922-0720/prepared/input.json`.
Re-execution uses the frozen source and a new output directory; exact per-case
arguments are in run.json. See [the comparison contract](../lh-scale-benchmark.md)
for preparation/build entry points and what a matched Tide run must establish.

No long-context attention, full-scale backward/optimizer, batch-size sweep,
same-ratio granularity study, or large Tide execution was measured. The next
performance increment is the reusable weight-preserving Tide importer and its
small independent parity gate, then a bounded same-host matched scale ramp.
