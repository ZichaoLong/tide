# Exact versus single-batch fiber attention

Qualified 2026-09-23 (Asia/Shanghai), clean implementation source
`9e950bfbb7ceee6a5105d7978c0419aab6877859`. The optional
[execution policy](../attention-packing-policy.md) changes built-in same-fiber
attention grouping. Default remains `exact`; graph v13, checkpoint v5 and
tide-core-3 semantics are unchanged. This report covers CPU aarch64 only.

## Result and interpretation

The option is implemented and numerically qualified. The one measured wide
pair does **not** show a speedup from `single`: its mean latency is
7.4042% higher. Keep `exact` as the default; retain both
policies for explicit target-machine comparisons.

Both processes use the same freshly compiled binary, topology, seed 7, fixed
external IDs, parameters and runtime settings. Shape: 17,269,426,339 parameters,
D2048/B512/V50304, four-head Attention/all-softmax, SiLU/RMS, clear, two body
ticks/token, nominal leaf activation 1/32, FP32/no_grad. There are 12 growing-context
tokens; the measured window is indices 4–11. Construction is excluded. The metric
is elapsed time divided by 512 samples, not per-sequence decode latency.

| Metric | exact | single |
| --- | ---: | ---: |
| Mean ms/sample-token | 29.35656 | 31.53019 |
| Median ms/sample-token | 29.53867 | 32.33077 |
| Population standard deviation, ms | 1.03525 | 1.31343 |
| Sample-tokens/s | 34.06393 | 31.71563 |
| Whole-process peak RSS, GiB | 109.51699 | 114.45406 |
| Attention groups per batch-token | 5769.250 | 921.625 |
| QKV projections per batch-token | 921.625 | 921.625 |
| Executed/valid attention score elements | 1.000000 | 1.807569 |
| Counted matrix GFLOPs/sample-token | 6.950791842 | 6.954875870 |
| Update phase seconds/batch-token | 7.40027 | 8.48276 |
| Full/Emit phase seconds/batch-token | 6.07546 | 6.04389 |

Attention grouping calls fall 84.0252%, while executed score elements increase
80.7569%. Total counted matrix work increases only 0.05876%. All other recorded
model/work/operator quantities agree at every one of the 12 tokens: candidate
and selected counts, edges, cached-state counts, QKV/output/Emit/head work and
useful attention pairs. The largest measured phase difference is update:
7.40027→8.48276 seconds/batch-token. Peak RSS increases 4.93707 GiB.

The new path still constructs functional compact KV, gathers an owner's cache
once per query and pools each event separately. It does not adopt LH's reserved
in-place cache or CSR pooling. More temporary KV movement and different matmul
shapes are plausible costs, but this pair does not isolate their contributions.
Fewer attention calls alone is not evidence of improved total performance.

Raw window times, ms/sample-token:

- exact: 26.837914, 29.847665, 29.228582, 29.456215, 29.251591, 30.509554, 29.621122, 30.099868.
- single: 28.770083, 30.476471, 30.704819, 32.479095, 32.687488, 32.308215, 32.353323, 32.462031.

This is one sequential repetition per policy on a shared host, with counters
and phase profiling enabled. Startup load averages were 77.58/82.33/81.66 before
exact and 145.48/115.08/94.81 before single; the latter also includes the preceding
run's recent activity. Terminal load was not collected. These are not controlled
external-load measurements. No confidence interval across repetitions, longer
context, narrow-shape or training-performance claim follows. The older LH timing
in [operator-work evidence](lh-pdg-operator-work.md) is a separate host window;
no fresh large LH timing was performed here.

## Correctness and package qualification

- Fresh CMake build, then `scripts/verify.py --device cpu --dtype both`:
  **6553 passed/745.74s** on the clean source. This is the complete repository
  CPU regression, not a rerun of the separate original-LH Cartesian oracle gate.
- New ragged tests cover distinct old-cache/query/event lengths, future and
  cross-sample masking, streaming/frontier, serial/3 workers, complete state,
  trace, history, pending messages, outputs and routes. FP64/FP32 values use the
  existing tolerances; route identities and gradient presence remain exact.
- Public-root VJPs, absent paths, five pooling modes, cyclic selection/clear,
  shared AdamW updates (`eps=1e-5`), cursors and checkpoints across policy changes
  pass. Training retains the existing scalar semantic VJP replay. This does not
  establish an optimized backward path or higher-order AD.
- Native analytic work counters check padded versus useful scores, multi-event
  prefill and policy validation. The independent Python scalar/exact packed
  anchors were not replaced by wrappers around the new native implementation.
- Freshly exported the kit and extracted it into a path containing spaces.
  Both native builds and D16/B4/V257 six-token smokes passed. PDG-single's native
  `--check 1` compares complete results against scalar slot/row Emit and
  serial/parallel packed execution. LH full logits match the earlier qualified
  small anchor exactly (maximum absolute difference 0).
- All 12 large exact model/work/operator inventories and output checksums match
  the prior qualified exact run. Exact/single output sums differ by at most
  0.00561373056553; each sum reduces 512×50304 logits. These
  checksums are not full large-logit/state/route equality evidence. The complete
  semantic equivalence claim comes from the small tests above.
- Post-run audit passed: 428 frozen tracked files match their Git archive;
  11 native build hashes, 244 packet files, both prepared source trees and their
  binaries/CMake caches, archive/manifest and all four completed run records
  agree. The relocated PDG binary also matches the wide binary exactly.

The two development failures remain terminal failed: attention-dev-20260923-092147
missed a private constructor call; attention-dev-20260923-092415 built and had
402 passed/4 failed because the new test reconstructed an AdvanceResult as the
wrong record type. After repair, that 80-case test file passed / 13.27s against the
same C++ binary; the clean 6553-case qualification covers both fixes. Their logs
and archived sources remain retained, not relabeled.

## Reproduction and retained identities

The portable [runner instructions](../../tools/cpu_compare/README.md) apply to
the refreshed archive. After extraction, use a matching Torch Python or explicit
`--torch-prefix`; these commands build locally and require new output paths:

```bash
python run_pdg.py --device cpu --threads 56 --attention-packing exact --output-dir runs/pdg-exact
python run_pdg.py --device cpu --threads 56 --attention-packing single --output-dir runs/pdg-single
python run_lh.py --device cpu --threads 56 --output-dir runs/lh-wide
```

Run them sequentially on the same target affinity. `--smoke` is available first.
The standalone Python-without-Torch discovery path retains its
[earlier evidence](cpu-comparison-kit.md); it was not freshly rerun here. Intel
x86_64 and other LibTorch versions remain target-machine verification work.

Local environment: aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu,
GCC 10.3.1/C++11 ABI. Correctness ATen/BLAS 1. Wide affinity 160–319, node/head
workers 160 in separate phases; actual ATen/OpenMP/OpenBLAS/inter-op 1. Each wide
native process had a 1200s and 1024 GiB address-space bound. Construction took
228.707522s exact and227.316563s single. No weight checkpoint was written.

Unit `tide-attention-policy-20260923-093207` completed with exit 0; all 12 stages
passed. Persistent status/pipeline and terminal systemd agree: inactive/dead,
MainPID 0, Result=success. Live placement was background.slice, Nice10, bounded
7200s, with 4 build jobs. Source and fixed driver remained unchanged throughout.

- Frozen source: `/var/tmp/zlong-graph-execution-foundation/qualification/attention-policy-20260923-093207`.
- Records: `artifacts/attention-policy-20260923-093207/`: status.json,
  pipeline.json, full-cpu/, export/, relocated kit with spaces/, kit-pdg/,
  kit-lh/, wide-exact/, wide-single/, analysis.json and post-run-audit.json.
- Fixed driver: `artifacts/qualify_attention_policy.py`; terminal audit:
  `artifacts/audit_attention_policy.py`. Exact commands are in pipeline.json and
  the individual run.json files.
- C++ source SHA256: `75629a44b836e891024280783e7f864a61a0defe63eaff4cff123acd9a14acaf`.
- Wide/relocated PDG binary SHA256: `654b10a19551cbe3641706a48ed0dbb5548f83472af1a751e8c0de732c5d98e4`.
- Delivered archive: `artifacts/attention-policy-20260923-093207/export/cpu-attention-compare.tar.gz`,
  384753 bytes; SHA256
  `b45ab7b2fdb18b032f44382c2a5074377adcff854343c83839aacff7efc3b6c5`.
- Manifest SHA256: `ca6cd7187debbfe3063937b904a0c22adce286bb375da74cab2bedd375e560fc`.

Trackio was best-effort/degraded (not installed). The complete local run and
metrics records remain authoritative; no external dashboard is needed. The
archive is the qualified export itself, without a later provenance-changing
repack. No reference repository was modified; no push.
