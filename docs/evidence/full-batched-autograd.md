# Native Full batched autograd: correctness and Add performance

Implementation: `764794f3d3936a40a507da6e1e0056c46c740b9a`. Default `full_autograd=replay` is retained;
`batched` opts into row-isolated affine VJPs. Contract and navigation:
[full-batched-autograd.md](../full-batched-autograd.md).

## Correctness and provenance

Clean read-only worktree: qualification/full-vjp-20260924. Independent build2,
ATen/OpenMP/BLAS1. All **690** selected CPU FP64/FP32 tests passed in104.25s.
This is a bounded gate, not a new7741-test whole-foundation qualification.
It covers finite-difference linear VJPs, independent/aliased/frozen rows, weight
versions, fresh Full and phased Emit roots, HARD/SOFTP/HST, None versus connected
zero, shared/unused owners, native SGD/AdamW, detached and retained continuation,
policy changes at complete cuts, native streaming/frontier/independent schedules
and native Settle. Small Add/Attention scale cases also run actual scalar versus
batched backward checks. Invalid policies and unsupported custom Full are rejected.

The compiled primitive batches only defined output cotangents in backward;
zero-valued cotangents remain connected. Full values and emitted slots retain
separate dependencies. This removes expensive selected-event Full projection
replay without changing Aggregate, Next, clocks, routes or delivery. SemanticValue
and HST propagate undefined upstream cotangents. Arbitrary custom autograd
ancestors require that same contract; higher-order AD is explicitly unsupported.

Audit: artifacts/full-vjp-comparison-20260924-a/reviewed-audit.json. It checks
505 frozen tracked files against both Git and the qualification archive,
13 build/binary hashes, exported packet inventory,10 completed/exit0 native
records, exact timing denominators, logical work and checksum comparisons, CPU
pools/affinity, memory bounds and terminal units. Qualification raw records:
artifacts/full-vjp-qualification-20260924-a/. Performance raw records, drivers,
build/host/resource identities and publication state are under the comparison
artifact directory and named by its manifests. Both units are inactive/MainPID0,
exit0; all native children were reaped. Reference LH/Obsidian sources are untouched.

Retained development failures: -a archive of unstaged deletion; -b153 passed,
25 failed from test profile/exception expectations; -c682 passed,8 failed from
new scale-check fixture's uncompiled identity. Corrected -d passed49 scale tests;
clean690-case gate then passed. None of the old failures has been relabeled.

## Controlled workload and results

Same fixed Add graph as [prior comparison](add-scale-comparison.md): D2048,
B512, V50304, FP32, nominal leaf1/32,224 leaves+8 hubs per cortex,465 PDG nodes,
2208 logical/4418 physical edges; **9,468,020,899 parameters** (9.468 decimal B,
8.818 divided by1024³). All-softmax, fixed0.99 retention, SiLU/RMS body Full,
selected clear,2 body ticks/token. Seed7; no weight or module simplification.

Same56 physical CPUs160–215 on HiSilicon/aarch64, Torch2.10.0+cpu/GCC10.3.1,
from host half-CPU budget160. Dynamic half-memory bound about814.7GiB,
not256GiB. Native pools are1, node/head workers56 in separate phases.
No concurrent heavy project job during timing; shared-host interference remains.
Build and initialization are excluded.12 token steps, warmup4, measured indices
4–11; state and forward graph persist, no detach/backward/optimizer. The metric
is wall seconds*1000/512. Phase/operator/work instrumentation is off. Prior-logit
disposal, token-ID generation and checksum checks remain outside the interval.

4 smoke runs passed, then D256/B32 cost probe means replay6.35159 versus
batched4.58476ms/sample-token. Probe and smoke are not target-scale claims.
The wide run order was batched1, replay1, batched2, batched3, all fresh processes.
The new replay control uses the **same binary**. Earlier repeated LH/PDG records
are retained; LH was not rerun for this optimization.

| Execution | Independent runs | ms/sample-token | Peak RSS GiB |
| --- | ---: | ---: | ---: |
| PDG replay, current control | 1 | 110.43917 | 85.30 |
| PDG batched, median of run means | 3 | **10.49906** | 77.48–78.37 |
| PDG replay, retained earlier median | 3 | 108.23288 | 84.11–85.17 |
| LH grad-forward, retained earlier median | 3 | 7.12832 | 66.08–67.02 |

Optimized run means: 10.22666, 10.49906, 10.61330; range
10.22666–10.61330. Median throughput is95.25sample-tokens/s,
or5.376s/batch-step. Relative to the current replay control,
latency falls90.49% (10.52x throughput).
Relative to the earlier three-run replay median, the ratio is10.31x.
The control has one new repetition; do not describe it as three new paired runs.
Compared with retained LH, optimized latency is47.3% higher.
LH and PDG have independently initialized weights and distinct training contracts;
this is a graph/module/parameter-scale comparison, not function or training equality.

Within each tested scale, all variant/mode/repeat logical-work sequences and
logit checksum sequences agree exactly. The first optimized wide sequence also
matches the retained earlier baseline exactly. This checksum is not a proof of
all wide-model tensor/gradient equality; the independent detailed VJP anchors
are the bounded correctness matrix. Selected Full events remain16896 per
measured batch-token; their scalar Full replays become0 and batched_full_events
becomes16896. The observed gain supports Full projection replay as the dominant
old grad-forward bottleneck for this wide configuration.

Aggregate/state/Read semantic replay and per-event graph construction remain.
This is not a backward/train-step speed claim, a long-context benchmark, an
arbitrary custom/open-weight model certificate, or a TimedDAG/Settle speed result.
Those families share the implementation and have bounded correctness evidence.
Trackio was unavailable/degraded; all authoritative local run records validate.

## Reproduce and continue

New packet: artifacts/full-vjp-comparison-20260924-a/export/cpu-attention-compare.tar.gz.
SHA256: `fccd563e3bccec98d72a6f0830531919f085171c8a5a4d1499bbf571f4c8d281`.
Extract it, activate matching target Torch/LibTorch, and run:

```bash
python run_pdg.py --device cpu --memory add --mode grad-forward --threads 56 \
  --full-autograd batched --phase-profile 0 --work-count 0 \
  --output-dir runs/add-grad-batched
```

Use a new output directory for every run. Replace `batched` with `replay` for
control. `--smoke` performs a bounded rebuild/run/gradient check before wide
allocation. Both policies use the same sources and can reuse an audited binary
through the retained experiment driver; the portable wrapper builds each owned
run directory. Defaults remain conservative. Recommended next performance
question is phase-attributed Aggregate/state/Read VJP cost or actual complete
training-step timing, before opening another large topology/model sweep.
