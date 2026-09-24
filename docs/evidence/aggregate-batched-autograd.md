# Native Aggregate batched autograd: correctness and performance

Implementation: `8cafa2dbbf913d292835af255db8128a21dc8645`.
Contract/API: [aggregate-batched-autograd.md](../aggregate-batched-autograd.md).
Default `aggregate_autograd=replay` remains. Full and Aggregate policies are
independent; this comparison fixes Full to `batched` in both variants.

## Mechanism and correctness

The optional Aggregate Function batches source products and ordered sums,
returning summary and each source contribution as independent public roots.
It preserves undefined versus connected-zero cotangents, physical source aliases
and shared-owner accumulation. Normalization keeps the event dimension through
softmax backward; positive weighted mean keeps per-event normalization graphs.
Changing normalization cancellation near zero can change AdamW updates, so the
optimization does not replace small residual gradients with mathematical zero.
State, Read, Next, routing, clocks and message presence semantics are unchanged.
No-grad and inference-mode use the existing numeric batch implementation.

Frozen clean source: qualification/aggregate-vjp-20260924, independent build jobs=2.
**552 selected CPU FP64/FP32 tests passed in 105.35s**. This is a bounded gate,
not a rerun of the whole-foundation 7741-test acceptance. It covers independent
scalar formulas and finite differences; separate/combined summary and source
roots; frozen/shared inputs; None/connected-zero; HARD/SOFTP/HST; complete
observables; input/parameter/state VJPs; native SGD/AdamW; retained/detached cuts;
policy switches; checkpoint roundtrips; serial/workers; native streaming,
frontier, independent schedules and Settle. Small Add/Attention scale checks run
actual scalar/batched gradient roots. Invalid policies/custom capabilities fail.
The scope is declared CPU first-order programs, not arbitrary autograd ancestors,
higher-order AD, open-weight imports or other backends.

Retained development history under artifacts/:

- grad-profile-dev-20260924-a: failed nonexistent pytest path after a successful
  build; -b passed 31 tests/55.54s.
- aggregate-vjp-dev-20260924-a: 269 passed/16 failed; exception expectations,
  Settle helper forwarding and weighted-mean cancellation.
- -b: 532 passed/1 failed; exact singleton-zero VJP changed the AdamW update.
- -c: 534 passed, then an additional equal-source stress probe exposed softmax
  shared-Jacobian cancellation. -d: 552 passed after the event-axis fix.
- probe_aggregate_nearzero.py and aggregate-nearzero-before.txt retain the
  failing probe. Existing comparison tolerances were not relaxed.
- aggregate-vjp-comparison-20260924-a: cancelled during qualification wait,
  before experiments, to correct the training manifest to actual seed 7. Its
  original drivers and cancellation record remain; it is not a passing run.

## Why Aggregate was selected

Diagnostic source de87fba, artifacts/grad-update-profile-20260924-a. Its binary
was copied from a fingerprint-checked directed build; it is not the independent
rebuild qualification above. All three diagnostic records completed/exit 0.

With Full batched, the wide Add diagnostic measured Aggregate/state/Read replay
at 24.26147/11.83567/6.93020 worker-seconds per batch-token, and Full replay at 0.
Coordinator Update/Full phases were 2.07167/2.38312 seconds. The instrumented
latency was 10.99047 ms/sample-token; it is not an uninstrumented formal baseline.
Worker sums overlap across threads and cannot be added to coordinator latency
or interpreted as wall-time shares. D256/B32 Attention instead measured state
replay 1.12900 worker-seconds against Aggregate 0.11049 and Read 0.10117; state is
a separate remaining optimization candidate.

## Repeated wide Add comparison

Same graph/module scale as [Add comparison](add-scale-comparison.md): D2048,
B512, V50304, FP32, nominal leaf activation 1/32; 224 leaves+8 hubs per cortex,
465 PDG nodes, 2208 logical/4418 physical edges, **9,468,020,899 parameters**
(9.468 decimal B, 8.818 when divided by 1024³). All-softmax, fixed 0.99 retention,
SiLU/RMS body Full, selected clear, 2 body ticks/token. Seed 7, unchanged weights,
module choices, state/message reset and token inputs across the two policies.

HiSilicon/aarch64, Torch2.10.0+cpu, GCC10.3.1. Same 56 physical CPUs 160–215 within
host half-budget 160; node/head workers 56 in separate phases, ATen/OMP/BLAS=1.
Dynamic half-memory bound 747.38 GiB, no fixed 256 GiB cap. No concurrent heavy
project job during formal timing; shared-host interference remains possible.
Each fresh process constructs/resets the model, then retains state/autograd
across 12 token steps; first 4 warmup, measured indices 4–11. No detach, backward
or optimizer in the wide interval. Initialization, checksum evaluation and
logging are excluded. Phase/operator/work instrumentation is disabled for
formal timings; ms/sample-token divides wall latency by 512.

Run order: replay1, batched1, batched2, replay2, replay3, batched3. All use the same
binary. Four smoke runs and two D256/B32 cost probes precede the wide runs;
probe means 4.69340 versus 4.20946 ms/sample-token justified this bounded test.

| Aggregate policy (Full batched throughout) | Three run means, ms/sample-token | Median | Peak RSS GiB |
| --- | --- | ---: | --- |
| replay | 12.12016, 11.61919, 11.04204 | 11.61919 | 77.43–78.76 |
| batched | 8.43641, 8.55906, 8.60314 | 8.55906 | 73.54–74.09 |

Latency falls 26.34% and throughput rises
35.75% (1.358x). Optimized median throughput is
116.84 sample-tokens/s, or 4.382 s/batch-step.
Use these same-binary repeats as the primary comparison; the earlier Full-only
median 10.49906 came from a different run batch and is not the current control.

Within each scale, logical-work/model/logit-checksum sequences agree exactly
across policies, modes and repetitions, including the instrumented diagnostic.
This does not certify every large-model tensor or gradient; independent public
root/VJP anchors are the bounded correctness gate. Selected Full events remain
16896 per measured batch-token, with Full replay 0. Aggregate batched events equal
each token's candidate count (mean 74997.625), with Aggregate replay 0.

The new diagnostic is 8.62134 ms/sample-token, outside formal timing. Aggregate
worker time is 6.99139 s, with state/Read replay 6.69923/5.04631 worker-seconds.
Coordinator Update takes 0.59177 s and Full 2.77939 s of a 4.33591 s tick loop.
Full now occupies the largest
wall phase; it includes actual Full work and dispatch, not scalar Full replay.
Different diagnostics are not paired repeated speed measurements. State/Read
still replay, and worker sums cannot be treated as additive wall shares.

## Actual small training cost

Separate workload: 16 nodes in a four-region PDG feedback ring, 64 edges, each
region selects 2 of 4 nodes; D128, B16, T16, four windows of 4. Native workers 8,
ATen/OMP/BLAS=1, all-softmax, SwiGLU, Add-repeat or event Attention, HST, native AdamW
(lr 0.0002, epsilon 1e-5, weight_decay 0.01). Seed 7; one warmup trajectory followed
by resetting weights, state and optimizer. Three fresh processes per policy and
memory module, alternating order. These are not wide-model training timings.

The table gives the median of three complete measured trajectories, in seconds
(total of four windows). Grad-forward times only execution.advance; backward
and optimizer are timed separately. Complete train-step also includes loss,
zero_grad, detach and associated boundary handling, so it exceeds their sum.

| Module | Phase | Replay median seconds (range) | Batched median seconds (range) | Latency decrease |
| --- | --- | --- | --- | ---: |
| add | grad-forward | 0.26457 (0.25347–0.26503) | 0.23099 (0.22062–0.23473) | 12.69% |
| add | backward | 1.08760 (1.08313–1.09970) | 0.74082 (0.73295–0.78433) | 31.88% |
| add | optimizer | 0.04350 (0.03925–0.04480) | 0.03934 (0.03883–0.04005) | 9.57% |
| add | train-step | 1.44187 (1.42213–1.45650) | 1.04816 (1.04623–1.10725) | 27.31% |
| attention | grad-forward | 0.48258 (0.45804–0.48479) | 0.43695 (0.42291–0.44268) | 9.45% |
| attention | backward | 4.81901 (4.78009–4.92680) | 4.39162 (4.28123–4.39797) | 8.87% |
| attention | optimizer | 0.05691 (0.05568–0.05795) | 0.05489 (0.05452–0.05652) | 3.55% |
| attention | train-step | 5.47745 (5.40677–5.57781) | 4.98471 (4.88223–5.00905) | 9.00% |

Optimizer ranges overlap; its small differences do not establish a separate
optimizer-kernel speedup. Add has 1,841,372 parameters; Attention has 2,889,932.

Saved complete parameter values and final-window gradients pass the existing
FP32 atol=1e-6/rtol=1e-5 comparison for all six pairs, including gradient absence;
loss sequences and logical work agree. The directed gate supplies full public
observable/VJP and optimizer-state checks. Maximum observed parameter/gradient
absolute differences were 1.49e-08/1.75e-10; all six loss sequences match exactly.

## Audit and limits

Audit files: audit.json, reviewed-analysis.json and terminal-audit.json in the
comparison directory. They verify 514 frozen tracked files against Git and the
qualification archive, all 13 build/binary hashes, the packet inventory, driver
hashes, exact timing denominators, CPU pools/affinity, logical work/checksums,
saved training values/gradients and terminal units/PIDs. All 25 local run records
completed/exit 0 and schema validation passed: 4 smoke, 2 cost probes, 12 actual
small training runs, 6 formal wide grad-forwards and 1 separate diagnostic.

Performance records: artifacts/aggregate-vjp-comparison-20260924-b; qualification:
artifacts/aggregate-vjp-qualification-20260924-a. Frozen source/build, source
archive, build manifest, driver hashes, resources, commands/cwd/environment,
per-run logs/raw metrics, saved training values and retained failures remain.
Both job units are inactive/MainPID0/exit0, with no live native descendants.
Trackio was unavailable/degraded; local records are authoritative and validated.
This turn did not rerun LH, wide no-grad or wide backward/optimizer. Historical
LH grad-forward 7.12832 ms/sample-token is only context, not a fresh paired result.
There is no new TimedDAG/SettleGraph performance or long-context Attention claim.
The shared kernel has bounded correctness coverage on their legal schedules.

## Portable reproduction

Updated source kit:
artifacts/aggregate-vjp-portable-20260924/cpu-attention-compare.tar.gz.
SHA256: `fa5099e7e4424cc7fc69ab24e2974e24d8273ff7fffb528832d85ab5f0106334`.
Packet source 1065978 differs from the measured packet only in its README;
267 inventoried files, native/runner byte equality and archive contents are
verified in publication-audit.json. Relocated SHA256SUMS and PDG help pass;
this is not a second native rebuild or a different-hardware qualification.

After extraction, use the target's matching Torch/LibTorch environment:

```bash
python run_pdg.py --device cpu --memory add --mode grad-forward --threads 56 \
  --full-autograd batched --aggregate-autograd batched \
  --phase-profile 0 --operator-profile 0 --work-count 0 \
  --output-dir runs/add-grad-batched-aggregate
```

First add `--smoke` and use a different output directory. For a controlled
comparison retain Full batched and replace only Aggregate with replay. Both
options remain explicit defaults-off choices. State/Read replay remains;
for wide Add, profile the remaining Full phase before selecting another kernel.
For Attention, state/KV VJPs remain the stronger measured candidate. Keep each
follow-up bounded, with independent roots and real backward/optimizer checks.
