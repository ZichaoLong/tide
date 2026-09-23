# Frozen medium performance assessment

Clean source `9f0cfff00142d55bd595ea31fcd15ec72dfb9815`; all12 logical configurations,108 runs completed with exit0. Three independent processes per variant, two reset warmups. All108 records validate; repeat logical/operator counts are identical within each variant. No new tuning candidate was selected and no default was changed.

Raw records: `artifacts/foundation-medium-20260923-a/`. `reviewed-audit.json` verifies source, all binary hashes, actual thread pools/affinity, process-group cleanup, metric denominators and record schemas. `report.json`/`report.md` retain all phase medians, raw independent durations, min–max and coefficient of variation. Unit `tide-foundation-medium-20260923-a` inactive/dead, MainPID0, exit0. The tuple/list audit readback defect is retained in `audit-readback-repro/`; it changed no experiment record.

CPU aarch64/Torch2.10.0+cpu, FP32, ATen/BLAS1, interop1. Serial variants use the caller thread; worker/frontier/native Settle variants create4 persistent node threads. No vocabulary-head workers. The common dynamically selected affinity spans160 of320 effective CPUs across8 NUMA nodes; this is a shared host, not an exclusive reservation. No heavy task overlapped formal timing. The matching native build is the immutable archived `qualification/foundation-bench-dev-20260923-a/build`; its manifest retains its actual dirty-build provenance.

Each number below is a whole-workload median in seconds. CV is sample standard deviation / mean over the three independent repeats. Shapes/topologies/modules and all defaults are frozen in `benchmarks/foundation-v1.json`; detailed timing/reset/detach definitions are in [the entry contract](../foundation-benchmarks.md). Training rows show the full training step; separate grad-forward, backward and optimizer values remain in the raw report.

| ID / family | variant | median seconds | min–max seconds | CV |
| --- | --- | ---: | --- | ---: |
| P01 / pdg | python-stream | 11.8777 | 11.8015–11.9774 | 0.007 |
| P01 / pdg | native-stream-scalar | 0.1999 | 0.1976–0.2002 | 0.007 |
| P01 / pdg | native-stream-packed | 0.1885 | 0.1864–0.1907 | 0.011 |
| P02 / pdg | native-stream-packed | 0.7464 | 0.7375–0.7556 | 0.012 |
| P03 / pdg | native-stream-packed | 6.1419 | 6.0097–6.2730 | 0.021 |
| P03 / pdg | native-stream-workers | 2.2778 | 2.2372–2.3470 | 0.024 |
| P03 / pdg | native-stream-transport | 2.2012 | 2.1248–2.4970 | 0.086 |
| T01 / timed-dag | native-stream-packed | 0.5788 | 0.5766–0.5926 | 0.015 |
| T01 / timed-dag | native-frontier | 0.2988 | 0.2878–0.3312 | 0.074 |
| T01 / timed-dag | native-diamond | 0.2468 | 0.2347–0.2484 | 0.031 |
| T02 / timed-dag | native-frontier | 3.2052 | 3.1694–3.3196 | 0.024 |
| T02 / timed-dag | native-frontier-step | 4.0173 | 3.8763–4.0176 | 0.021 |
| S01 / settle | python-settle | 1.9793 | 1.9713–1.9846 | 0.003 |
| S01 / settle | python-layered | 4.5655 | 4.5347–4.6161 | 0.009 |
| S01 / settle | native-settle | 0.6027 | 0.5332–0.6246 | 0.081 |
| S01 / settle | encoded-frontier | 0.7276 | 0.6152–0.7514 | 0.104 |
| S02 / settle | native-settle | 0.5772 | 0.5653–0.5856 | 0.018 |
| S02 / settle | native-settle-step | 0.8380 | 0.8372–0.8461 | 0.006 |
| A01 / timed-dag | native-stream-packed | 13.5474 | 13.5062–13.6327 | 0.005 |
| A01 / timed-dag | native-frontier | 3.9566 | 3.7482–4.0496 | 0.039 |
| A02 / timed-dag | native-frontier | 1.8153 | 1.7559–1.8458 | 0.025 |
| A02 / timed-dag | native-frontier-single | 5.4749 | 5.3779–5.7455 | 0.034 |
| A02 / timed-dag | native-frontier-efficient | 3.7184 | 3.7067–4.3483 | 0.094 |
| M01 / timed-dag | native-stream-packed | 0.2992 | 0.2894–0.3029 | 0.023 |
| M01 / timed-dag | native-frontier | 0.5834 | 0.5653–0.6213 | 0.048 |
| M01 / timed-dag | native-frontier-step | 0.3872 | 0.3749–0.4247 | 0.066 |
| M02 / timed-dag | native-stream-packed | 0.3447 | 0.3390–0.3525 | 0.020 |
| M02 / timed-dag | native-frontier | 2.8170 | 2.8143–2.9694 | 0.031 |
| M02 / timed-dag | native-frontier-step | 0.4112 | 0.3919–0.4140 | 0.030 |
| TR01 / pdg | native-stream-scalar | 4.4229 | 4.3982–4.4661 | 0.008 |
| TR01 / pdg | native-stream-packed | 5.1337 | 5.0806–5.1769 | 0.009 |
| TR01 / timed-dag | native-stream-scalar | 4.5409 | 4.4932–4.6615 | 0.019 |
| TR01 / timed-dag | native-stream-packed | 5.1162 | 5.0943–5.1556 | 0.006 |
| TR01 / timed-dag | native-frontier | 5.1368 | 5.1350–5.2994 | 0.018 |
| TR01 / settle | native-settle | 5.7030 | 5.6058–5.9764 | 0.033 |
| TR01 / settle | native-settle-step | 5.3611 | 5.3327–5.6460 | 0.032 |

Observations within this finite suite:
- P01/P02 keep32 body state owners while static nodes increase128→8192. Sparse state allocation holds; client/executor wall time still rises. No attribution experiment was added.
- T02 records max_state_sequence128 with16384 counted state-prefill fallbacks. Legal time batching is real; reset-dependent events retain causal steps.
- S01 generic, independently layered, independently constructed native and Python-encoded frontier paths all run. Python references retain full traces, while native timing uses trace=false; the whole gap is not a kernel-only comparison.
- A01 reaches ragged context2048. A02 single/combined policies are slower than exact in this workload. These observations do not justify changing conservative defaults.
- Linear/Gated Delta frontier prefill is slower than step here; dense affine scan work is explicitly counted. A legal sequence implementation is not automatically cheaper.
- TR01 PDG packed grad-forward/backward/optimizer medians are1.0494/4.0294/0.0203s; scalar0.6572/3.7122/0.0204s. Full steps are5.1337 versus4.4229s. Scalar semantic replay remains a correctness baseline and backward is not claimed optimized. Replay worker durations are measured separately and never added to wall time.

ms/effective-sample-position divides wall seconds×1000 by sum(input lengths); throughput uses the same denominator. ms/batch-position divides by configured sequence length. A01 ragged lengths are explicit. Whole-process peak RSS includes initialization/warmups/profile; combined worker/coordinator RSS and cgroup context are retained separately. Operator counts cover major matrix work and source operations, not all pointwise/backward FLOPs.

These timings use finite-output/state/slot/pending checks and the directed full-trace/VJP benchmark-client anchors (plus the complete CPU qualification). They are not full large-instance equivalence proofs. Source export/relocation and final full qualification remain separate gates. Large-preset outcomes will be reported separately; this report does not certify their target sizes.
