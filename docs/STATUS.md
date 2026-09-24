# Current handoff

Updated:2026-09-24 (Asia/Shanghai). Branch graph-execution-foundation.
**Active follow-up: profile grad-forward with Full batched, then implement one
measured Aggregate/state/Read optimization and check full training cost.**
Diagnostic gate31 tests passed. Profiling job grad-update-profile-20260924-a
completed/exit0 (unit tide-grad-update-profile-20260924-a, inactive/MainPID0).
Frozen de87fba source/build retained in qualification/grad-profile-20260924.
Add wide Aggregate replay24.26 worker seconds vs state11.84/Read6.93; Update wall
2.07s, Full wall2.38s per batch-token. Attention small state replay dominates.
Chosen candidate implemented: aggregate_autograd=batched, default replay.
552 directed CPU FP64/FP32 tests passed/104.09s in aggregate-vjp-dev-20260924-d.
Keep failed -a/-b and passed -c/-d plus the equal-source probe reproducer.
Implementation commit:8cafa2d. Frozen read-only source:
/var/tmp/zlong-graph-execution-foundation/qualification/aggregate-vjp-20260924.
Clean qualification completed: tide-aggregate-vjp-qualification-20260924-a;
552 selected CPU FP64/FP32 tests passed/105.35s after an independent build.
Unit inactive/MainPID0/exit0. Raw records:
artifacts/aggregate-vjp-qualification-20260924-a/{status.json,task.log,development.json}.
Not a rerun of the whole foundation gate.

Active comparison: tide-aggregate-vjp-comparison-20260924-b;
artifacts/aggregate-vjp-comparison-20260924-b/{status.json,task.log,pipeline.json}.
4 smoke,2 cost probes,12 small training runs completed/exit0; saved owner values,
final-window gradients and loss sequences pass replay/batched parity. Six wide
Add grad-forwards (3 per policy) and one diagnostic are still running/pending.
Do not call the comparison passed until its terminal record is verified.
Command: artifacts/run_aggregate_vjp_compare.py --source <frozen source>
--build-dir <frozen source>/build --output-dir <comparison artifact>
--host-resources artifacts/grad-update-host-resources-20260924.json
--qualification artifacts/aggregate-vjp-qualification-20260924-a.
CPU160-215 (56),dynamic half-memory,OMP/BLAS1; no heavy overlap during timing.
Paths under /var/tmp/zlong-graph-execution-foundation. Drivers retained read-only:
artifacts/run_aggregate_vjp_compare.py,aggregate_training_probe.py,
full_vjp_compare_support.py. Do not modify active inputs or frozen source/build.
Inspect systemctl --user show UNIT and above logs/exit records. Stop with
systemctl --user stop UNIT if required. RUNNING/WAITING is not passed.
Next: await terminal results, audit source/archive/build/packet/run records,
review wide repeats and small actual backward/train-step results, write
aggregate-batched-autograd evidence, update ROADMAP/STATUS and commit evidence.
Audit helper: artifacts/audit_aggregate_vjp.py (run only after terminal success).
No push, subagents or reference-repository writes. STATUS is the sole handoff;
ROADMAP is the sole backlog; semantics.md is the canonical local contract.

## Latest delivery

[Full VJP evidence](evidence/full-batched-autograd.md),
[contract/API](full-batched-autograd.md). Implementation:764794f3d3936a40a507da6e1e0056c46c740b9a.
Optional `full_autograd=batched` / `--full-autograd batched`; default replay retained.
Removes Full affine replay with first-order isolated row VJPs. Aggregate/state/Read
replay remains. Supports declared CPU FP64/FP32 profiles, preserving None/zero,
shared/unused owners and controls. Custom ancestors need undefined-safe VJPs.

Clean read-only source: /var/tmp/zlong-graph-execution-foundation/qualification/full-vjp-20260924.
Independent build in its build/;690 selected tests passed/104.25s (not a global
foundation rerun). Source/Git/archive505 files and13 binary hashes match.
Raw qualification: artifacts/full-vjp-qualification-20260924-a/.
Retain development failures -a/-b/-c and passed -d as described in evidence.

Performance: artifacts/full-vjp-comparison-20260924-a/.
4 smoke,2 cost probes,4 wide runs; all10 completed/exit0 and schemas validated.
Reviewed audit, analysis, pipeline, native-build, resources and per-run records
are present. Three optimized means:10.22666, 10.49906, 10.61330ms/sample-token.
D2048/B512 Add9.468B: new median10.49906, same-binary replay control110.43917;
about10.52x throughput. Grad-forward only, no backward/optimizer/detach.
56 CPUs160-215 within half-host budget160; dynamic half-memory about814.7GiB.
Peak optimized RSS77.48–78.37GiB.
Exact forward checksum/work sequences agree; independent gradient anchors above.
Both qualification/comparison units inactive/MainPID0/exit0, no native descendants.
Portable source kit: export/cpu-attention-compare.tar.gz beneath the performance
artifact directory. Trackio degraded; authoritative local records are complete.

## Retained foundation and previous comparison

Six-stage finite CPU foundation acceptance remains complete:
[evidence/foundation-final.md](evidence/foundation-final.md),
[capability matrix](execution-capabilities.md). Original gate7741 tests,
17 relocated smoke variants,36 fresh-process checkpoint trajectories. Six
implementation classes, independent schedules, native Settle construction and
encoding, first-order ownership/optimizer/checkpoint scopes are retained.
No claim of arbitrary model import, higher-order AD, GPU/NPU or complete
controller/RNG/data-cursor recovery. Native named-value checkpoint does not
contain graph continuation. Large assessments retain their timeouts/unlaunched
cases; do not relabel or mechanically repeat them.

[Earlier Add evidence](evidence/add-scale-comparison.md):3 repeats per engine/mode;
PDG nograd5.59769, LH nograd5.82417, PDG replay grad-forward108.23288,
LH grad-forward7.12832ms/sample-token. Independent engine weights; graph/module
scale comparison. Original raw artifacts/add-scale-comparison-20260924-a stay.

## Re-entry and next work

```
git status --short --branch
git log -6 --oneline
/home/zlong/anaconda3/bin/python scripts/status.py
```

User-authorized follow-up is implemented and independently qualified; the live
comparison above must finish before closing. Inspect its terminal status and
review all repeats, not just the fastest sample. Keep simple oracles, explicit
options, immutable sources, retained failures and coherent local commits.
No whole-foundation restart. Read ROADMAP for extensions and resource bounds.
Python:/home/zlong/anaconda3/bin/python; aarch64/Torch2.10.0+cpu/GCC10.3.1,
TORCH_DEVICE_BACKEND_AUTOLOAD=0, correctness pools1/build2. Resource limits remain
half effective CPU/memory and at most8 Ascend cards for a separately requested
extension. Reference LH, fractal-latcarf and ObsidianVault are read-only.

Retain grad-profile-dev-20260924-a failed wrong-path record and -b passed.

Retain aggregate-vjp-dev-20260924-a:269 passed/16 failed. Failures: expected
exception class, missing Settle test option forwarding, FP32 positive-singleton
normalization cancellation. -b fixes these without changing comparison tolerances.

-b:532 passed/1 failed. Exact singleton zero changes tiny scalar VJP residuals
that AdamW amplifies. -c keeps weighted-mean normalization per event, batching
only its products/sums; no tolerance change. Other normalization profiles shared.

-c passed534 tests. Additional equal-source near-zero probe found shared
softmax Jacobian accumulation can change AdamW updates (raw reproducer retained).
-d keeps a batch/event axis through softmax VJP before summing owner gradients.
Weighted-mean coefficients remain per event. Added explicit optimizer regression.

Comparison -a cancelled during qualification wait, before any run, to correct
training manifest seed to actual explicit7. Original drivers/cancellation record
retained in its directory. -b uses the corrected read-only helper.
