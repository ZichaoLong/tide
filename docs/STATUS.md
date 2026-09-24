# Current handoff

Updated:2026-09-24 (Asia/Shanghai). Branch graph-execution-foundation.
**Requested PDG grad-forward optimization COMPLETE. No live task job remains.**
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

No further heavy experiment is queued. Suggested next bounded question:
attribute remaining Aggregate/state/Read replay or measure complete backward/
optimizer/train-step cost before optimizing another layer. Keep simple oracles,
explicit options, source isolation, failure records and coherent local commits.
No whole-foundation restart. Read ROADMAP for extensions and resource bounds.
Python:/home/zlong/anaconda3/bin/python; aarch64/Torch2.10.0+cpu/GCC10.3.1,
TORCH_DEVICE_BACKEND_AUTOLOAD=0, correctness pools1/build2. Resource limits remain
half effective CPU/memory and at most8 Ascend cards for a separately requested
extension. Reference LH, fractal-latcarf and ObsidianVault are read-only.
