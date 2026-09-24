# Current handoff

Updated: 2026-09-24 (Asia/Shanghai). Branch: graph-execution-foundation.
**The authorized Aggregate grad-forward follow-up is complete. No live project
job or queued task remains.** No push, no subagents, no reference-repository
writes. STATUS is the current handoff, ROADMAP is the sole backlog, semantics.md
is the local semantic contract. ROADMAP lists optional bounded extensions.

## Latest delivery

[Aggregate VJP evidence](evidence/aggregate-batched-autograd.md),
[contract/API](aggregate-batched-autograd.md). Implementation: 8cafa2d.
Optional `aggregate_autograd=batched` / `--aggregate-autograd batched`, independent
of `full_autograd=batched`; both defaults remain replay. Built-in source programs
preserve isolated public roots, None/connected zero, shared owners and per-event
normalization backward. No-grad/inference retain the old numeric path. State and
Read replay remain; CPU FP64/FP32 first-order scope only.

Frozen read-only source:
/var/tmp/zlong-graph-execution-foundation/qualification/aggregate-vjp-20260924.
Independent build in its build/; 552 selected tests passed in 105.35s. This is
not a whole-foundation rerun. Qualification raw records:
artifacts/aggregate-vjp-qualification-20260924-a/.
Unit tide-aggregate-vjp-qualification-20260924-a: inactive/MainPID0/exit0.

Performance and reviewed audits:
artifacts/aggregate-vjp-comparison-20260924-b/.
All 25 records completed/exit0 and schemas validate: 4 smoke, 2 probes, 12 small
training, 6 wide Add forwards, 1 diagnostic. Audit verifies 514 frozen tracked
files against Git/archive, 13 binaries, packet/driver hashes, timing denominators,
logical work/checksums and complete saved small-training owners/final gradients.
Unit tide-aggregate-vjp-comparison-20260924-b: inactive/MainPID0/exit0; no native
children or live project records. Trackio degraded; local records authoritative.

D2048/B512/V50304, Add 9.468B, FP32, 56 CPUs, Full batched in both controls:
Aggregate replay median 11.61919 → batched 8.55906 ms/sample-token (3 processes
each), latency -26.34%, throughput +35.75%; RSS 77.43–78.76 → 73.54–74.09 GiB.
No backward/optimizer in this wide interval. Small D128/B16/T16, 4 windows:
total training time Add 1.44187→1.04816s, Attention 5.47745→4.98471s (3 repeats).
All six small parameter/gradient/loss comparisons pass. No new LH or wide
no-grad result. The wide diagnostic now places most wall time in the Full phase;
Attention's small diagnostic points to state/KV. See evidence for exact scopes.

Updated portable source kit:
artifacts/aggregate-vjp-portable-20260924/cpu-attention-compare.tar.gz.
SHA256: fa5099e7e4424cc7fc69ab24e2974e24d8273ff7fffb528832d85ab5f0106334.
Packet source1065978 changes only README versus the measured packet; 267-file
inventory/native/runner equality, archive contents and relocated hash/help checks
pass. Target-local LibTorch rebuild remains required.

Retain the raw Aggregate development failures, near-zero AdamW reproducer,
profile wrong-path failure and comparison -a cancellation. Their exact reasons
and corrected gates are linked in the evidence; never relabel old attempts.
Drivers and reviewed analysis helpers remain under ignored artifacts/.

## Retained foundation and earlier results

Six-stage finite CPU acceptance remains complete:
[evidence](evidence/foundation-final.md), [capabilities](execution-capabilities.md).
Original gate: 7741 tests, 17 relocated smoke variants, 36 fresh-process checkpoint
trajectories. Six implementation classes, independent schedules, native Settle
construction/encoding, first-order ownership/optimizer/checkpoint scopes remain.
This does not certify arbitrary models, higher-order AD, GPU/NPU, or full
controller/RNG/data-cursor recovery. Native named-value checkpoint excludes
continuation. Keep large timeouts/unlaunched cases; do not mechanically repeat.

[Full VJP evidence](evidence/full-batched-autograd.md): clean 690-test gate,
Full-batched Add median 10.49906 versus same-binary Full-replay 110.43917
ms/sample-token. That earlier run batch is not the current Aggregate control.
[Earlier Add comparison](evidence/add-scale-comparison.md): PDG/LH nograd
5.59769/5.82417 (overlapping ranges), LH grad-forward 7.12832 ms/sample-token,
3 repeats each. LH was not rerun in this increment; it is historical context.

## Re-entry and next work

```bash
git status --short --branch
git log -6 --oneline
/home/zlong/anaconda3/bin/python scripts/status.py
```

Read ROADMAP for bounded next choices. Wide Add: inspect remaining Full phase
before optimizing another kernel. Attention: state/KV VJPs are the stronger
measured candidate. TimedDAG/Settle reuse this kernel with correctness coverage;
performance needs its own bounded question. Do not reopen completed acceptance.
Keep serial/replay oracles and optional policies; qualify coherent implementation
commits from frozen independent builds and commit evidence separately.

Python: /home/zlong/anaconda3/bin/python; aarch64/Torch2.10.0+cpu/GCC10.3.1;
TORCH_DEVICE_BACKEND_AUTOLOAD=0, correctness pools1/build2. Resource budget remains
half effective CPU/memory, at most8 Ascend cards only for an explicit backend
extension. The completed comparison used CPUs160–215 (56), host half-budget160,
and dynamic half-memory747.38GiB. LH, fractal-latcarf and ObsidianVault stay read-only.
