# Current handoff

Updated: 2026-09-23 (Asia/Shanghai). Branch graph-execution-foundation.
Overall acceptance is active, not complete. ROADMAP is the only backlog.
Stage1 audit/frozen twelve-config suite accepted. S2 native standalone SettleGraph
verified at ac19aad; evidence/native-settle-frontend.md has exact qualification.
S2 independent ring/diamond Python+C++ and Python layered Settle implemented at
70809fd; 554 directed tests passed. Complete CPU gate now passes; reviewed evidence below.

S3.2 frontier/native Settle and independent specialization option migration now
passes 1098 directed CPU FP64/FP32 tests (182.07s), build/test exit0. Changes:
packed source/event/sequence transport, causal same-time Next/reset batching,
parallel independent sample-region selection, compact events/deferred cleanup,
explicit prefill fallback reasons and unsupported-policy rejection. Scalar
semantic replay remains the training baseline; defaults unchanged. Source was
frozen throughout the gate. Raw source archive/development/build and terminal
audit: artifacts/frontier-options-dev-20260923-a/. Unit
`tide-frontier-options-dev-20260923-a` inactive/dead, MainPID0/exit0; no descendants.
S3.2 implementation committed as4a2c366; clean stage3 gate remains.

Stage2 clean qualification finished:7167 passed/933.90s, build/test exit0.
70809fd frozen source, all459 source hashes and every binary audited.
Unit tide-foundation-stage2-20260923-a inactive/dead, MainPID0/exit0; no descendants.
Reviewed evidence: docs/evidence/foundation-stage2.md; raw artifacts and frozen
worktree retained at the paths in that report. No stage2 process remains.

S3.1/S3.3 module implementation directed gate passed:736 tests/92.06s;
build/test exit0. Unit tide-modules-dev-20260923-a inactive/dead, MainPID0/exit0,
no remaining processes. Source snapshot qualification/modules-dev-20260923-a;
artifacts/modules-dev-20260923-a has source archive, development manifest,
terminal-inspection.json and logs. Its independent build remains immutable.
Ungated delta-rule-v1, physical same-fiber projection strides, scalar-policy
fallback counters and the tiny RMSNorm/RoPE/GQA/SwiGLU model adapter committed as1a982a1. Module adapter first Python-only gate12 passed/3.36s.
S4 directed gate passed:274 tests/189.86s, exit0. Unit
`tide-training-dev-20260923-a` inactive/dead, MainPID0/exit0, no descendants.
Frozen snapshot qualification/training-dev-20260923-a; verified immutable build
from qualification/modules-dev-20260923-a/build. Artifacts at
artifacts/training-dev-20260923-a include source archive, development/terminal
manifests and retained test-tmp. Audit verified36 unique fresh worker manifests
and payload hashes (pytest current symlink excluded): both dtypes, SGD/momentum
and AdamW, existing single-graph v5, two-clock application and native TIDENCK1
scopes, partial buffers/owners/ledgers and resumed updates. No new checkpoint
format or controller/RNG scope. Six-class multi-update, isolated roots/initial
slots, None/zero/shared owners, truncation and application DenseLinear head also
passed. S4 implementation/scripts/tests committed as5014c3d.

Next: clean combined stage3/4 full regression from5014c3d, using the already
terminal immutable module build (exact C++ hash checked). Its build provenance
is the archived module snapshot; final S6 will rebuild clean source in a new
directory. Follow with separate reviewed evidence commit.

Uncommitted S5 preparation: dynamic resource discovery and optional accounting
for body work, event Attention/SSM/Linear/Delta/Full matrices and scalar replay
worker durations; dedicated bindings adapter. New accounting remains off during
formal wall timing. These edits are outside the S4 source/gate. No formal
benchmark has started and no new performance tuning bottleneck is selected.
No unrelated user changes, no push/sub-agents/reference writes. Native value
checkpoint already qualified; no wide LH/PDG tuning is queued.

Re-entry:
```
git status --short --branch
git log -6 --oneline
/home/zlong/anaconda3/bin/python scripts/status.py
```
Python /home/zlong/anaconda3/bin/python; aarch64/Torch2.10.0+cpu/GCC10.3.1.
CPU FP32/FP64 only. TORCH_DEVICE_BACKEND_AUTOLOAD=0, correctness threads1/build2.
Startup resources: artifacts/foundation-audit-20260923/resources.json;320 CPUs,
aggregate default160, eight NUMA nodes; dynamic half-effective-memory ~755GiB.
Recompute before large work. Disk ~27GiB free. Main repository resolves to
/var/tmp/zlong-graph-execution-foundation/repository; build/artifacts are siblings.

Retain artifacts/specialized-linear-fp32-repro-20260923/: original test squared an
already quadratic composite loss. Direct declared-objective gradients pass the
unchanged tolerance (max abs4.77e-7); fixed only test composition. Prior packed
transport full gate6897 passed/891.90s and wide/narrow performance evidence remain
valid within their scope; no repeated stable combined-option gain, defaults off.
Native TIDENCK1 named values, graph continuation v5 and application bundles are
distinct; none restores the entire training controller/RNG/data cursor.

Durable writes use scripts/durable_records.py and readback. Commit tested code,
then freeze isolated source for long qualification and separately commit reviewed
evidence. Never edit inputs read by a live development job or call a live run
passed. STATUS is current handoff, not a session diary.


Qualification tide-foundation-stage34-20260923-a was cancelled (exit143,
MainPID0, no remaining processes) after review found the S4 Settle training helper
sliced values by logical cut rather than cut/stride. The 274-pass directed gate
therefore proves only the first Settle update; other class/checkpoint scopes
remain valid. Full gate was partial and is not accepted. Raw status/log/result
and cancellation-audit.json are retained. Explicit failing repro:
artifacts/settle-training-clock-repro-20260923/ (exit1, helper/test source hashes).
Main helper is fixed; new assertion requires fresh positions/outputs each update.
Corrected directed gate54 passed/21.92s (CPU both); hashes and exact command
in corrected-gate.json. Ready to commit the test correction, then run a new
clean combined S3/S4 gate. No live job remains.
