# Current handoff

Updated 2026-10-03. **ACTIVE: the user cancelled the old goal and explicitly
authorized creation/execution of its replacement.** The new goal is active.
The previous post-commit pause is superseded. Repository next-action text never
overrides a later user pause; commits/evidence/context compression do not stop
or resume work by themselves. Overall F1–F7 remains incomplete.
Eager owner implementation 55c396073afa2a78376ab84d3b29e5e192850f7e is now
qualified below. Actual multi-device consumer integration has passed development checks and is
committed as e5d91d7a20321822fc69f8516331d1da371c0719 and pushed.
Clean affected qualification and its audit passed; evidence is recorded below.
No subagents; reference repositories and ObsidianVault remain read-only.

Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; authorized branch
`graph-execution-foundation`. This implementation follows qualified implementation
`af137df13198b4e0bbd5713249369c691204e8e2` and evidence parent
`dd78b80860a6c9a016ce6a23394238e4c661bb1d`. Resolve this owner implementation with
`git log -1 --format=%H -- python/tidegraph/ownership.py`.
Re-entry: `git status --short --branch`; `python scripts/status.py`.
[execution-flows](execution-flows.md) owns the contract;
[ROADMAP F1–F7](ROADMAP.md) is the sole backlog.

## Contract and current priorities

Deliver the 20-tide public execution/equivalence foundation, including independent
CPU, mixed and resident execution, complete training and continuation, correctness
and formal performance comparisons. Model convergence belongs to later experiments.
Candidates independently consume common inputs/parameters/initial state; CPU events,
routes and gradients never drive them. General online greedy accepts legal inputs
and topologies, including positive-delay PDG feedback, and may become streaming.
Preserve int64/stable order, duplicate edges, missing/zero messages, None/zero
connections and complete window continuation.

Performance scope: PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch;
CPU/mixed/resident × streaming/prefill × inference/complete training. Five presets
and fine switches; FP32 main, FP16 separate. Prefer calibrated aggressive-safe
splitting. Keep 3000s, 1.15 and current capacities comparable unless measurements
justify changes; never remove protection or rewrite old refusals to pass a gate.

The user accepted bounded performance diagnosis: normally at most two measured
improvement rounds and about 90 minutes of active diagnosis/tuning per issue;
already launched jobs retain their declared timeout. Correctness/functionality,
full-size completion and formal measurement remain required. Optional further
speed tuning must not indefinitely block independent mainline work. A justified
longer full-size feasibility run can have a separately declared budget while
preserving the original 3000s refusal. No speedup threshold was specified.

Implementation commit → clean fixed-source affected qualification → separate
reviewed evidence commit. Commit/push is authorized for the active goal.
Minimal useful experiment records; user contract outranks experiment skill.
No unrelated passed-test reruns, unlimited queues, blind retries or OOM search.
Formal heavy timing is serial and separate from profiling. Never stop other work
for resources. Temporary card shortages should leave independent work moving.

## Current increment: eager payload owners

Implemented in public Python, native and independent C++ paths:

- Construction-time node ownership with canonical parameter aliases and
  preallocation rejection of conflicting shared-node placement.
- Owner-local parameters/state/KV, destination-owned messages/pending, complete
  region selection and differentiable cross-device controls/scales/messages.
- Both schedules, complete updates and connected windows; Python checkpoint v5
  restores continuation and optimizer slots under a changed owner map.
- Public `node_devices`, actual manifest placement, all-owner synchronization
  and worker inheritance of nondefault streams on both devices.
- Owner availability preflight restores the runtime's default device on success
  and rejection. Custom Region.initial retains node-zero vector/value/VJP.

[Owner contract](execution-placement.md#eager-payload-owners) records the limits.
Actual consumer integration, per-device fixed-constant caches and static
learned-parameter/locality planning are qualified below. Packed copies are
implemented with clean CPU gates complete; NPU execution is pending. Total-memory
admission, scale performance and FP16 owner qualification remain open. This is not F5/F6 closure.

### Clean qualification completed

[Reviewed evidence](evidence/eager-payload-owners-20261003.md) qualifies exact clean
55c3960. Eight accepted jobs passed/exit0 with empty cgroups and completed leases:
three builds, CPU310/NPU87 (no skips), standalone CPU12 configurations per FP64/
FP32 and NPU36 configurations, and the separate two-device profile. Each standalone
configuration has two updates/four connected windows. Python additionally tests
checkpoint remapping and schedule changes. NPU workers inherit both nondefault
streams. This is an affected gate, not the complete historical regression suite.

The profile reports 6786 device operators, including31 AiCPU operations (23
BOOL/INT64 ScatterElements,8 INT64 Sort), without an observed host fallback. It
includes reference/construction/assertions and is not throughput. Both leased
physical devices appear in raw traces. Do not replace exact integers with float
for speed. No new formal performance result is claimed.

Source `TASK/sources/mixed-owners-clean01`; fresh-linked/source-verified builds
`TASK/builds/mixed-owners-{cpu,npu-python,npu-standalone}-clean01`. Jobs:
`build-mixed-owners-{cpu,npu-python,npu-standalone}-clean01`,
`mixed-owners-{cpu,cpp-cpu,npu,cpp-npu}-clean01`, `mixed-owners-profile-clean02`.
Raw records `TASK/runs/NAME/{status.json,task.log}`, queue/profile records where
applicable; units `tide-execution-flows-NAME`. Audit passed:
`python TASK/launchers/mixed_owners_evidence.py 55c396073afa2a78376ab84d3b29e5e192850f7e`.

Retain development `mixed-owners-cpu-dev01` as failed (304 passed/4 error-text
assertion failures); clean310 covers the repaired cases and custom Region.initial
anchor. The first profile `mixed-owners-profile-clean01` passed operationally but
recorded dirty solely from generated fusion_result.json. All1499 frozen source
hashes were unchanged. After all jobs ended, that inspected output was preserved
under its run with `source-artifact-audit.json`. Replacement clean02 ran with
runtime cwd `{out}` and qualified. Initial audit rejection/helper and all old
records remain intact. Future NPU children use `env -C {out}` and absolute entry
paths to keep vendor artifacts outside immutable sources.

## Established scale results and remaining overall gaps

Original **Add B512 complete training passed** on clean26176de: D2048/B512/T12/
V50304, 9.468B parameters, nine devices, physical B2×256, two connected windows,
one FP32 SGD update in 2170.549862239s. This is cold feasibility, not formal
throughput or a full-size CPU gradient oracle. [Evidence](evidence/original-b512-add-training-20261003.md).
All ten representative performance submatrices and original Add/Attention B512
resident/prefill FP32 inference are already qualified under F6; do not rerun them
without an affected change or unresolved concern.

**Attention B512 complete training passed** on exact29effae, with terminal audit:
D2048/B512/T12/V50304,17,521,117,376 parameters,11 cards,physicalB1×512,
two connected windows,one complete FP32 SGD update in5695.490452595s.
Construction164.822193006s; sample work5692.808817143s; optimizer2.681635452s;
outputs12288,cut408,finite loss21.380962371826172. All allocator and saved-context
peaks pass unchanged capacity guards. The separate9000s budget passes; the original
3000s forecast refusal and1.15 coefficient remain unchanged. This is cold feasibility,
not formal throughput or a full-size CPU gradient oracle. [Reviewed evidence](evidence/original-b512-attention-training-20261003.md) records
the complete audit and limitations.
The B4 forecast was7036.453031774s; measured/forecast=0.8094263441.

Unexecuted task-local `finite_ranked_horizon.py` and `wide_attention_horizon_pilot.py`
are unqualified drafts. Review their claimed 48-row bound and physical B4 proposal
before use; never generalize a fixture-specific bound into the online scheduler.
Do not modify audited `wide_attention_owner_pilot.py`; use a new helper/config.

Current authorized priorities are: integrate
actual mixed consumers with per-device constant caches and complete head/loss/
synchronization; add capacity/locality and packed transfers with independent gates;
retain the completed Attention B512 evidence; finish original-scale CPU /
screened-mixed / resident comparisons for both schedules and required languages/
families, three fresh processes per recommendation, with separate profiles.
F7 must then audit integration, evidence, support and portable commands. CUDA has
no local hardware: deliver portable source/build/test commands and retain explicit
target-machine-pending status, never claim local execution.

## Actual eager consumer qualification completed

Clean implementation e5d91d7a20321822fc69f8516331d1da371c0719 is qualified by
[actual-consumer evidence](evidence/eager-consumer-owners-20261003.md).
Python/native/standalone independently initialize named learned leaves on owners,
use per-device fixed-constant caches, gather head rows, synchronize/observe all
cards and agree on finite gradients before updating. Static learned-parameter/
locality planning or explicit maps retain node zero/boundaries/head/embedding on
owner zero. Total peak admission and packed transfer remain separate work.

Five clean terminal jobs passed/exit0, empty cgroups, completed leases:
`build-mixed-consumer-{cpu,npu}-clean01`, `mixed-consumer-{cpu,npu}-clean01`,
`mixed-consumer-profile-clean01`. CPU73/NPU40, no skips; explicit deselection48/9.
Each actual training comparison has two AdamW updates/two connected windows,
physicalB1×2, independent CPU full-record/gradient/update checks. Inference,
three clients/families, both schedules, mixedA/B/C, CLI owner replay, native
workers/packed-sources/batch-next are represented. Fresh installed consumers
reuse byte-verified qualified55c3960 core/adapters. All1504 source hashes match.
The separate actual Attention two-device profile includes construction/head/loss/
backward/optimizer without CPU reference or diagnostics; no observed host fallback,
not throughput. Audit:
`python TASK/launchers/mixed_consumer_evidence.py e5d91d7a20321822fc69f8516331d1da371c0719`.
Frozen source `TASK/sources/mixed-consumer-clean01`; builds
`TASK/builds/mixed-consumer-{cpu,npu}-clean01`. Raw records remain under
`TASK/runs/NAME/{status.json,task.log}` with queue/profile records where applicable.
Development dev01–dev03 snapshots/builds/tests and the earlier profile are retained;
clean qualification covers the final C++ resident-alias preservation correction.

## Next implementation and independent feasibility

Eager packed cross-card transport is committed/pushed as
a785d43cfd9662c7d7262c0fdd8c59ec44729f6f. Its clean CPU gates passed;
NPU execution and complete evidence audit remain pending. Python/independent C++ multi-output copy
preserve missing/zero cotangents, frozen outputs and aliases, grouping only
completed messages by source/target/dtype/shape in at most8MiB packs. Unpacked
execution keeps individual copies; host metadata/scaling remains eager work.
New independent copy tests include isolated/upstream/optimizer roots, packing
boundaries and second-order linear-copy anchors. Standalone checks also exercise
copy roots and actual grouped-message counters. All1511 frozen hashes in
`TASK/sources/packed-transfer-dev01` match; source changes after its snapshot are
handoff/contract text only.

Development passed/exit0: `packed-transfer-python-cpu-dev01`30 directed Python
cases, `packed-transfer-cpu-dev01`352 affected FP64/FP32 cases with no skips,
`packed-transfer-cpp-cpu-dev01`12 standalone configurations per FP64/FP32 plus
copy anchors (two updates/four windows each), `packed-consumer-cpu-dev01`64
actual-consumer/host-control checks with8 accelerator cases explicitly deselected.
Five builds passed: `build-packed-transfer-{cpu,npu-python,npu-standalone}-dev01`
and `build-packed-consumer-{cpu,npu}-dev01`. All affected scheduler objects,
new transfer/delivery objects, binding and standalone check are compiled; core
archive reuse is source/hash verified and outputs freshly linked. No NPU runtime
pass is inferred from compilation.

Exact committed source is frozen at `TASK/sources/packed-transfer-clean01`.
Five clean builds passed/exit0: `build-packed-transfer-{cpu,npu-python,npu-standalone}-clean01`
and `build-packed-consumer-{cpu,npu}-clean01`. Clean gates passed/exit0:
`packed-transfer-cpu-clean01`352 cases, `packed-transfer-cpp-cpu-clean01`
12 configurations per FP64/FP32 plus copy anchors, and
`packed-consumer-cpu-clean01`64 cases/8 accelerator cases deselected.
No NPU runtime pass is inferred from compilation.
After B512 is terminal, run two-device `test_transfer.py` + `test_payload_ownership.py`
with `TIDE_TRANSFER_BACKEND=npu TIDE_PAYLOAD_BACKEND=npu` (119 cases), actual
consumer40, standalone ownership36 and a separate actual consumer profile. Use
`packed-transfer-npu-python-clean01`, `packed-transfer-npu-standalone-clean01`
and `packed-consumer-npu-clean01` builds. Audit source/archive/binary/job/lease/profile
records, then commit the evidence separately. No stage implies a pause.

Current capacity implementation is committed/pushed as
2c04005255b3d0b674cef302894bb9a5d0f0378e:
Python/native/independent C++ eager preallocation admission, finite static DAG
traffic bounds, complete-run state/KV/optimizer/head envelopes, bounded automatic
sample halving and retained pre/post-run refusals. CPU uses at most half available
host memory and reports incremental lifetime-RSS proxy; accelerators use driver
free memory and allocator peaks. CLI/offline planning and target-pending CUDA
commands are documented in [capacity](eager-consumer-capacity.md) and
[target validation](eager-target-validation.md). Generic online feedback support
is unchanged; no numerical prepass or CPU-driven route is introduced.

Frozen `TASK/sources/eager-capacity-dev03`: 68 CPU FP64/FP32 affected tests passed,
47 explicitly deselected, no skips; both fresh installed CPU/NPU consumers and
static planning probes built. All source inventory hashes match. Jobs
`build-eager-capacity-{cpu,npu}-dev03`, `eager-capacity-cpu-dev03` passed/exit0 with
empty cgroups. Preserve dev01 (35 passed/23 failures: missing C++ JSON array
closure) and dev02 (66 passed/1 failure: missing packet hash in refusal) plus their
snapshots/builds. The final fixes retain strict wrapper identity checking.

Clean source is frozen at `TASK/sources/eager-capacity-clean01`. Fresh builds
`build-eager-capacity-{cpu,npu}-clean01` passed/exit0 using `build_eager_consumer_core.py` and byte-verified
`packed-transfer-{cpu,npu-standalone}-clean01` cores. `eager-capacity-cpu-clean01` passed68/deselected47/no skips.
`eager-capacity-calibration-cpu-clean01` passed four fresh CPU processes
(Add/Attention × Python/LibTorch), D256/B8/T4/V4096, two threads, two SGD updates
and two connected windows per update, per-child480s/outer2100s. Helper
`calibrate_eager_capacity.py`; records `TASK/runs/NAME/calibration/result.json`.
Observed incremental RSS bytes: Add Python224952320/C++128286720, Attention
Python348594176/C++254287872; all below the unchanged estimates. These are
process-lifetime RSS proxies, not accelerator allocation measurements.
That calibration is for memory admission, not throughput; no formal timing claim
may include concurrent B512 work. After B512 is terminal, finish packed-transfer
NPU qualification/profile above, then capacity two-device forced-split gate,
actual-consumer40, finite calibration and separate trace. Planned helper bounds:
480s per fresh calibration child, CPU4 cases/NPU7 cases, no blind retry. Commit
reviewed evidence separately only after its audit. Source compilation and CPU
correctness do not yet certify the memory coefficients for original NPU scale.

Independent F7 host-only CUDA build/CPU/export qualification passed and audited:
[CUDA-host evidence](evidence/eager-cuda-host-20261003.md) records exact2c04005,
fresh core/installed client, CPU187/deselected47/no skips and seven no-Git relocated
entry checks. Four terminal jobs passed/exit0 with empty cgroups:
`build-eager-cuda-clean01`, `build-eager-cuda-consumer-clean01`,
`eager-cuda-host-clean01`, `eager-relocated-entry-clean01`.
Existing isolated `torch-cuda/2.10.0-cu128` module, Python
`/mi/data2T/zlong/gpu-toolchains/envs/torch2.10.0-cu128-py311/bin/python`;
Torch2.10.0+cu128/CUDA Toolkit12.8.1/aarch64, explicit SM80 headless configure.
No environment was changed and no GPU execution is claimed. Audit helper
`eager_cuda_host_evidence.py 2c04005255b3d0b674cef302894bb9a5d0f0378e` passed.
The no-Git export `TASK/exports/eager-execution-2c04005` matches all1522 hashes;
relocated entries explicitly reuse matching CPU core/client binaries, not a fresh
core rebuild. CUDA support remains implemented/device-unverified; real GPU and
x86_64 target commands are in `docs/eager-target-validation.md`.

Static original-scale eager planning is retained at
`TASK/plans/eager-wide-capacity-2c04005.json` (no Torch/model execution).
For B512 one complete update, 512GiB CPU cap admits Add/Attention with physicalB32;
11×60GiB mixed cap admits AddB32/AttentionB8. At3 measured+1 warmup connected
updates,11-card Attention refuses evenB1 under current locality (estimated54.06GiB,
usable53.875GiB); preserve that refusal, calibrate before deciding placement/budget.
This is an estimate, not measured capacity or completion.

Independent CPU scale pilot is authorized/prepared as `wide-eager-cpu-pilot01`,
source `TASK/sources/eager-capacity-clean01`, build `TASK/builds/eager-capacity-cpu-clean01`,
helper `wide_eager_cpu_pilot.py --source SOURCE --build BUILD --out OUT/assessment`.
Two fresh processes: original D2048/T12/V50304/480-node Add/Attention parameters,
B4/physicalB2, two connected windows, one complete FP32 SGD update each;16 ATen
threads/one node worker,512GiB CPU cap/80GiB parameter cap/4GiB head cap,
900s per child/1850s outer. First failure stops this finite pilot. This is CPU
feasibility/RSS calibration concurrent with NPU feasibility, never formal timing
or a full-size CPU oracle. Record terminal status and complete peaks/phase/loss/
output/update checks before claiming a pass. Do not signal the historical task.

Keep the CPU reference independent.
Use affected gates, commit implementation first, then clean qualification and
separate evidence; no unrelated core-owner reruns. Formal heavy timing is serial.

Independent feasibility `wide-attention-b512-extended01` passed/exit0 at
2026-10-03T07:48:06Z; service inactive/dead,empty cgroup,11-card lease completed.
Frozen `TASK/sources/owner-map-clean01`,exact source29effae; build
`TASK/builds/owner-map-consumer-clean01`,original owner-pilot binary. Raw records
`TASK/runs/wide-attention-b512-extended01/assessment/{result.json,original/result.json}`.
Terminal audit passed:
`python TASK/launchers/wide_attention_b512_evidence.py 29effaed064d59b9da930ec3acec810be58b2e1a`.
No other current-task NPU qualification/profile overlapped. Finite CPU build/gate/
calibration overlapped, so do not claim isolated throughput. Release of its lease
unblocks sequential packed-transfer/consumer NPU qualification and profiles.

Next NPU job `packed-transfer-npu-clean01`: exacta785d43 frozen
`TASK/sources/packed-transfer-clean01`,build`packed-transfer-npu-python-clean01`,
`tests/test_transfer.py tests/test_payload_ownership.py --dtype float32`,
`TIDE_TRANSFER_BACKEND=npu TIDE_PAYLOAD_BACKEND=npu`; two-device lease120s,
900s command bound. Expected119 cases; confirm actual terminal count.
Then standalone36,actual-consumer40 and separate profile; do not claim passage
from submission or compilation.

All task outputs are `TASK/runs/NAME/{status.json,task.log}`, units
`tide-execution-flows-NAME`. Historical failures and the protected paused task
remain unchanged.

## Environment and protected state

`TASK=/mi/data2T/zlong/tide-execution-flows`; user-authorized public module
`libtorch-npu/2.10.0-cann9.0.0`; Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
The authorized /opt stack supersedes the old private guide. Preserve module
PYTHONPATH and prepend frozen source/python; use logical devices after leases.
`TASK_QUEUE_ENABLE=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0`. Long jobs use background.slice,
Nice10, two build workers, explicit timeouts and 120s device lease waits.
Last free disk: data157GiB/root11GiB; recheck before large writes.

**Protect deliberately SIGSTOPped historical-cpu-attention-01**: never resume,
stop or clean it. Its historical running record and held old `timing.lock` do
not block current work, which uses `online-measurement.lock`. No signal or cleanup
was issued to it. Historical 1.6438× was a restricted-flow result; archive
`archive/restricted-flow-20260930` at964bf628c67270200dabe55b1bca026bd403cd37.
Historical `build-reverse-gather-python-dev01` metadata inconsistency remains:
`status.py` exits1 for that record, not a current-increment failure. Do not rewrite it.
