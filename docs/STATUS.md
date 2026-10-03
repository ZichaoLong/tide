# Current handoff

Updated 2026-10-03. **ACTIVE: the user cancelled the old goal and explicitly
authorized creation/execution of its replacement.** The new goal is active.
The previous post-commit pause is superseded. Repository next-action text never
overrides a later user pause; commits/evidence/context compression do not stop
or resume work by themselves. Overall F1–F7 remains incomplete.
Eager owner implementation 55c396073afa2a78376ab84d3b29e5e192850f7e is now
qualified below. Actual multi-device consumer integration has passed development checks and is
being committed for clean qualification; bounded jobs are recorded below.
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
Message copies are currently individual. Actual consumer integration, per-device
fixed-constant caches and static learned-parameter/locality planning are implemented
below. Total-memory admission, packed transport, scale performance and FP16 owner
qualification remain open. This is not F5/F6 closure.

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

**Attention B512 complete training remains pending.** The fixed-map original-width
B4/physical B1×4 pilot on clean29effae completed one update in 50.418376377s;
its 1.15 phase forecast was 7036.453031774s > 3000s, so B512 was not entered.
[Retained diagnosis/refusal](evidence/original-width-attention-owner-diagnostic-20261003.md).
The latest qualified KV-journal admission correction is af137df
([evidence](evidence/consumer-journal-capacity-20261003.md)); accounting alone did
not execute B512. Keep these failures and memory/calibration guards intact.

Unexecuted task-local `finite_ranked_horizon.py` and `wide_attention_horizon_pilot.py`
are unqualified drafts. Review their claimed 48-row bound and physical B4 proposal
before use; never generalize a fixture-specific bound into the online scheduler.
Do not modify audited `wide_attention_owner_pilot.py`; use a new helper/config.

Current authorized priorities are: integrate
actual mixed consumers with per-device constant caches and complete head/loss/
synchronization; add capacity/locality and packed transfers with independent gates;
advance Attention B512 as a separate bounded item; finish original-scale CPU /
screened-mixed / resident comparisons for both schedules and required languages/
families, three fresh processes per recommendation, with separate profiles.
F7 must then audit integration, evidence, support and portable commands. CUDA has
no local hardware: deliver portable source/build/test commands and retain explicit
target-machine-pending status, never claim local execution.

## Current consumer increment and next qualification

Actual consumer ownership is implemented and development-tested, awaiting its
own clean qualification. Python/native/standalone independently initialize the
same named leaves, use per-device fixed-constant caches, gather outputs to the
head owner, synchronize and observe every device, and agree on finite gradients
before updating. Static learned-parameter/locality planning or explicit maps keep
node zero/boundaries/head/embedding on owner zero. Total peak admission and packed
cross-device transfer remain separate work; resident aliases are preserved.
Resolve this implementation after commit with
`git log -1 --format=%H -- tools/online_bench/eager_placement.py`.

Terminal development jobs, no skips (deselection is explicit):
`mixed-consumer-python-cpu-dev01`9 planning/rejection tests;
`mixed-consumer-cpu-dev01`64 affected CPU tests;
`mixed-consumer-python-npu-dev01`13 Python/native training/constant cases;
`mixed-consumer-cpp-npu-dev01`6 standalone training cases;
`mixed-consumer-options-npu-dev01`27 inference/CLI/native transport interactions.
Final C++ resident-alias preservation uses snapshot`mixed-consumer-dev03`:
`build-mixed-consumer-{cpu,npu}-dev02`both passed/exit0;
`mixed-consumer-cpp-cpu-dev02`24 FP64/FP32 standalone comparisons and
`mixed-consumer-cpp-npu-dev02`6 two-device full-training comparisons passed/exit0.
Their service cgroups are empty, leases complete and all1504 frozen hashes match.
The separate earlier development profile`mixed-consumer-profile-dev01`passed
(14090 operators,128 AiCPU,no observed host fallback); it is not the final binary
or a throughput measurement. No new modification is qualified by owner-only evidence.

After this implementation commit, freeze that exact clean revision as
`TASK/sources/mixed-consumer-clean01`. Launch two600s builds with
`TASK/launchers/build_eager_consumer.py --backend cpu|npu --name mixed-consumer-{cpu,npu}-clean01`.
This installs byte-verified qualified55c3960 core into a fresh package, builds the
current public-header-only consumer and checks standalone loader closure.
Then run the affected CPU73 (existing consumer/host controls plus9 static-owner
checks) and NPU40 actual-owner tests using those fresh binaries; use at most900s
per gate, two-card lease120s. Separately run
`TASK/launchers/profile_eager_consumer.py --source SOURCE --build NPU_BUILD --out OUT/profile`
with a300s child bound. Every NPU child runs from `{out}` with absolute source paths.
Audit fixed source/binary hashes, counts, terminal units/leases and trace placement,
then commit evidence separately and push. Continue with calibrated eager capacity/
packed transfers and a separately budgeted Attention B512 feasibility run; do not
pause after this increment. Formal heavy timing remains serial.

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
