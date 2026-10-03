# Current handoff

Updated 2026-10-03. **PAUSED after this implementation commit/push, by the user's
explicit request.** Do not start clean qualification, another increment or a new
goal. The user will cancel the old paused goal and then authorize creation and
execution of its replacement. Repository next-action text never overrides a user
pause; commits, evidence and context compression do not themselves resume work.
Overall F1–F7 remains incomplete. No current-increment job is live.
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

## Contract and priorities for the replacement goal

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
reviewed evidence commit. Commit/push is authorized when execution resumes.
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
Message copies are currently individual. Large-model consumer integration,
per-device fixed-constant caches, memory/locality planning, packed cross-device
transport and FP16 owner qualification remain open. This is not F5/F6 closure.

### Development verification only

All jobs below are terminal; successful jobs exited 0, the retained failure exited
1. Systemd cgroups are empty and NPU lease records are completed. No clean-source
qualification or separate evidence commit has been performed for this increment.
All frozen development snapshots derive from dirty dd78b80, not a clean source claim.

| Job | Observed result |
| --- | --- |
| `mixed-owners-cpu-dev01` | Retained failure: 304 passed / 4 failed; bool/float checkpoint owners were rejected, but error text lacked the required int64 label |
| `mixed-owners-cpu-fix-dev01` | 6 passed: four repaired rejection cases plus two custom Region.initial FP64/FP32 anchors |
| `mixed-owners-cpp-cpu-dev01` | FP64 and FP32 each passed 12 configurations / 24 updates / 48 connected windows |
| `mixed-owners-cpp-npu-dev01` | 36 configurations / 72 updates / 144 connected windows; two NPUs, all mixed presets, nondefault streams |
| `mixed-owners-npu-final-dev01` | 87 Python/native FP32 cases passed, no skips; two NPUs, both schedules, three families, Add/Attention, remapped checkpoint suffix |
| `mixed-owners-preflight-{cpu,npu}-dev01` | One directed case passed on each backend; real two-NPU availability/default-device success and failure paths |
| `mixed-owners-profile-dev01` | Separate bounded two-NPU CANN trace passed; one mixed-C Attention/greedy case, 2 updates / 4 connected windows |

Full CPU/NPU-Python/NPU-standalone builds `build-mixed-owners-*-dev01` passed.
Their source-verified rebuilds/fresh links `build-mixed-owners-*-dev02` also passed;
source/header/options/binary hashes and loader closure are recorded. Earlier Python
CPU 28, Python NPU 42 and native NPU 43 directed cases passed. These overlapping
runs must not be added into a synthetic qualification total.

Snapshots: `TASK/sources/mixed-owners-dev04` retains the CPU failure; dev05 has
repaired error text/initial reference and final owner gates; dev06 adds the profile
checker mode; dev07 adds the final preflight correction. Main-tree implementation,
test and build bytes were compared with frozen dev07 before this commit.
Builds: `TASK/builds/mixed-owners-{cpu,npu-python,npu-standalone}-dev01` (full),
corresponding dev02 (repaired core and fresh links), and `mixed-owners-profile-dev01`
(latest checker mode). Preserve these sources/builds and reuse receipts.

Profile `TASK/runs/mixed-owners-profile-dev01/profile/result.json` records 6,786
operators: 5,174 AI_VECTOR_CORE, 1,017 MIX_AIV, 564 AI_CORE, 31 AI_CPU (23 BOOL/INT64
ScatterElements and 8 INT64 Sort). The diagnostic observed no host CPU fallback;
AiCPU remains accelerator execution. Exact int64 ranking cannot be changed to
float to remove those tasks. The trace includes independent CPU reference,
initialization and assertions: its 2,184 aclrtMemcpy / 8,072 aclrtSetDevice calls
and summed task durations are not pure candidate cost or throughput. No new formal
performance conclusion is claimed. Physical leases were 3,9 for C++/profile/final
preflight and 1,11 for the final Python/native matrix, remapped to logical 0,1.

Raw status/log/commands: `TASK/runs/NAME/{status.json,task.log}` and
`TASK/launchers/NAME.sh`; units `tide-execution-flows-NAME`. Snapshots have adjacent
`.snapshot.json` inventories. Failed runs and old snapshots were not rewritten.

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

After new authorization, priorities are: qualify this owner increment; integrate
actual mixed consumers with per-device constant caches and complete head/loss/
synchronization; add capacity/locality and packed transfers with independent gates;
advance Attention B512 as a separate bounded item; finish original-scale CPU /
screened-mixed / resident comparisons for both schedules and required languages/
families, three fresh processes per recommendation, with separate profiles.
F7 must then audit integration, evidence, support and portable commands. CUDA has
no local hardware: deliver portable source/build/test commands and retain explicit
target-machine-pending status, never claim local execution.

## Exact next entry, only after the user starts the replacement goal

Read this handoff and the execution contract first. The first required action is
clean qualification of this implementation, not another performance diagnosis.
Use a new frozen source and distinct outputs; the existing reuse helper validates
all original production source/header/options inventories before fresh linking.

```bash
TASK=/mi/data2T/zlong/tide-execution-flows
OWNER_COMMIT=$(git log -1 --format=%H -- python/tidegraph/ownership.py)
for variant in cpu npu-python npu-standalone; do
  python "$TASK/launchers/freeze_run.py" \
    --name "build-mixed-owners-$variant-clean01" --snapshot mixed-owners-clean01 \
    --commit "$OWNER_COMMIT" -- timeout --signal=TERM --kill-after=10s 180s \
    '{python}' "$TASK/launchers/mixed_owner_build.py" '{source}' \
    "$TASK/builds/mixed-owners-$variant-dev01" "$TASK/builds/mixed-owners-$variant-clean01"
done
```

Inspect all three terminal builds before submitting dependent gates. If source
verification rejects reuse, investigate the precise mismatch; do not bypass it.
Use the same `freeze_run.py --snapshot mixed-owners-clean01 --commit "$OWNER_COMMIT"`
for all gate commands. Prefix NPU jobs with `--npu --npu-count 2 --max-wait 120`.
Set `TIDE_BUILD_DIR` to the corresponding clean build, with one ATen/BLAS thread:

- CPU (900s): matching Python `-m pytest -q tests/test_payload_ownership.py
  tests/test_placement.py tests/test_library.py tests/test_library_cases.py
  tests/test_checkpoint.py tests/test_checkpoint_ownership.py
  tests/test_coordinate_checkpoints.py tests/test_checkpoint_process.py
  tests/test_region_continuation.py tests/test_fp16_library.py --dtype both`.
- Python/native NPU (600s): `TIDE_PAYLOAD_BACKEND=npu`, matching Python `-m pytest
  -q tests/test_payload_ownership.py --dtype float32`, NPU-Python clean build.
- Standalone (180s CPU / 300s NPU): `tidegraph-payload-ownership-check --device=cpu
  --dtype=float64`, then CPU float32; NPU `--device=npu:0 --dtype=float32`.
- Separate profile (600s): matching Python `scripts/profile_execution_placement.py
  --build-dir "$TASK/builds/mixed-owners-npu-standalone-clean01"
  --output-dir '{out}/profile' --device npu:0 --workload owners`.

Inspect exact source/binary identities, exit codes, expected counts, empty cgroups
and released leases before the separate reviewed evidence commit. These commands
are a future handoff, not authorization to run them during the current pause.

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
