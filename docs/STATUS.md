# Current handoff

Updated 2026-10-10 (Asia/Shanghai). **New batched execution / CUDA resident stage
active (G1–G5); State/Read development gates passed; implementation commit and clean qualification next.** The current prompt supersedes
the previous local-delivery closure and any historical pause-after-commit note.
Read [execution-flows §11](execution-flows.md) and [ROADMAP G1–G5](ROADMAP.md).
Authorized: repository changes, necessary builds/tests, bounded experiments,
commit and push. Continue until the locally achievable stage is closed. No subagents.
Selector/new upstream semantics remain deferred. Reference repositories are read-only.

## Recovered source and environment

Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Initial HEAD exactly matches handoff
`c0ce4ee5d1ac92465e57734de533c33dba7a9445`, initially clean and in sync with origin.
`TASK=/mi/data2T/zlong/tide-execution-flows` owns large new artifacts.
Initial disk check: root ~6.8GiB available (below old 8GiB experiment reserve),
data ~130GiB. Recheck before large writes; do not weaken old admission records.

Public NPU entry required by this prompt: `libtorch-npu/2.10.0-cann9.0.0`;
Python `/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
This explicitly supersedes older private-stack guide paths; actual toolchain must
be checked before use. GPU tools: `/mi/data2T/zlong/gpu-toolchains`, private module
`~/privatemodules/torch-cuda/2.10.0-cu128`; no actual GPU expected. Drivers unchanged.
Long work: fixed-source detached background.slice service, Nice10, bounded threads,
KillMode=control-group, durable job/log/status. Use scripts.durable_records for handoff.

## Next work and live records

Stage-start documents committed/pushed as `05af8f8`. G1 native/Python State
batch VJP, Read VJP/grouped finite checks, ordered Add sample batching and counters
are ready for the G1 implementation commit. Frozen `g1-state-dev02` built and passed **1538** directed CPU
FP64/FP32 tests; unit terminal success, MainPID0, empty cgroup. Its predecessor
`g1-state-dev01` failed compilation; records retained. Earlier Read gate passed193.
Read input mutation/version checking was then added; final development gate
`g1-state-dev03` uses only the G1 changes, isolated from unfinished CUDA work.

Prepared job `tide-g1-state-dev03.service`, source `TASK/sources/g1-state-dev03`,
build `TASK/builds/g1-state-dev03`, records `TASK/runs/g1-state-dev03`.
Command: `scripts/job.py --output-dir RUN -- python scripts/develop.py
--output-dir RUN --build-dir BUILD --jobs 8 tests/test_read_batched_vjp.py
 tests/test_state_batched_vjp.py tests/test_isolated_gradients.py`.
Budget45min, CPUQuota1000%, MemoryMax16G, TasksMax192, OMP/BLAS1; no accelerator.
Terminal result: native build and324 directed checks passed, including final Read
input version checks. Service exit0/MainPID0/empty cgroup confirmed. Commit G1,
then run full CPU and related NPU qualification from that clean fixed commit;
commit evidence separately. G1 does not contain the in-progress resident changes.

Independent uncommitted CUDA work under `tools/device_online`: shared backend
interfaces and launch facades, CUDA conditional graph control, numerical and peer
adapters, build/loader/target contracts. It is incomplete and uncompiled. Do not
include it in the G1 commit or claim any CUDA execution on this host.

## Preserved previous-stage disposition

F1–F5 implemented/qualified only for their declared prior CPU/NPU source profiles.
F6/F7 local selection review closed at c0ce4ee5; this does not qualify new changes.
Original matrix: 61/120 bound FP32 accepted; 8 historical unbound observations;
59 missing bound cases and two later failed follow-ups preserved unchanged.
CPU Attention full-size training follow-up reached24400s without a complete result;
Settle resident Attention inference follow-up failed11-card admission before execution.
Neither produced an accepted timing; no automatic retry or old manager restart.
All old current-task services terminal/empty. Detailed paths and terminal review:
[evidence/selection-terminal-20261010.md](evidence/selection-terminal-20261010.md),
[selection advice](evidence/selection-review-20261009.md). The old pending-measurement
notes are historical, not an execution queue for this stage.

Prior qualified sources: CPU78e9df6 (9458 tests,654 scoped skips,12 CTests),
exact comparator e69b3bd (121 CPU/49 NPU), NPU e69 (49 eager+93 resident),
CUDA-linked aarch64 host187+7 relocation and updated108 consumer checks.
Actual NVIDIA/x86/other versions remain target-pending. Keep cited hashed builds,
raw profiles, packets and immutable worktrees. Current target commands:
[eager-target-validation](eager-target-validation.md).

Preserve the original-scale strict near-tie failure: CPU246 ahead by one FP32 ULP,
resident245/246 tie and choose245; proposal error7.7039e-6,events2325/2327.
Do not relax discrete comparison or claim universal strict cross-device equivalence.
[Retained witness](evidence/original-add-route-witness-20261004.md).
Full/Aggregate batched VJPs already have active consumers; resident reverse already
uses explicit device VJPs. Python, Python-owned native and standalone LibTorch
identities, CPU FP64/FP32 and accelerator FP32/FP16 contracts stay separate.

## Protected history

**Never resume, stop, signal or clean historical-cpu-attention-01 / worker2686919.**
Its historical running record is unrelated to this stage and is left untouched.
`scripts/status.py` currently exits1 for the known malformed historical
build-reverse-gather-python-dev01 metadata. Preserve it; it is not a current blocker.
Do not write project files in ObsidianVault or change reference repositories.
