# Current handoff

Updated 2026-09-30T16:59:58.741144+00:00. **ACTIVE — user resumed; tested commits may be pushed.**
Repository `/home/zlong/llm/graph-execution-foundation`, real path
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
HEAD **740fa87**, optional public resident inference committed and pushed.
No pending authorization/pause. No subagents. Reference repositories and ObsidianVault
are read-only. Preserve unrelated older dirty work below. **F1–F7 are not complete.**

## Contract and current priority

[execution-flows.md](execution-flows.md) is the contract; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog. Current user alignment takes precedence over run-ml-experiments:
keep source/input/config/environment identity, raw results/failures, synchronized
complete timing, bounded resources/stops. Trackio must not block implementation.
Training delivery is independent forward/backward/VJP/optimizer/continuation and
complete throughput, not task convergence.

Candidates consume their own inputs/state/parameters; no CPU numerical routing
prepass. General online greedy prefill accepts each family's legal topology/input,
including PDG positive-delay feedback, and may degenerate to streaming. Preserve
int64, stable order, duplicate edges, missing/zero and None/zero gradients. Independent
CPU FP64/FP32 remain anchors. Device residence includes actual online decisions.
Prioritize complete semantic/ownership gates and bounded profiling, public/device
integration, peer progression and resident training, then representative/full-size
performance. PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch; CPU/NPU × streaming/
prefill × inference/complete training. Five presets/fine switches; FP32 main, FP16
separate; three processes for recommendations. CUDA execution is target-machine pending.

## Completed increments and formal evidence

- Placement implementation **d412541** is pushed. All exact-commit qualifications
  now PASSED, terminal0: three builds `build-placement-{cpu,npu,npu-python}-clean01`;
  `placement-cpu-clean01` **8,952 tests**, `placement-ctest-clean01` **10 CTests**;
  `placement-public-npu-clean01` **26** public Python/native cases;
  `placement-standalone-npu-clean01` placement **121 schedules/363 updates**, Read
  precision6/18 plus accelerator parity/gradients/optimizer/checkpoint/non-default
  stream; `placement-profile-clean01` independent trace.
  Reports `docs/evidence/execution-placement-20261001.{json,md}` ready for evidence
  commit; source d412541, not the later resident implementation.
- Formal placement profile:4,079 AIV+862 MIX_AIV+86 AI Core+66 AiCPU. AiCPU:
  INT64 Sort30/1947.50us; BOOL,INT64,BOOL ScatterElements36/3130.44us.38.32% is
  summed device task time, NOT complete wall-time. No host CPU fallback. Keep
  exact integer counts; no lossy FP32 key conversion. This is host-dispatched
  mixed execution, distinct from the Ascend C resident selector.
- Prior qualified device increments are indexed in ROADMAP F4 and their evidence:
  Aggregate f1b7168/evidence fdd2b86; fiber batches c83aec3/evidence7b43f4d;
  event batches26aa09f/evidence28295c1; budgets523b323/evidence0b724a3; key tiling
  7b03614; event/fiber attention, pool, clocks, vector Read, Full, emission/origins.
  All cited immutable logs/failures remain. These are single-device FP32 HARD
  inference profiles, not resident backward, peer progression or full throughput.

## Public resident implementation740fa87

[resident-library.md](resident-library.md): optional `tide::ResidentSession` shared
library/package and separate Python-owned `_tide_resident` binding. GraphRuntime
resident sessions own device state across windows. `advance_device()` returns
borrowed NPU outputs/counts; `result()`/`snapshot()` explicitly export CPU values,
never feed continuation back automatically. CPU/NPU external payloads are packed
once; metadata-only ledger validation plus one packed finite check. Parameters
freeze on construction; normal mutations/replacements refuse until explicit
reconstruction. Checked close and failed-owner poisoning; no hidden autograd.
**Single NPU FP32 HARD inference only.** No resident training, FP16 or multi-card.
Python is a client of C++/CANN, not a separately qualified pure-PyTorch scheduler.

Development terminal passes:
- `build-resident-public-standalone-dev01`: full standalone build/four CTests.
- `build-resident-public-python-dev02`: full Python-owned build/loader, no standalone
  SDK dependency. Core builds remain `placement-{npu,npu-python}-clean01` at d412541;
  C++ source hash is unchanged by740fa87.
- `build-resident-public-{python,standalone}-dev03`: isolated new-directory relinks
  replacing only content_input.cpp; all other production hashes/parent binaries
  checked. `TASK/launchers/resident_input_relink.py` records derivation/commands in
  each control-build.json. Never modify the old builds/snapshots. Development only.
- `resident-public-standalone-dev03`: resident64 windows plus window/content gates.
- `resident-profile-dev03`:10,534 AIV+16 MIX_AIV,no AiCPU/fallback; not throughput.
- `resident-host-cpu-dev02`:125 CPU tests,20 optional-device skips; dev03 loader/
  config2 tests passed,21 optional-device skips.
- `resident-public-python-dev06`:24 NPU/CPU-config cases; EMA/event/fiber attention,
  three families/two schedules,CPU/NPU inputs,non-default streams,exported-cut
  restore/schedule switch,parameters,capacity poisoning,lean no-export,invalid input.
- `build-resident-consumer-dev01` + `resident-consumer-dev01`: installed public-header
  C++ client (prefix with spaces),3 live feedback windows. Uses original dev01
  standalone backend; final clean consumer must use final backend.

Retained failures: Python build dev01 referenced a disabled checker and selected
wrong Python; fixed CMake guard/explicit interpreter. Python gate dev02 reached
three correct windows then failed save because CPU state was validated against
NPU model; fixed CPU structural view preserving aliases without a duplicate NPU
model. Dev03 test used node ranks instead of positive region ranks (2 passed);
dev04 test wrongly requested fiber GQA (12 passed); corrected fixtures, no formula
or tolerance changes. All failed logs/snapshots remain failed.

## Running qualification and next exact actions

Two full clean component builds from **resident-public-clean01 at740fa87**:
- `build-resident-public-python-clean01` -> `builds/resident-public-python-clean01`.
- `build-resident-public-standalone-clean01` -> `builds/resident-public-standalone-clean01`.
Each1800s/jobs2,no NPU lease; inspect real terminal status before submitting gates.
Both use matching immutable d412541 placement cores; C++ hash matches.

A following Python-only increment is uncommitted: `ResidentSession.load/reset`,
`resident_checkpoint.py`, tests and docs. Disk restore validates schema, parameter
aliases and CPU complete-cut semantics before closing/mutating live state. Device
construction failure after preflight leaves session closed. Invalid records leave
it unchanged. Tests now restore continuation from the file itself, not merely
exported memory plus a weight file. New run snapshot **resident-restore-dev01**:
- `resident-restore-python-dev01` PASSED **25 cases**,terminal0,dev03 plugin.
- `resident-restore-host-dev01`:125+ host-surface regression,inspect terminal.
After its pass, commit/push that Python-only increment. It does not change the
C++ component hash; the clean740fa87 compiled backend is reusable with provenance.

Next:
1. Commit/push placement evidence separately (d412541); keep resident uncommitted
   files out of that evidence commit.
2. Verify/commit the restore increment. Freeze its exact commit for final Python
   qualification; reuse unchanged compiled backend fingerprints transparently.
3. Once clean component builds finish, run full standalone gate (33 cells):
   `scripts/verify_device_control.py --build-dir BUILD --output-dir NEW --device npu:0`.
   Run Python `tests/test_resident_library.py --dtype float32` with TIDE_BUILD_DIR
   set to matching NPU Python core,TIDE_RESIDENT_DEVICE=npu:0 and
   TIDE_RESIDENT_LIBRARY=clean Python backend. One lease each,900s,queue120s.
4. Independent profile: `scripts/profile_device_control.py --build-dir CLEAN_STANDALONE
   --output-dir NEW --device npu:0 --check resident` (one lease,900s).
5. `scripts/build_resident_consumer.py --core-build CLEAN_NPU_CORE --resident-build
   CLEAN_STANDALONE --output-dir NEW` (600s,jobs2,no lease); then its installed
   consumer/resident-consumer --device=npu:0 --dtype=float32 (one lease,300s).
6. Full CPU scripts/verify.py on the final Python commit using unchanged
   placement-cpu-clean01 native build (FP32/FP64,bound2400s). Previous full clean
   placement gate took1698.47s; do not start redundant concurrent full CPU gates.
7. All required terminals/output audits must pass before resident evidence commit.
   Then continue remaining F4/F5 modules,FP16,peer progression,resident backward/
   VJP/optimizer and F6 representative/full-size matrix. Overall delivery incomplete.

## Preserved older work and timing

Do not stage/clean old dirty `scripts/build_accelerator_scale.py`,
`tools/accelerator_scale/CMakeLists.txt`, `bounded{.h,_export.cpp,_program.cpp,
_select.cpp,_update.cpp}`, `peer_transport.cpp`, `resident.cpp`,
`scripts/benchmark_execution_flow.py`, `scripts/verify_execution_flows.py`,
`tests/test_flow_semantics.py`, and `tools/accelerator_scale/flow_*`.
They are restricted DAG/rank-aligned consumers, not general-online delivery.
flow-dev05 CPU24/NPU18 FP32 passed before FP16 Add-gradient failure; dev06 builds
have no gates. Peer CPU8/NPU16 passed separately. Keep strict-FP32 near-zero
failures/reproducers and historical snapshots; never silently relax tolerance.

Historical `tide-execution-flows-historical-cpu-attention-01.service` is deliberately
SIGSTOP. TASK/runs/historical-cpu-attention-01/pause.json overrides running state.
It holds host memory and TASK/timing.lock. Do not blindly resume/stop. Resolve this
interrupted timing deliberately before formal timing. Add historical full training
CPU78.793172/NPU4 47.932888 ms/token means NPU throughput1.6438x faster; it does not
qualify the new resident backend. Attention still has no valid full CPU training ratio.

## Environment and commands

TASK=`/mi/data2T/zlong/tide-execution-flows`. Unit `tide-execution-flows-NAME.service`
in background.slice; artifact link `artifacts/execution-flows-NAME`; raw records
`TASK/runs/NAME/{status.json,task.log}` and verified/profile result.json. Check both
terminal exit and expected outputs. Build jobs2; CPU/BLAS threads1. Last disk check
280GB free data/25GB root; recheck before large writes.

Module `libtorch-npu/2.10.0-cann9.0.0`; Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
User-authorized public /opt stack overrides dated personal-anaconda defaults.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0;retain module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392;16 chips64GiB. Cooperative leases select
physical cards; programs use logical npu:0. Never link standalone SDK into torch_npu
wheel processes. No fixed physical device IDs.

```
python TASK/launchers/freeze_run.py --name NAME --snapshot SNAPSHOT [--commit SHA] [--npu --max-wait 120] -- timeout --signal=TERM --kill-after=10s 900s '{python}' scripts/COMMAND ...
```
Placeholders {python},{base},{source},{out}. `norm32_after_core.py` is a600s bounded
dependency wrapper,placed before the queue helper;no card held waiting for build.
Use scripts/durable_records.py atomic fsync/read-back. Re-entry: git status --short
--branch;python scripts/status.py;read STATUS and actual job terminals. Profile
summaries[].engines is a dict,operators_by_type a list,inputs has CSV hashes;do not
print entire raw task arrays. Old artifacts cited by evidence/failures stay intact.
