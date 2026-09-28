# Current handoff

Updated: 2026-09-28 (Asia/Shanghai). Branch: graph-execution-foundation.

## Completed authorized work

L1-L4 reusable experiment library is complete. All implementation increments
were committed and pushed; this evidence-only update closes the task. No live
job or required implementation remains. Reference repositories and
ObsidianVault stayed read-only; no subagents or new platform support.

[Library contract](library.md) owns installation, GraphConfig/GraphRuntime/
Session, external inputs, checkpoints and qualification. New experiment repos
pin the library version/commit, own their data/head/loss/optimizer/controller
and artifacts, and consume installed code without writing into Tide source.
Package 0.2.0, config schema 1, continuation 5, native identity 13.

## Completed evidence

[Reusable-library qualification](evidence/library-foundation.md):

- clean c08cc90: fresh 109-target build and installed Python/native/CMake clients.
- clean ff708a1: 8621 CPU FP32/FP64 tests in 1531.34 s; 22 complex cells;
  6 installed apps, 3 installed config gates, 1 installed C++ client passed.
- clean aa03a03: validator-only trace-disabled VJP hardening; 18 qualifier tests
  in 95.87 s and all installed consumers passed. Executors are unchanged from
  ff708a1; the full 8621 count belongs to that earlier exact source.
- Complex gates retain full graph records at D4/B1/T2/three optimizer updates:
  active 64-node/768-edge TimedDAG/Settle, active 32-node feedback, exact benchmark
  P01/P02/T02/S01/A02. P02 has 8192 total nodes, 4 observed and 8188 dormant.
- CPU aarch64, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, ABI 1;
  ATen/OMP/BLAS threads 1, build parallelism 2.
  Static audit: 0 errors, 45 reviewed warning leads.

All terminal, exit 0:
`tide-library-consumption-20260928-a.service`,
`tide-library-release-20260928-a.service`,
`tide-library-validator-20260928-a.service` (finished 03:03:27 UTC).
Corresponding artifacts directories contain status.json/task.log and nested
qualification or consumer manifests. Old installed -001/-002 failures remain.
No running task is being handed off.

Verified native build: /var/tmp/zlong-graph-execution-foundation/library-l3-build.
Latest installed wheel:
artifacts/library-validator-20260928-a/consumer/wheels/tidegraph-0.2.0-py3-none-any.whl.
Consumers may use that matching build or build a new target directory from the
pinned source. The older default build/ and historical artifacts were preserved.
Read-only source snapshots remain under /var/tmp/zlong-graph-execution-foundation/.

## Retained earlier scope

Six-stage foundation and E1-E5 remain complete; see
[evidence/foundation-final.md](evidence/foundation-final.md) and
[evidence/cross-family-qualification.md](evidence/cross-family-qualification.md).
Extended narrow Settle: 46912 nodes, 8496773056 parameters, 1034.865380 s under 1800 s;
original 900 s failure retained. Four active layers plus dormant nodes; completion
is not full-scale observable/gradient equivalence.
[Python NPU](evidence/npu-python-20260925.md) retains its ten finite FP32 eager
cases at clean 7811418. New public API NPU qualification is absent; native NPU
remains unsupported without a standalone SDK. State/Read replay remains.

## Next use / re-entry

Run git status --short --branch and python scripts/status.py. There is no new
execution task queued. Future experiments use docs/library.md and a pinned
version; qualify each changed config/backend/shape explicitly. Preserve cited
artifacts and historical failures. Implementation and evidence commits remain
separate; commit and immediately push each future authorized tested increment.
