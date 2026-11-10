# Implementation inventory and public-boundary review — 2026-11-10

This is a source/contract audit of baseline
`4d777bc9bdb622a16f0585ae642f1536475331ba`, not a new performance or accelerator
qualification. The three resulting fixes have targeted development checks;
immutable implementation/build qualification is recorded separately after integration.
The mathematical source remains the revision in [upstream.json](../upstream.json).

## Review scope

[`review_inventory.py`](../../scripts/review_inventory.py) reads immutable Git blobs,
including exact file SHA256 values. It does not accidentally audit dirty/untracked
source. Python functions, async functions, methods, nested functions and lambdas
are extracted with the standard AST; generated/dynamic callables are excluded.
The optional C++ parser uses Clang AST with explicitly supplied flags and records
its version/command. Missing compiler, timeout, failed/partial parsing and output
limits stay visible; inactive preprocessor branches are outside that parse.
No regex scan is presented as a complete C++ symbol inventory.

The baseline contains 2,031 selected source/build/test files. Python AST parsing
enumerates 4,540 functions in 899 files. The other 1,132 files have no configured
parse in this run, including C++ files; zero extracted symbols does not mean zero
functions. The source manifest hash is
`d4d9e643a0420adb470e15aabe502f4cfa79e7c1b34578726d4f63098551ad1c`.

The explicit [review claims](implementation-audit-20261110-reviews.json) cover
25 files and their 128 extracted functions. The remaining 2,006 files and 4,412
extracted functions are **not reviewed by this audit**. A file marked `reviewed`
means its source and stated boundary were examined; it does not certify every
formula, configuration, device or scale. Claims require exact file hashes,
reviewer, scope and evidence; stale source/symbol claims fail rather than silently
carrying review status forward.

| Reviewed layer | Files and scope |
| --- | --- |
| Public runtime/configuration | `library`, `config`, `execution_options`, `runtime`, `session`: dispatch, explicit device/owner boundaries, lifecycle and identities |
| Checkpoint | `checkpoint`, `checkpoint_io`, `checkpoint_values`, `checkpoint_ownership`: actual raw values, named optimizer owners, preflight and publication |
| Planning and task execution | `planning/execution`, `planning/record_capacity`, `experiment/config`, `eager`, `record_policy`, `tasks`, `task_identity`: selection, return requirements, capacities and resume identities |
| Experiment persistence/evidence | `experiment/checkpoint`, `training_layout`, `identity`, `records`: actual selected execution, canonical owners, installed bytes and records |
| Build/job helpers | `scripts/source_identity`, `durable_records`, `build_resources`, `build`, `job`: source/build identity, resource entry, durable terminal state |

These groups are navigation aids. The JSON contains the exact source file list
and review scope; no second live backlog is created. Current remaining work stays
in [ROADMAP](../ROADMAP.md), current integration in [STATUS](../STATUS.md).

## Findings and resulting changes

| Finding at the baseline | Mechanism and change | Independent check / coverage boundary |
| --- | --- | --- |
| `Session.load` reads and decodes the same checkpoint twice | Session validates one read and then `checkpoint.load` reads again before mutation. `load` now reads/decodes once and invokes a continuation validator before weights/optimizer change; Session checks its batch and Settle boundary there. | `test_session_restore`: exactly one read/decode, malformed ownership/state/weights/schema and incompatible batch/Settle cut preserve live owners; successful next-window/optimizer restore. Existing optimizer ownership and public session tests retain their independent constraints. No large checkpoint timing claimed. |
| All plugins force complete event records | `needs_trace` treats plugin presence as complete regardless of its loss inputs, blocking the existing boundary-only sequence VJP and capacity reservation. New `task.plugin.observations=outputs|complete` permits explicit output-only plugins; absent declaration remains complete. `selected-boundary-v2` records the policy and old frozen v1 retains original plugin behavior. | `test_task_observations`: actual PyTorch/native untraced execution, independent reference training, head gradients, complete diagnostics, exact resumed next window, rejected cross-observation resume and allocation-free capacity identity. A plugin's declaration is its contract; arbitrary plugin code is not statically proven output-only. |
| `GraphRuntime.manifest` hashes only top-level Python files | Changes inside weighted/planning/experiment subpackages were absent from `package_sha256`. A lightweight `package_content.content_identity` now supplies both runtime and experiment identities, including relative names and recursive installed files. | `test_package_content`: independent SHA256/JSON formula, creation-order/location independence, nested implementation/resource changes, cache exclusion and both entry points. `tide-runtime-v2` makes the wider hash scope explicit. Experiment-lock content hashing is unchanged; installed bytes still do not prove Git/build provenance. |
| Public API documentation still described the old default schedules | `ExecutionOptions.resolve` already selects component windows unless `prefill=False`; `library.md` said PDG streaming and other families frontier. | The documentation now describes actual dispatch and points to the component-window contract. No new scheduler is claimed here. |

The checkpoint change removes duplicate work and the two-read validation gap;
it does not promise rollback from arbitrary custom model/optimizer load hooks
with external side effects. The existing optimizer preflight boundary remains.
The observation change preserves exact raw continuation/checkpoint state; omitting
event logs is not permission to decode missing results through a reference recurrence.

## Architecture observations to retain

1. `GraphConfig` exposes the generic module catalog while weighted planning and
   experiment APIs describe the core-5 candidate catalog. They are distinct
   interfaces; navigation must identify the appropriate configuration domain.
   Sharing a scheduler does not make their local programs interchangeable.
2. `ExecutionSpec.record/from_record` combines current policy resolution with
   historical omitted-field identities. A future schema redesign should move
   historical migration into an explicit decoder and keep the current resolved
   form simple. Compatibility work should not spread into local compute paths.
3. `SGDLayout` correctly decodes actual owner values/connectivity and retains the
   complete checkpoint for exact comparisons. Its legacy name rewrites and
   template optimizer construction are a seam for a canonical owner schema;
   this audit supplies no timing evidence that this is a throughput bottleneck.
4. `event_reservations` reduces only the audited host, ranked-local, inference
   domain. Training/resident/other schedules retain complete reservations. The
   estimate does not measure total device peak or vendor workspace.
5. `exact_next` correctly compares the actual selected execution and decoded raw
   state before/after the next window. Keep this distinction from independent
   complete-record correctness; do not switch to a traced algorithm to certify
   an untraced run's exact resume.
6. Build/job status, installed content, source provenance, dependency metadata,
   numerical qualification and performance remain separate evidence. The
   existing identity and durable-record helpers mostly maintain these boundaries.

## Reproduction and development checks

Generate a fresh inventory from a fixed commit; the output path must not exist:

```bash
python scripts/review_inventory.py --revision 4d777bc9bdb622a16f0585ae642f1536475331ba \
  --reviews docs/evidence/implementation-audit-20261110-reviews.json \
  --output artifacts/implementation-audit/reproduced.json
```

The full generated inventory belongs under ignored `artifacts/`; this reviewed
report and small hash-bound claims are sufficient to regenerate it from Git.
For a bounded C++ scope use `--include 'cpp/path/*' --cpp clang` and explicit
`--cpp-arg` include/define flags matching its intended translation unit. Headers
requiring an enclosing TU must not be called completely parsed after an isolated
syntax failure. This audit's Clang fixture checks enumeration/filtering, not the
repository's entire C++ build.

Short CPU development checks used the existing project-private Torch environment,
`TORCH_DEVICE_BACKEND_AUTOLOAD=0`, source `PYTHONPATH=python` and the prior matching
native library. No C++ source is changed by these fixes; no NPU was initialized.

| Check group | Observed development result |
| --- | --- |
| Inventory extraction, fixed source and explicit review claims | 6 passed, 0.44 s |
| Session restore, task observations, selected capacity, existing task/module and optimizer ownership | 107 passed, 27.93 s |
| Public library, checkpoints/publication, experiment config and selected execution | 67 passed, 14.25 s |
| Recursive content hash, installed identity, public library and inventory | 39 passed, 9.07 s |
| Final inventory/content/session/task-observation/installed-identity regression after integration of these fixes | 33 passed, 8.31 s |

Groups overlap; counts must not be added as distinct tests. These are working-tree
development checks, not a frozen implementation qualification or performance
measurement. They validate public boundary behavior and tooling. The plugin
reference owns its schedule but reuses declared local functions; independent
mathematical evidence remains in their module tests and semantic map.
