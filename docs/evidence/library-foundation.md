# Tide 0.2 reusable library qualification

Completed 2026-09-28, aarch64 CPU, Python 3.11.15, Torch/LibTorch 2.10.0+cpu,
GNU 10.3.1, C++17, CXX11 ABI enabled. ATen/OMP/BLAS threads 1; build parallelism 2. No new NPU,
CUDA, x86_64 or performance qualification is claimed by this report.

## Sources and terminal gates

| Clean source | Scope | Terminal result |
| --- | --- | --- |
| `c08cc90bf9234954b1e337067255eccd06d9999a` | Fresh 109-target build, installed Python/native and CMake consumers | passed, exit 0 |
| `ff708a101e243240c94c6cd91fc685cd5218bad5` | Full CPU regression, 22 complex configuration cells, installed consumers | passed, exit 0; 8621 tests in 1531.34 s |
| `aa03a03ffe5541720b02994c89f662e509cafe2d` | Validator-only trace-disabled VJP hardening, 18 qualifier tests, installed consumers | passed, exit 0; 18 tests in 95.87 s |

The only production-code difference from ff708a1 to aa03a03 is in
`python/tidegraph/qualification.py`: an additional trace-disabled gradient
comparison. Graph representation, models, kernels, scheduling, sessions and
checkpoints are unchanged. The completed full executor regression is retained
at ff708a1; affected qualifier tests and installed consumers were rerun at the
later clean commit. The 8621 count is not represented as a run at aa03a03.

All three jobs ran from separate read-only worktrees under detached user services
in background.slice, Nice=10, KillMode=control-group. Terminal records/logs:

- `artifacts/library-consumption-20260928-a/{status.json,task.log}`;
  `consumer/{result.json,commands.log}`.
- `artifacts/library-release-20260928-a/{status.json,task.log}`;
  `qualification/{result.json,cpu/result.json,cpu/tests.log,complex/result.json,consumer/result.json}`.
- `artifacts/library-validator-20260928-a/{status.json,task.log}`;
  `consumer/{result.json,commands.log}`.

Unit names have the corresponding `tide-` prefix and `.service` suffix. The full
gate finished 02:54:59 UTC. The validator follow-up finished
2026-09-28T03:03:27.316844+00:00 (UTC).
No gate is live. Exact argv, working directories, clean-source records and
per-case inputs are retained in these artifacts.

Reproduce full acceptance from a clean checkout with a fresh output directory:

```sh
python scripts/qualify_library.py --build-dir BUILD --output-dir NEW
```

`--reuse-build` validates C++/CMake source and every recorded binary hash.
The reused build originated at c08cc90 and matches both later source trees:

- C++/CMake SHA256: `497bfd7278fcafcfd5f9b045a338c3a27d8dea01f3577532712669ba27fd05d7`.
- Native adapter SHA256: `399b7181cb45037d6bc66dbe8922cae830b769140fc64df9f6066ef8903c3e87`.
- Final installed generic wheel SHA256: `a9cbd6d7162247bd13f62a0464be2a6cd23db3e847925b86aa5f6a3496062a29`.

## Public dependency contract

[Library guide](../library.md) owns the public usage contract. Config schema 1,
package 0.2.0, continuation schema 5 and native structural identity 13 are distinct
versions. GraphConfig supports explicit graph records or topology factories,
per-node/per-region modules and explicit initialization. GraphRuntime owns the
model/executor; Session accepts caller data and preserves continuation. The
experiment owns task data, heads, losses, optimizer control and whole-task state.
Settle preserves encoded session state and projects body diagnostics, avoiding
loss of boundary state during resume or policy changes.

The generic wheel contains Python code. A matching native adapter is loaded
explicitly with its build manifest; no sys.path mutation or hidden compilation.
CMake installs/exports `tide::tidegraph` with matching Torch/Threads, ABI flags,
headers and C++17 requirements. External consumers need no source/scripts/tests
import path. Source exporters now include the CMake package templates.

## Complex configuration matrix

All graph records were retained. Validation used D4/B1/T2, three AdamW updates,
seeded inputs and explicit scalar-weight initialization (.25 for mixed cases,
.8 for benchmark cases). Each row passed FP64 and FP32. The three active mixed
rows each use Python and native candidates (12 cells); the five benchmark rows
use native against the independent Python oracle (10 cells).

| Configuration | Nodes | Edges | Regions | Observed nodes | Pending at end |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fully active mixed TimedDAG | 64 | 768 | 4 | 64 | 0 |
| Fully active mixed Settle | 64 | 768 | 4 | 64 | 0 |
| Mixed feedback ring/hub, varied delays and parallel edges | 32 | 64 | 32 | 32 | 79 |
| Exact benchmark P01 | 128 | 4 | 128 | 4 | 2 |
| Exact benchmark P02 | 8192 | 4 | 8192 | 4 | 2 |
| Exact benchmark T02 | 16 | 48 | 4 | 16 | 0 |
| Exact benchmark S01 | 8 | 12 | 4 | 8 | 0 |
| Exact benchmark A02 | 4 | 4 | 3 | 4 | 0 |

The mixed graphs contain EMA, Linear, Gated Delta, DeltaRule, SSM, event
Attention, repeat Add and same-fiber Attention, tanh/SwiGLU, and multiple
Aggregate profiles. The 32-node graph exercises in-flight message persistence.
P02 preserves 8188 unobserved nodes: its 32780 gradient leaves include 32752
explicitly disconnected leaves in the combined diagnostic root. This is not
8192 fully active nodes, nor a full tensor-scale or arbitrary-model guarantee.

Every cell compares complete output/state/slots/history/routes/trace/pending/
ledger, input and parameter VJPs with None connectivity and isolated roots,
whole/chunk results and VJPs, three optimizer steps, weights/optimizer state,
and fresh-process checkpoint continuation. Settle additionally checks its
independent direct scalar schedule. Trajectory diagnostics are trace-enabled;
requested trace-disabled output/state values are separately checked. The later
validator hardening also checks trace-disabled VJPs and rejects a deliberately
wrong backward with identical forward values. Native compact/deferred release,
packed source/Next and batched Full/Aggregate paths passed the added checks.

Floating tolerances stayed at FP64 atol=1e-10/rtol=1e-8 and FP32
atol=1e-6/rtol=1e-5. Discrete record identities/metadata are exact; nonfinite values
fail. Diagnostic squared losses are divided by participating scalar count;
isolated roots use means. Per-case report.json and input.pt record both
requested/effective configs, topology identity, actual inputs, policy, package/
binary identity, coverage and comparisons. Reduced widths and test losses do
not certify original benchmark initialization, tensor dimensions or convergence.

## Installed consumption and retained failures

At both ff708a1 and aa03a03, the source-independent consumption gate passed:
6 applications (three families × Python/native), 3 installed CLI qualifications
including fresh-process resume, and 1 installed C++ client checking FP32/FP64
forward, chunking and backward. The wheel was built in staging, installed into
a new environment retaining the existing Torch stack, and imported from that
environment. The native .so/manifest were relocated, and the C++ application
used only the installed CMake package. All app outputs lived outside source.
The example is a three-update integration smoke, not a convergence experiment.

Earlier development failures remain in `artifacts/library-installed-directed-001`
and `-002`. They exposed FP32 cancellation in unnormalized sum-loss gradients:
absolute errors 7.15e-6 and 7.25e-5 at the reported leaves. Normalizing diagnostic
losses made the same installed configurations pass the original strict default
thresholds in `-003`; final acceptance uses those defaults. Explicit user
atol/rtol options are recorded, not silently applied. The invalid initial
same-fiber/mixed-Aggregate probe was corrected to the module's declared sum
Aggregate capability before execution; no successful result is claimed for it.

NPU evidence remains the separately scoped [Python eager report](npu-python-20260925.md).
The new public API has no additional NPU qualification. Native NPU requires a
standalone SDK and remains unsupported. Custom external programs, task-specific
heads/losses, full application RNG/data-controller resume, new topologies/shapes
and other devices need their own explicitly scoped validation.
