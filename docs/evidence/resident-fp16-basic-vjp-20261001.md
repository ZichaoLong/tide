# FP16 state and basic Full adjoints

Implementation: `d2a1afccac85c1e53afe45968d6ff0d5f256ce09`.
Verified 2026-10-01 on aarch64 Ascend910_9392, Torch/TorchNPU2.10.0,
CANN9.0.0. [Machine-readable audit](resident-fp16-basic-vjp-20261001.json).
All seven clean-source jobs passed; the portable core was unchanged.

This qualifies local identity/EMA/Add state and identity/tanh Full VJPs for
actual FP16 forward operands with **FP32 cotangents and accumulation**. It does
not qualify whole-graph FP16 reverse, LH/SwiGLU/Attention/control half adjoints,
retained FP16 training, public FP16 training, peer execution or throughput.
The public complete-training guard remains. Candidate device work is independent;
CPU component oracles verify derivatives and do not supply graph execution plans.

## Precision and semantic checks

State adjoints consume the actual widened journal and half parameter banks. EMA
uses a sigmoid coefficient in forward precision. Add reconstructs every literal
multiplication with half rounding, including prefix recomputation for smaller
scratch chunks. It never replaces the chain with a power/division shortcut.
State links, adoption/clear and reverse progression remain device decisions.

Tanh Full recomputes matrix product, bias and tanh in half precision, then widens
saved operands/results for FP32 adjoints. It does not subtract a large residual
to recover an activation or recompute the entire half forward in FP32. Temporary
half storage is included in admission; active rows are packed on device and
unused poison never enters matrix/activation work.

The reference for half payload uses independent CPU quantized forward arithmetic,
FP32 adjoints and the ordinary identity cast VJP. This is a declared mixed precision
policy, not bitwise pure-half backward accumulation or a derivative of rounding's
staircase. Half comparison tolerances stay rtol2e-3/atol2e-5; FP32/FP64 comparisons
stay1e-5/1e-6. Connection flags/None/zero and discrete metadata remain exact.

| Component | FP32/FP64 reference cases | Half forward reference cases | Actual device tapes per dtype |
| --- | --- | --- | --- |
| State |216 |108 |4, streaming/greedy × EMA/Add |
| Identity/tanh Full |96, each with19/7/0-row replay |48, same replay lengths |2, streaming/greedy |

Additional half anchors use strict FP32 tolerances:
- A1024-tick Add chain, chunks17/1024, compares lifted CPU autograd. The test
  asserts that an incorrect whole-forward FP32 replay differs by more than1 in
  the scalar retention adjoint, then checks device adjoints against the quantized
  reference. Both chunk sizes passed.
- A Full matmul produces100.03125 before rounding to half100, adds-97 and runs
  tanh. Its weight derivative differs by more than1e-4 from whole-forward FP32
  recomputation. Device input/weight/bias adjoints passed strict checks. A32768
  residual prevents recovering the activation by subtraction.

The state cases retain int64 clocks/counters above2^55, periodic clocks,
zero/negative retention, poison/None/connected zero, clear/adoption, empty replay,
nonaligned widths and explicit bounds. Full retains poisoned unused owners,
sentinel/padding safety, repeated owners, bounded rows and explicit refusals.
FP32 event/fiber complete-training regressions passed their existing strict gates:
66 roots/eight trajectories and172 roots/20 trajectories. Python resident client
regressions passed215 cases with zero skips, including unsupported-half-training
and checkpoint/inference lifecycle checks. These do not turn local half adjoints
into a complete half-training result.

## Frozen runs and build reuse

All jobs read `low-precision-basic-vjp-clean01` at the full implementation hash.
Locally, `TASK=/mi/data2T/zlong/tide-execution-flows`; each job has
`TASK/runs/NAME/status.json`, `task.log` and gate/profile artifacts.

- `build-low-precision-basic-vjp-clean01`: affected Full host/fixtures/checkers
  rebuilt; terminal byte-matched state kernel/content archive and other CANN
  dependencies reused. State runtime sources had already passed the development
  gate; every retained archive member was checked.
- `build-low-precision-basic-vjp-python-clean01`: three affected Python-owned
  host objects rebuilt and client relinked, independent of the standalone SDK.
- `low-precision-basic-vjp-components-clean01`:four cells,physical1→logical0.
- `low-precision-basic-vjp-regression-clean01`:two FP32 cells,physical3→logical0.
- `low-precision-basic-vjp-python-clean01`:215 passed,physical9→logical0.
- `low-precision-state-vjp-profile-clean01`:physical13→logical0.
- `low-precision-full-vjp-profile-clean01`:physical11→logical0.

This is immutable-source qualification with controlled dependency reuse, not a
from-scratch CANN rebuild. The audit verifies component/core/source identities,
retained content/fixture archive members, rebuilt objects, kernel/binary/loader
hashes, raw gate logs and profiler CSV inputs. The unchanged8,954-check portable
CPU gate was not repeated.

Standalone gate command: `scripts/verify_device_control.py --device npu:0 --checks
state-vjp full-vjp`, separately `--checks event-training fiber-training`, with
matching build and unique output directories. Python tests are the resident
precision/library/training/event-training/fiber-training files at `--dtype float32`;
precision cases explicitly select half. Exact commands/hashes are in raw records.

## Profiling and retained failures

Separate `profile_device_control.py --device npu:0 --dtype float16 --check
state-vjp|full-vjp --storage-limit-mb 256` runs completed the full respective
component checkers. State trace:5614 AI_VECTOR_CORE tasks. Full trace:5424
AI_VECTOR_CORE and162 AI_CORE tasks. No AiCPU task or logged CPU fallback was
observed. These traces include correctness-driver work and are not throughput
measurements or CPU/NPU speed comparisons.

Preserved failed jobs:
- `build-low-precision-state-vjp-dev01`: ambiguous mixed int/int64 checker tensor
  initializer; fixed with explicit vector<Index>.
- `low-precision-state-vjp-dev02`: original oracle also accumulated backward in
  half, contrary to the candidate's FP32-adjoint contract. The cancellation case
  reported maximum difference about5.05e-5. The corrected oracle preserves the
  same half forward inputs and rounds every forward operation, with FP32
  derivatives. Runtime and both dtype tolerance policies were unchanged.
- `build-low-precision-basic-vjp-dev01`: missing at::Tensor qualification in the
  new fixture. Runtime unchanged.

None of these original records was relabelled passed. See the linked precision
and component contracts for scope; F1–F7 remain incomplete.
