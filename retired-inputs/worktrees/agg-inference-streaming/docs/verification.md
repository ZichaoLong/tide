# Verification of the current instance library

Current facts are in [STATUS](STATUS.md) and source-scoped [evidence](evidence/README.md).
The active tests are `tests/`; no retired LH/native/full-matrix gate is a prerequisite.
Use [development](development.md) for a clean environment and [execution](execution-flows.md)
for frozen long jobs. [ROADMAP](ROADMAP.md) owns the cycle's work; completed
qualification is recorded in the linked evidence, not inferred from the plan.

| Change | Relevant independent checks |
|---|---|
| Topology/configuration | topology/parameter-binding tests, legal/illegal combinations |
| Reception, KV, SSD, allocation | respective formula tests and small FC/Timed oracles |
| Packing/shared projection/autograd | parameter/input contributions, repeated owners, selected support and state |
| State/checkpoint | split/whole windows, pending/flush, private state and actual resumed updates |
| Precision/device adapter | primitive forward/VJP, integrated cases, declared policy and actual device |
| Benchmark/statistics | independently enumerable work/parameters, timing boundary and installed entry |
| Documentation/cleanup | links/commands, immutable math, clean installed smoke and preserved user data |

Run directed checks during editing. Broaden to the current CPU suite when an
integrated semantic change warrants it. Commit implementation before durable
qualification; run that immutable source and then commit reviewed evidence.
An outer successful job is not proof that each internal numerical check passed.

Current commands (with the matching environment):

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q
python scripts/check_upstream.py
# Finite CPU formula/VJP and whole/chunk diagnostics, each with a new output:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m scripts.qualify_precision --device cpu \
  --precision bf16 --primitives-only --output NEW-primitives.json
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m scripts.check_continuation --device cpu \
  --precision fp64 --output NEW-continuation-fp64.json
```

The [current numerical report](evidence/roadmap-numerics-followup-20261118.md)
owns actual-width CPU fixture generation, admitted NPU FP32/BF16 comparison,
multi-step/restore and whole/chunk commands. `tests/device_attention.py` and
`scripts/check_device.py` retain their earlier FP32 scope; current BF16 coverage
uses `scripts.qualify_precision`. The [placement contract](placement.md) and
[resident probe](evidence/roadmap-resident-20261118.md) separate execution completion,
numerical outcomes and unqualified backend paths. NPU admission and the frozen
lifecycle for long jobs follow [execution](execution-flows.md).

Use formula witnesses, solved-map VJPs, direction derivatives and independent
schedules for the questions they answer. The mathematically different
Transformer/SSD comparison cannot replace those checks.
Returned values/strengths, relevant state, support, input/parameter gradients,
None/zero distinctions, updates and continuation matter, not just a readout.

The [integrated CPU record](evidence/capacity-validation-20261118.md) records
544 tests on frozen `8a22530d` and 17 targeted integration tests on `5ae060db`.
The [new installed wheel](evidence/capacity-installed-20261118.md) covers optional
bindings, batching/recomputation, logical CPU placement and new-process restored
updates. The [execution-candidate investigation](evidence/capacity-rounding-20261118.md)
and [regional NPU record](evidence/region-capacity-20261118.md) retain their finite
value/support/gradient/update failures. Earlier [FP32](evidence/instances-fp32-20261118.md)
and [installed results](evidence/roadmap-installed-20261118.md) remain unchanged;
no test count certifies a later source or relabels numerical failures.
Finite deviations follow [precision](precision.md).

The [training validation record](evidence/training-validation-20261118.md) adds
625 CPU tests on frozen `ae247dee`, targeted checks for later harness/preset
changes, and a new isolated wheel exercising prefix summaries and Reception
recomputation through actual restored updates. Production `src/tide` identity
is verified separately from later documentation and harness commits. The
[path investigation](evidence/training-paths-20261118.md) and
[continuous training](evidence/training-continuous-20261118.md) retain finite
and structural comparison failures; installed completion and test counts never
erase those failures or certify arbitrary configurations.
