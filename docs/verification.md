# Verification of the current instance library

Current facts are in [STATUS](STATUS.md) and source-scoped [evidence](evidence/README.md).
The active tests are `tests/`; no retired LH/native/full-matrix gate is a prerequisite.
Use [development](development.md) for a clean environment and [execution](execution-flows.md)
for frozen long jobs. The next FP32/BF16 programme is [ROADMAP](ROADMAP.md).

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
# Only in an admitted device job:
python tests/device_attention.py --device npu:0 --output NEW-attention.json
python scripts/check_device.py --device npu:0 --output NEW-models.json
```

These existing device scripts certify their recorded FP32 scope, not future BF16.
Use formula witnesses, solved-map VJPs, direction derivatives and independent
schedules for the questions they answer. The mathematically different
Transformer/SSD comparison cannot replace those checks.
Returned values/strengths, relevant state, support, input/parameter gradients,
None/zero distinctions, updates and continuation matter, not just a readout.

The current frozen model passed 308 CPU tests and 10 device Attention checks;
six integrated NPU cases have separate structural and numerical outcomes. Those
counts remain attached to their source; a new change does not inherit
qualification automatically. Finite deviations follow [precision](precision.md).
