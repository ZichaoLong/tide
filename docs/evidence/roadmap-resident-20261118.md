# Bounded NPU resident/fullgraph feasibility probe

The actual Tide forward was attempted with the installed TorchAir backend on NPU.
All four finite attempts stopped at the same frontend guard before the backend
callback ran. This qualifies the probe's execution and records the current
capture barrier; it does not qualify a resident runtime or establish that one
cannot be built.

[Machine-readable evidence](roadmap-resident-20261118.json.gz) retains source,
commands, launch bounds, queue mapping, input/parameter identities, result/log
hashes, process lifecycle and the executed failure. The source is
`6fc53d81910252ac42bd5fce1476f7a70b968ed7`; mathematics remains
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`.

## Attempt and observation

`resident-npu-v11` used physical NPU 9, mapped to `npu:0`, through the cooperative
exclusive queue. It ran in a detached `background.slice` service with 4 CPU
threads on cores 168–171, a 600-second outer limit, a sampled process-tree RSS
limit of 32 GiB, and 60 seconds per isolated attempt. The complete job took
51.927 seconds; each attempt took 10.662–11.731 seconds. These include eager
audits, imports and failed capture, and are not inference performance timings.

The two actual model configurations are FC-K and HB Lattice-K, using width 64,
B1/L2 and FP32. Each has two fixed inputs, `x` and `-x`, whose eager executions
both completed with distinct work/support observations. The probe invokes
`torch.compile(fullgraph=True, dynamic=True)` on unmodified `TideModel.forward`.
Readout and comparisons remain outside capture. No routing oracle or dense Full
substitution is supplied. Two settings form the complete finite matrix:

| Configuration | Scalar/dynamic-output capture settings | Result | TorchAir callback invocations |
|---|---|---|---|
| FC-K | both disabled | frontend blocked | 0 |
| FC-K | both enabled | frontend blocked | 0 |
| HB Lattice-K | both disabled | frontend blocked | 0 |
| HB Lattice-K | both enabled | frontend blocked | 0 |

TorchAir was discovered at the software baseline's
`torch_npu/dynamo/torchair/__init__.py`, and its real `get_npu_backend()` was
selected in all four attempts. The first compiled invocation stopped with
`Could not guard on data-dependent expression Eq(u0, 1)` at
`src/tide/model.py:259`:

```python
if bool(roles.all()) and self.graph.config.partition == "singleton":
```

The `roles` tensor is constructed from fixed graph metadata. This first barrier
can be moved outside capture, or the known static partition condition can be
evaluated first. The observed guard therefore does **not** establish an inherent
dynamic-routing or TorchAir limitation. Because the first compiled call failed,
the second compiled input and compiled output comparisons were never reached.
There is no evidence here of successful or failed TorchAir graph lowering.

Actual data-dependent selection and state progression remain uncaptured. The
source scan also records candidate host decisions involving `bool`, `nonzero`,
counts and `tolist`; it is a lexical inventory containing construction-only
sites, not a measured cost table or a set of executed backend failures.
Further capture changes or a device-owned pending/next-step scheduler require
separate implementation and validation. This bounded probe does not expand
into an open-ended compiler investigation.

## Lifecycle and reproduction

Each attempt records an independent terminal lifecycle. A dedicated Linux
subreaper retains descendant PID/start identities across orphaning and nested
sessions, and uses the shared process-tree stop helper on timeout or exception.
Incomplete cleanup or interruption stops subsequent attempts. All four actual
NPU attempts ended normally with `cleanup_complete=true`; no timeout or forced
kill occurred. The service's `passed` status means these explicit blockers were
successfully recorded, not that resident execution passed.

Use the frozen source, private software baseline and admitted finite launcher
described in [execution](../execution-flows.md), with a new output identity:

```sh
python -m scripts.probe_resident --device npu:0 --timeout-seconds 60 \
  --output NEW-resident-probe.json
```

`tests/test_resident_probe.py` also exercises nested-session orphan cleanup,
interruption, unrelated-process preservation and the actual CPU frontend scope.
CPU uses `backend=eager` and is explicitly frontend-only evidence. Neither CPU
nor this NPU probe certifies autonomous device scheduling, compile performance,
CUDA execution or successful compiled training.
