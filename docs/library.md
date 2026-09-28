# Using Tide from an experiment repository

Tide 0.2 is a graph execution dependency. The experiment owns its dataset,
task head, loss, optimizer, training loop, metrics and artifacts. Tide owns the
graph/module configuration, execution policy and graph continuation. CPU
FP32/FP64 is the required baseline. Python NPU FP32 is experimental; historical
NPU smoke evidence does not certify the new library API or arbitrary configs.
Native NPU and CUDA are unsupported in this release.

## Configuration and runtime

```python
import torch
from tidegraph import GraphConfig, GraphRuntime

config = GraphConfig.from_dict({
    "schema_version": 1,
    "family": "settle",
    "topology": {"kind": "layered", "layers": [2, 3, 2],
                 "module": {"memory": "ssm", "full": "swiglu"}},
    "model": {"width": 8, "dtype": "float32", "seed": 7},
    "execution": {"implementation": "python", "schedule": "frontier"}
})
runtime = GraphRuntime(config, device="cpu")
session = runtime.session(batch_size=2)
x = torch.randn(2, 4, 8)  # Provided by the experiment.
result = session.advance(x)
loss = torch.stack([value for _, _, _, value in result.outputs]).square().mean()
loss.backward()
session.detach()  # Explicit truncation before an optimizer changes parameters.
```

`GraphConfig.load(path)` loads JSON. `to_dict()` emits an explicit normalized
graph; `identity` hashes that configuration. Graph identity is separate from
execution policy; changing packing or schedule does not invalidate a compatible
checkpoint. Supported factories: chain, diamond, ring, self_loop and layered.
`node_overrides` and `region_overrides` use integer owner IDs (JSON string keys);
each accepts the corresponding Node/Region fields. `budget` applies exactly to
every region; omit it for full region budgets or use per-region overrides.
Dormant layers must be declared explicitly. An explicit `graph` object supports
arbitrary nodes, edges, regions, input/output owners, port layouts, origins and
source domains. Parallel edges retain separate identities.

Select `family="pdg"` for positive-delay feedback graphs, `"timed-dag"` for
acyclic graphs, or `"settle"` for rank-aligned complete-position execution.
Settle requires region ranks for explicit graphs; factories provide ranks.
Existing [module capabilities](execution-capabilities.md) define legal profiles.
No configuration interface makes an arbitrary pretrained module supported.

For PDG/TimedDAG, pass `External(batch, port, position, time, tensor)` records to
`session.advance(records, stop=cut, sealed_until=cut)`. The half-open logical
window and complete external-input promise are explicit. Positions per sample
and port start at zero and are contiguous; times strictly increase. For Settle,
pass `[batch, positions, width]`; the same position vector feeds each body input
port. Separate input modalities use explicit TimedDAG/PDG boundary wiring.

No input conversion, hidden detach, optimizer operation or no_grad context is
performed. `result` exposes outputs, state, history, pending messages and, with
`trace=True`, full event/route diagnostics. Output tuples are
`(sample, logical_time, output_port, tensor)`; Settle has one encoded output
boundary aggregating its body outputs. Session state persists between calls.
Multiple sessions can share one runtime's parameters. Do not update shared
parameters while another session retains an autograd segment needed backward.

## Policies and state

`ExecutionOptions` selects Python/native and auto/reference/streaming/frontier
or supported chain/diamond/ring/self_loop specializations. Auto resolves to
streaming for PDG and frontier for the other families. Reference requires
Python and `packed=False`. Invalid/inapplicable explicit options fail. The
public Settle session uses encoded executors and retains the complete encoded
continuation; its result projects body diagnostics. A projected result is not
the serialized session. Direct Settle remains an independent equivalence oracle.

`GraphRuntime(..., model=model)` accepts an explicitly assembled `Model` with
custom Python programs or shared owners. It must match graph/device/dtype/width.
Custom Python programs need separate native implementations to use native.
The caller must construct the same owner sharing before restoring weights.
Moving/replacing parameter owners requires a new runtime; in-place optimizer
updates and ordinary `load_state_dict` preserve the bound owners.

`session.save(path, optimizer)` publishes a new checkpoint atomically;
`session.load(path, optimizer)` restores graph state, weights and named optimizer
state. `runtime.load_weights(path)` initializes only weights and starts no
session progress. `session.reset()` clears sequence state and preserves weights.
The [v5 checkpoint contract](semantics.md) excludes the experiment's task head,
scheduler, data sampler, RNG and controller; save these in the experiment's own
bundle for exact application resume. Loading starts a new autograd segment.
`runtime.manifest()` returns config/package/runtime/policy/binary identities for
the experiment's record without writing into the dependency checkout.

## Compatibility

Pin an exact version or source commit in each experiment environment. Config
schema v1, continuation v5 and native structural identity v13 are distinct
contracts. Unknown config fields fail; schema changes require explicit migration.
This pre-1.0 API has no promise of compatibility across minor releases. Public
entry points are the exported configuration/runtime/record classes and the
documented qualification CLI. Internal schedulers and benchmark scripts are not
the experiment API. New module implementations belong here with independent
semantic tests; keep task-specific training code in the consuming repository.

## Qualify the configuration before an experiment

```bash
python -m tidegraph inspect graph.json
python -m tidegraph qualify graph.json --device cpu --output-dir results/gate-001 \
  --width 8 --dtype float64 --batch-size 2 --positions 4 --steps 3
```

Native configs additionally pass `--native-library /matching/build-directory`.
`--width` and `--dtype` are explicit qualification overrides; they never change
the topology or module assignments. Inspect `report.json` for requested and
effective config hashes, generated/caller input hash, package/native identity,
actual touched/unobserved nodes, tolerances, resolved policy and every passed
check. The retained `input.pt` contains the exact effective config and fixture
values, including caller data; choose its output location accordingly. Output
directories must be new. Failures retain a terminal failed report
and exit nonzero. A running report is not evidence of success.

The gate checks full outputs, state slots, history, routes, trace, pending and
ledger; input/parameter VJPs including None connectivity and isolated roots;
whole-versus-chunk observables/VJPs; a stateful optimizer trajectory; and a
fresh-process checkpoint continuation. Settle also compares its independent
direct scalar schedule against the encoded execution. Scalar Python scheduling
is the default oracle; a reference candidate uses independent scalar streaming.
Choose `--optimizer adamw|sgd|momentum`. The default three steps exercise state
carry, explicit truncation and repeated updates. These are semantic test losses,
not a claim about task convergence or performance.

For real inputs use `tidegraph.qualification.qualify(config, device="cpu",
output_dir=..., inputs=..., batch_size=..., positions=..., stop=...)`. Settle
accepts the same tensor format as its session; explicit External probes require
a final logical stop. CPU tensors must already match the effective dtype/shape.
This gate constructs the modules declared by the config; custom Python objects,
custom task heads and pretrained weight import require their own differential
tests. It currently certifies CPU only. Exact application resume still needs the
experiment-owned state described above. Repeat qualification whenever topology,
modules, execution policy, dtype or dependency version changes.
