# Using Tide from an experiment repository

Tide 0.2 is a graph execution dependency. The experiment owns its dataset,
task head, loss, optimizer, training loop, metrics and artifacts. Tide owns the
graph/module configuration, execution policy and graph continuation. CPU
FP32/FP64 is the required baseline. Python/native CUDA and NPU implementations
use explicit device selection and independent CPU qualification. See the
[accelerator guide](accelerators.md) for isolated builds, tested scopes and
target-machine acceptance. NPU requires FP32; CUDA device results need a CUDA host.

## Configuration and runtime

Install a pinned wheel into the experiment's own environment, keeping its
existing CPU/TorchNPU stack: `python -m pip install --no-deps tidegraph-0.2.0-py3-none-any.whl`.
Build that wheel from a pinned checkout with `python -m pip wheel --no-deps
--no-build-isolation . -w /path/to/wheels`. This build writes setuptools staging
files; use a writable build checkout or a source copy. The installed runtime
never writes into Tide source and needs no source/scripts/tests on PYTHONPATH.
The wheel is Python-only; native is an explicit separate dependency.

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

`model.scale_init` optionally initializes all graph-owned input/output,
Aggregate and edge scalar weights to an explicit finite value. Null retains
the original reference initializer (.8 + .03 * owner index); large fan-in
experiments should choose their initialization deliberately. This changes
initial weights, not graph semantics. A supplied Model owns its existing
weights; initializer settings do not overwrite it. The manifest records
whether the model was configured or supplied by the caller.

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

`ExecutionOptions` selects Python/native and auto/reference/streaming/frontier/greedy
or supported chain/diamond/ring/self_loop specializations. Auto resolves to
streaming for PDG and frontier for the other families. Reference requires
Python and `packed=False`. Invalid/inapplicable explicit options fail. The
public Settle session uses encoded executors and retains the complete encoded
continuation; its result projects body diagnostics. A projected result is not
the serialized session. Direct Settle remains an independent equivalence oracle.

`schedule="greedy"` enables [online node-time batching](greedy-scheduler.md),
including positive-delay PDG feedback. It defaults to `prefill=True`; `max_events`
limits live fibers rather than cumulative work. It is currently host-scheduled.

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
whole-versus-chunk observables/VJPs; requested trace-disabled values/VJPs;
a stateful optimizer trajectory with full diagnostics; and a
fresh-process checkpoint continuation. Settle also compares its independent
direct scalar schedule against the encoded execution. Scalar Python scheduling
is the default oracle; a reference candidate uses independent scalar streaming.
Choose `--optimizer adamw|sgd|momentum`. Tensor comparisons reject nonfinite
values. Defaults are FP64 atol=1e-10/rtol=1e-8 and FP32 atol=1e-6/rtol=1e-5;
explicit `--atol`/`--rtol` overrides are recorded and never alter discrete checks.
Diagnostic losses divide weighted squared observables by their scalar count;
isolated roots use means, keeping test-gradient scale independent of graph size.
The default three steps exercise state
carry, explicit truncation and repeated updates. These are semantic test losses,
not a claim about task convergence or performance.

For real inputs use `tidegraph.qualification.qualify(config, device="cpu",
output_dir=..., inputs=..., batch_size=..., positions=..., stop=...)`. Settle
accepts the same tensor format as its session; explicit External probes require
a final logical stop. CPU tensors must already match the effective dtype/shape.
This gate constructs the modules declared by the config; custom Python objects,
custom task heads and pretrained weight import require their own differential
tests. Accelerator candidates are compared with an independent CPU oracle,
including live tensor placement and checkpoint handoff. Exact application resume still needs the
experiment-owned state described above. Repeat qualification whenever topology,
modules, execution policy, dtype or dependency version changes.

## Native and C++ dependencies

Build the adapter from the same pinned source using the experiment's exact
Torch/Python/ABI/architecture: `python scripts/build.py --build-dir /path/to/build
--jobs 2`. For native execution pass `native_library="/path/to/build"` to
GraphRuntime. The loader validates the adjacent build manifest, host/Torch/ABI
and binary hash; it performs no compilation or sys.path mutation. A relocated
adapter needs both `_tide_native.so` and `build-manifest.json`, and the matching
Torch dynamic libraries. Rebuild on another architecture or framework stack.
Different native binaries cannot be loaded into one process. Qualify the exact
package/native pair before running an experiment.

For a standalone C++ dependency, no Python bindings or repository clients are
required:

```sh
cmake -S /path/to/tide -B build/tide -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch -DTIDE_PYTHON_BINDINGS=OFF \
  -DTIDE_BUILD_CLIENTS=OFF -DCMAKE_INSTALL_PREFIX=/path/to/tide-prefix
cmake --build build/tide --parallel 2
cmake --install build/tide
```

In the consuming CMake project use `find_package(TideGraph 0.2 CONFIG REQUIRED)`
and `target_link_libraries(my_target PRIVATE tide::tidegraph)` for the graph core,
or `tide::runtime` when using the standalone device resolver/lifecycle. Include both the
Tide install prefix and matching Torch prefix in `CMAKE_PREFIX_PATH`. The export
propagates headers, C++17, ABI flags, Torch and Threads. It validates Torch
version and architecture; the consumer must still use the exact compatible
Torch distribution and runtime loader environment. C++ constructs graphs/models
through its typed API; schema-v1 JSON configuration is the Python frontend.
`examples/consumer_cpp` exercises standalone forward, chunking and backward.

Standalone clients should declare `portable_torch::RuntimeSession runtime;` at
the start of `main`, before tensor and worker owners. Its last active scope
finalizes an initialized LibTorch-NPU runtime after those owners are destroyed
and before main-thread TLS disappears. `close()` provides checked, idempotent
cleanup after all NPU work/owners have ended; an initialized NPU runtime cannot
be reopened after finalization. CPU/CUDA scopes perform no vendor teardown.
An embedding application can own teardown itself without creating this scope.
Device resolution no longer registers a process-exit finalizer: TorchNPU2.10's
finalizer accesses thread-local streams, which are already destroyed at that
late boundary. Do not place `RuntimeSession` in static/global storage.

`scripts/library_consumer.py --output-dir NEW --build-dir MATCHING_BUILD` is
the developer consumption gate: build a generic wheel in staging, install it
in a fresh environment reusing the existing Torch stack, run a copied external
application and installed qualification for all three families, relocate the
native adapter, install the CMake package, and compile/run the external C++
client. Consumers never import benchmark scripts or reference repositories.

## Complex topology acceptance

The developer suite `scripts/library_complex.py` runs 22 declared cells:
fully active 64-node/768-edge layered TimedDAG and Settle, a fully active
32-node feedback ring with a hub, varied delays and parallel edges, plus exact
P01/P02/T02/S01/A02 graph records from foundation-v1. The active graphs mix EMA,
Linear, Gated Delta, DeltaRule, SSM, event Attention, repeat Add and same-fiber
Attention, with tanh/SwiGLU and multiple Aggregate profiles. Both CPU dtypes
are required; active mixed graphs use both Python and native candidates.

Every node/edge/region/module is retained while validation uses D4/B1/T2 and
three optimizer steps. Source scalar initialization is explicit (.25 for mixed
graphs, .8 for benchmark graphs). The benchmark topology claim does not assert
the original benchmark initialization, full tensor sizes, trained weights or
performance. P02 retains all 8192 nodes but only its four-node ring is active;
the other 8188 nodes cannot count as active-work equivalence evidence. The full
acceptance command is `scripts/qualify_library.py`, combining the complete CPU
regression, complex suite and installed consumer checks on frozen source.


Explicit `dtype="float16"` and `FP32MasterOptimizer` are documented in
[precision.md](precision.md). Their target-specific qualification is separate
from the original FP32/FP64 acceptance. The optimizer belongs to the caller and
supports Session checkpoint continuation with FP32 masters and slots.
