# Using Tide instances

Install `tide-instances` from this checkout; import `tide`. Mathematics is fixed
at the revision in [upstream.json](upstream.json). No retired executor is used.
Current CPU/NPU FP32 and BF16 observations, including retained support and
finite numerical deviations, are in the [numerical evidence](evidence/roadmap-numerics-followup-20261118.md).
Use the explicit [precision policy](precision.md): FP32 parameters with BF16
matrix arithmetic under `precision_context`, rather than casting the whole model.
CPU FP64 supports bounded accuracy diagnosis. FP16 is outside this cycle.

```python
import torch
from tide import ModelConfig, TopologyConfig, TideModel, detach_state

model = TideModel(ModelConfig(
    topology=TopologyConfig(kind="fc", n=8, layers=4), d=128))
x = torch.randn(8, 128, 128)  # independent samples, input positions, content
result = model(x)
y = result.readout()         # [B,L,d] for Settle
following = model(x, result.state)
```

`Result.values` is `[output_node,B,logical_position,d]`; `strengths` omits the
last axis; `times` is int64. `readout()` is the explicitly supplied convenience
sum(cG)/max(1,sum c), not a task head or a prescribed downstream loss.
Settle returns one output slot per input position. Timed returns every logical
tick, including zero-output ticks, with `period` ticks per input position.
Call `model.flush(state)` after the final Timed window to receive the remaining
in-flight outputs without inserting fictional external messages. Settle needs
no flush. Returned state advances its clock through any flushed periods.

The state owns private KV/SSD history and Timed pending messages. To backpropagate
across windows, pass state unchanged and keep parameters fixed until backward.
For truncated training, use `detach_state(result.state)` at optimizer boundaries.
`save_checkpoint(path, model, state=..., optimizer=...)` saves parameters, exact
state and optional optimizer state. `load_checkpoint(path, device=...)` returns
`(model,state,optimizer_state)`; load the latter into an identically configured
optimizer. Checkpoint files are trusted Torch pickle files.
An installed-API example is `python examples/train.py --checkpoint NEW.pt`.
It performs three updates, checks exact restored data and next-window forward,
then executes a further AdamW update on both copies. The update comparison
reports finite FP32 deviations separately from exact decoding/forward; parallel
reductions need not give bitwise identical optimizer trajectories. Its scalar
objective verifies the mechanism and is not an experiment training recipe.

The finite choices in `ModelConfig` are:

| Field | Main choices |
|---|---|
| memory | K: KV; S: SSD; X: ordinary KV and Hub SSD |
| aggregation | source_residual, incoming_light_64, incoming_light_128, incoming_wide |
| allocation | quad, shannon, pool |
| sharing | independent, region_shared, layer_shared |
| d | 64,128,192,256,384,512,768 |
| adapter_rank | 4,8,16 |
| budget | 1,2,4,8,16,32,64 up to n; default max(1,n/32); pool ignores it |
| attention_backend | auto, math, tiled, sdpa, npu_fusion |

`TopologyConfig.kind` selects fc, lattice, exchange or multiscale. `n` is ordinary
nodes per region. FC uses `layers`; Lattice uses expansion `depth`; Exchange
uses `depth` and `rounds`. Multiscale uses `profile` (e.g. compact-9), `timing`
(settle/timed), Hub mask and partition. Complete legal combinations and defaults
are checked by [topology.py](../src/tide/topology.py); the shapes/edge definitions
are in [the fixed main binding](../math/instances/backbone-comparison.md).
Tiny graph widths are additionally accepted for formula witnesses.

Exchange/Multiscale can use `hubs=4`. Multiscale additionally allows group,
singleton or mixed partitions and custom/checkerboard Hub placement. Pool is
invalid for mixed regions containing Hubs. H groups always use role quad;
singleton Hubs pass their bound directly. K/S/X and all other main choices are
composable within these restrictions. The tests cover individual choices and
selected interactions, not all combinations or scales.

Optional group-Hub fixed pass, explicit local clocks/strict edge-delay pairs and
terminal Hubs are documented in [optional bindings](optional-bindings.md).
`python -m scripts.support_catalog` checks the complete main menu's metadata;
its output separates expression from execution, numerical and performance qualification.

Three Attention sites are separate calls with different shapes: historical KV,
same-tick incoming messages and regional candidates. `auto` selects implemented
native/fused or tiled math based on dtype/device/prior semantics. Trainable
visible zero priors require the right derivative and cannot use log-prior native
fusion. Explicit backend requests fail when inapplicable. Device qualification
and performance are separate from CPU formula correctness.

For timing use `python -m benchmarks.run --config configs/fc-k.json --device cpu
--output artifacts/fc.json`. Modes are inference, grad_forward and complete
train; all time fresh prefill windows. `B*L/seconds` is input-position throughput.
Window latency is not first-token or decode latency. Comparisons identify actual
parameter counts, layer/node counts and the intentionally different mathematics.
Use [execution-flows.md](execution-flows.md) for frozen long runs.

The timing harness generates inputs on CPU, moves them before timing, and
records input and initial-parameter hashes for paired device comparisons.
`TIDE_SOURCE_COMMIT` identifies the model source; `TIDE_HARNESS_COMMIT` can
separately identify a newer harness used to measure an earlier implementation.
Initial same-mode work audits fix the four joint Transformer matches before
warmup or updates. A separate post-timing audit records changed work; neither
is a sum of every timed step. The definitions and continuous cached comparisons
are in [benchmarking](benchmarking.md). CPU RSS is a process high-water mark;
NPU allocator peaks are recorded per timed observation. Shared-device timings
remain shared observations even when the inputs match.
