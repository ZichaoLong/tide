# Using Tide instances

Install `tide-instances` from this checkout; import `tide`. Mathematics is fixed
at the revision in [upstream.json](upstream.json). No retired executor is used.
Current CPU/NPU FP32 and BF16 observations, including retained support and
finite numerical deviations, are in the [numerical evidence](evidence/roadmap-numerics-followup-20261118.md).
Use the explicit [precision policy](precision.md): FP32 parameters with BF16
matrix arithmetic under `precision_context`, rather than casting the whole model.
CPU FP64 supports bounded accuracy diagnosis. FP16 is outside this cycle.

Execution candidates are explicit `ModelConfig` options: `region_batch_size`
(default 1) batches compatible regions within a layer, `checkpoint_regions`
(default false) recomputes regional activations during backward, and
`local_attention_backend` (default null) overrides only reception/regional
Attention while KV retains `attention_backend`. They are saved in checkpoints.
These choices preserve region budgets and private state; they are not assertions
of a universal speed or memory improvement. Compare them with the
[bounded search](capacity-search.md), and use `scripts.check_execution` for
same-function values, gradients, multiple updates and restored-update checks.

Two additional explicit training candidates retain the same ownership and state
contract. `checkpoint_reception=True` recomputes only the reception module; whole
region recomputation takes precedence to avoid nested checkpoints.
`ssd_summary_backend="prefix"` uses parallel affine composition for the short/long
vector summaries, including zero decay, without logarithms or division by decay.
The default `"block"` keeps the existing block algorithm; the 32×32 SSD matrix
state recurrence is unchanged. Both options are saved in checkpoints. Their
different floating evaluation order needs the same bounded numerical checks as
other implementation candidates; neither is enabled automatically.
The measured `npu-hb-*` training presets explicitly enable both, together with
compatible region batching. Select a preset and an admitted device explicitly;
the no-argument training example remains a small CPU run.

Reception self-Attention has two additional execution choices. With
`aggregation="incoming_light_64"`, `"incoming_light_128"` or `"incoming_wide"`,
`reception_qkv_backend="fused"` concatenates the compatible Q/K/V projections
while retaining their original parameters and private low-rank adapters.
The default is `"separate"`; placed owner wrappers explicitly fall back to
separate projections, reported by `model.receive.effective_qkv_backend`.
`reception_mask_padding=True` uses graph membership to exclude structural
padding from Attention. Genuine zero-strength edges remain visible to the
declared right derivative; this option never replaces that domain with
`mass > 0`. Its default is false. Both choices compose with reception/region
recomputation and compatible region batching and are saved in checkpoints.
They do not switch `source_residual` into self-Attention. The training entry
accepts `--aggregation`, `--reception-qkv-backend`, and
`--[no-]reception-mask-padding` as explicit overrides. These are implementation
candidates; qualification and measured recommendations retain their source scope.
The [Reception execution guide](reception-execution.md) gives the formula,
mask/gradient domain, full-Reception timing and integrated path-check commands.

The [training entry and recommendations](execution-recommendations.md) provide
loadable presets, explicit overrides, three modes and actual resume. Use
[training path checks](training-path-checks.md) for fixed nonempty state/AdamW,
same-owner repeat and fresh-process restored-update comparisons.

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
