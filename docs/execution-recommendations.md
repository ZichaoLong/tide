# Training entry and execution recommendations

[`examples/train.py`](../examples/train.py) is a finite example using the installed
public Tide API. With no arguments it trains FC-K on CPU for three updates and
prints the resolved model, clocks, runtime, precision policy and each update as
JSON lines. Synthetic normal inputs and the squared readout scalar demonstrate
the mechanism. Task heads, data, losses and training recipes remain downstream.
The entry performs diagnostic finite checks; its elapsed time is not a benchmark.

Actual training and subsequent training optimization prioritize **FP32**. BF16
remains a secondary option; its measured presets, numerical deviations and original
runtime settings are retained. Throughput, latency and memory still need their own
mode/configuration-specific evidence. See the [scale/topology tables and computation
flow](computation-and-throughput.md) for an FP32-first view of existing results.

```sh
python examples/train.py
python examples/train.py --list-presets
python examples/train.py --preset hb-timed --dry-run
python examples/train.py --preset fc-k --precision bf16 --steps 5
```

Use a Python environment with `tide` installed, or set `PYTHONPATH=src` when
running from this checkout. The script imports no `benchmarks` or `scripts`
modules. Its preset directory is relative to the script, so an absolute script
path also works from another directory. Examples and presets are checkout tools;
the wheel installs the `tide` package.

## Presets and explicit overrides

[`configs/recommended/`](../configs/recommended) separates the `model` dictionary
(`ModelConfig`) from `execution` settings. Each preset also carries a recommendation
status, evidence links and the scope of its claim. Runtime B/L, precision, mode,
optimizer settings and device are not fields of `ModelConfig`.

| Preset | Model | Current runtime identity |
| --- | --- | --- |
| `default` | FC-K, n8, 4 layers, d64 | Safe CPU FP32, B2/L8, 3 updates, 1 thread |
| `fc-k` | FC-K, n8, 4 layers, d128 | CPU FP32, B2/L8, 3 updates, 1 thread |
| `hb-settle` | HB X, compact-9, n8/hubs4, d128, region sharing, Settle | CPU FP32, B2/L8, 3 updates, 1 thread |
| `hb-timed` | Same HB geometry, Timed | Same small CPU runtime; flush is included in every finite window |

Templates are representative choices, not an exhaustive configuration menu or a
claim that these values maximize throughput. A measured recommendation must name
its geometry, mode, precision, hardware, B/L, memory boundary and evidence.
The small CPU entry stays separate from capacity-oriented device recommendations.
The `npu-*` presets below have `execution.device=null`, which requires an explicit
`--device` even for dry run. Selecting a preset never discovers or chooses a card.

Resolution order is:

1. Built-in safe execution defaults.
2. `--preset NAME` or `--preset /path/to/preset.json` supplies model and runtime
   settings; omission selects `default`.
3. `--config /path/to/model.json` replaces the **entire model configuration** and
   retains the selected preset's runtime settings.
4. Explicit CLI flags replace their individual model or runtime values, including
   `--no-checkpoint-regions`, `--no-checkpoint-reception` and `--no-foreach`.

`--checkpoint-reception` controls reception recomputation independently of
`--checkpoint-regions`. `--ssd-summary-backend block|prefix` selects the SSD
summary implementation. Both are model options, printed in the resolved config,
saved in checkpoints and checked against explicit model flags during resume.

The entry prints the final full `ModelConfig`, resolved topology and clock,
requested runtime, actual parameter dtypes/devices, optimizer group settings,
the paths/SHA256 hashes of the chosen JSON files, and explicit CLI overrides.
Recommendation metadata describes the preset as written; changed settings need
their own matching evidence. `--dry-run` validates
settings/topology without constructing model weights or initializing a device.
With `--resume`, dry run leaves checkpoint decoding for the actual run and labels
the checkpoint's model authoritative.

```sh
python examples/train.py --preset hb-settle --config configs/hb-multiscale-x-settle.json \
  --region-batch-size 4 --checkpoint-reception --ssd-summary-backend prefix \
  --batch 2 --length 8 --steps 3
python examples/train.py --preset fc-k --mode grad_forward --batch 1 --length 16
python examples/train.py --preset hb-timed --mode inference --precision bf16
```

| Mode | Graph | Backward and AdamW | Continued state |
| --- | --- | --- | --- |
| `train` | Enabled; training mode | One real backward/update per step | Detached after each window |
| `grad_forward` | Enabled; training mode | Neither performed; no optimizer | Detached after each window |
| `inference` | Inference context; evaluation mode | Neither performed; no optimizer | Carried without an autograd graph |

All modes include Timed `flush` after each finite input window. Its readout is part
of the mechanism scalar, so training differentiates through the tail. The next
window uses the tail's advanced private-state clock. These windows therefore
include explicit empty drain periods; they do not represent a stream with no
inter-window gap. Settle has no flush. For prefill/chunk/decode measurements with
the specified timing boundary, use [`benchmarks.continuous`](../benchmarks/continuous.py)
and the [benchmark contract](benchmark-contract.md).

## Measured NPU training choices

The [training search](evidence/training-search-20261118.md),
[20-update records](evidence/training-continuous-20261118.md),
[fixed-start numerical checks](evidence/training-paths-20261118.md) and
[validation record](evidence/training-validation-20261118.md) own these choices.
They are finite observations on this host's 64 GiB NPUs under a 48 GiB sampled
allocated-memory budget. It excludes reserved memory and allocations outside
Torch, and does not provide hard memory isolation. Records identify
`exclusive_cooperative_queue; external jobs not controlled`; they are not a claim
that external workloads could not contend. Each preset retains the producing
source, record SHA256, checkpoint hash, qualification and resource identity.

| Preset | B/L | Steady input positions/s | Peak allocated/reserved GiB | Selection and paired restore |
| --- | --- | ---: | ---: | --- |
| `npu-hb-settle-fp32` | 24/256 | 2,074.5 | 38.285 / 42.998 | Primary Settle choice; restored update within original threshold |
| `npu-hb-timed-fp32` | 12/128 | 317.4 | 41.653 / 46.387 | Primary Timed choice; restored update within original threshold |
| `npu-hb-0p5b-fp32` | 1/128 | 96.08 | 15.372 / 16.506 | Primary scale choice; restored update within original threshold |
| `npu-fc-k-bf16` | 128/256 | 38,248.5 | 25.692 / 29.936 | Secondary BF16 reference; restored update retains finite deviation |
| `npu-hb-settle-bf16` | 32/256 | 2,078.2 | 41.835 / 50.625 | Alternative; similar throughput with a higher peak at this B |
| `npu-hb-0p5b-bf16` | 1/128 | 90.61 | 13.180 / 14.641 | Alternative; lower peak and lower throughput than scale FP32 |

Preflight observed about 2.8–3.3 GiB of idle-device usage outside Torch. That is
an admission snapshot, not a measured running HBM peak; do not infer free HBM
from allocated peak alone. Settle BF16 B32's 50.625 GiB reserved peak makes it a
less conservative choice than selected FP32 B24. Adding downstream task heads
and data storage requires checking the actual remaining device memory again.

FC-K FP32 has separate [fresh-window capacity observations](evidence/capacity-search-20261118.md):
B8/L512 reached 16,352.99 input positions/s at 4.067 GiB allocated, and B8/L1024
reached 18,400.58 at 7.976 GiB, using source `f486a56e`. These are earlier
zero-private-state updates, not the latest 20-update continuous qualification.
An admitted-device starting configuration is:

```sh
python examples/train.py --preset fc-k --device npu:0 --precision fp32 \
  --batch 8 --length 512 --threads 4 --dry-run
```

Remove `--dry-run` only within the normal admitted, bounded execution workflow.
This is a starting setting, not a new measured preset. The large-batch FC
20-update record above remains BF16; overriding its precision does not transfer
its throughput, memory or restored-update qualification to FP32. A comparable
FP32 FC continuous capacity choice remains a follow-up measurement.

FC-K retains its original computation options: n8, 4 layers, d128, independent
parameters, `region_batch_size=1`, both recomputation options false and SSD
summary `block`. The selected small HB models are X, compact-9, n8/hubs4, d128
with region sharing. The scale model is X, profile 9, n32/hubs4, d128 with
independent parameters, totaling 494,110,704 parameters.

All five HB presets use `region_batch_size=4`, `checkpoint_reception=true`,
`ssd_summary_backend="prefix"`, `checkpoint_regions=false`, Attention `auto` and
`local_attention_backend=null`. The CPU examples and other execution candidates
remain available. Timed BF16 prefix retains structural path differences in the
fixed-start checks and is not selected as a default. The FP32 path checks also
retain finite deviations; a recommendation does not establish full numerical
equivalence. [Cost attribution](evidence/training-costs-20261118.md) and
[other-mode observations](evidence/training-modes-20261118.md) have separate scopes.

The table uses total input positions divided by total `complete_seconds` for
updates 3–20 (`observations[2:20]`). Allocated and reserved peaks are each their
maximum across all 20 updates. Timed
flush is included in forward, loss, backward and clock advancement. Complete
time includes AdamW, synchronization and detached continuation; initialization,
input construction/transfer, warmup, intrusive audits and checkpoint work are
outside that timing boundary. All 20 updates recorded finite output, parameters,
gradients, optimizer and private state, with detached continuation and optimizer
steps reaching 20. These finite checks do not replace the retained comparisons.

All six records have exact saved data and exact next readout/state checks. FC-K BF16 still
has a paired restored-update `finite_deviation` at the unchanged
`atol=1e-6, rtol=2e-3`; the other five paired updates are within that threshold at
these saved points. Six separate restore processes subsequently completed finite
updates with optimizer steps 20→21 and no increment errors. Those single-owner
processes have no original-owner comparator and do not establish trajectory
equivalence. Provenance uses the actual continuous producer source and checkpoint
hash; the evidence records corrections to the checkpoint-source CLI labels in
the three small-HB restore records without rewriting those raw files.

After device admission, resolve a named choice and run it inside the existing
bounded launcher:

```sh
python examples/train.py --preset npu-hb-settle-fp32 --device npu:0 --dry-run
python examples/train.py --preset npu-hb-settle-fp32 --device npu:0
python examples/train.py --preset npu-hb-timed-fp32 --device npu:0 \
  --checkpoint artifacts/NEW-timed-training.pt
python examples/train.py --preset npu-hb-timed-fp32 --device npu:0 \
  --resume artifacts/NEW-timed-training.pt --steps 3
```

These named presets select 20 updates, four ATen threads, AdamW lr1e-4/foreach and
seed 751. They transfer the measured model/runtime choices to the public example.
The example has its own fresh synthetic sequence, loss normalization and finite
diagnostic checks; its elapsed time or trajectory is not the recorded benchmark.
Use the frozen commands in the continuous evidence to reproduce the table.
`--checkpoint` additionally creates two model owners for the paired check; its
cost and memory are outside the table's update boundary. The example does not
enforce the measured 48 GiB budget itself. Explicit overrides remain available
and need matching evidence for a new performance or numerical claim.

## Other modes and starting settings

The [mode-specific A–D tables](evidence/training-modes-20261118.md) measure these
finite starting points. They do not select a universal fastest implementation:

| Objective | Starting setting and measured scope |
| --- | --- |
| Sustained training | Start with the FP32 choices above; FC FP32 currently uses the separately labelled capacity starting point. Recheck reserved/actual free memory after adding the task head and data. |
| Offline inference throughput | FC BF16 B128/L256 and Timed FP32 B4/L128 have separate no-grad observations; compatible batches/prefix remain candidates, while recomputation provides no inference saving. |
| Grad-enabled forward | The same two shapes have separate measurements; include graph construction and retained activations even though no backward/optimizer runs. |
| First backbone output / prefill | B1, prefill128 in one chunk has explicit output-ready timing. Select a smaller chunk only after checking launch overhead and when a usable output actually appears. |
| Cached decode | B1, eight single-position calls after prefill, with real KV/SSD state. Keep this separate from a large-batch fresh-window throughput result. |

For the example, `--mode inference` and `--mode grad_forward` select the first
two forward-only modes. For precise first-output/prefill/decode boundaries use
`benchmarks.continuous`, including the final output-assembly synchronization.
Timed input and flush remain separately reported and included in whole-window
cost; no task head is present, so backbone output-ready latency is not LM TTFT.
The large training presets should not be used as automatic latency choices.

## Save, verify and resume

```sh
python examples/train.py --preset fc-k --batch 1 --checkpoint artifacts/example-first.pt
python examples/train.py --resume artifacts/example-first.pt --batch 1 --steps 3
python examples/train.py --resume artifacts/example-first.pt --batch 1 --steps 3 \
  --lr 0.0002 --no-foreach --checkpoint artifacts/example-next.pt
```

`--checkpoint` requires training and a new path. It uses `save_checkpoint` to
store the actual model configuration/options, parameters, private state and
AdamW slots. It checks exact decoding and the next forward, then performs a real
next update on both the original and restored owners. The saved file remains at
the boundary **before** these two verification updates. The original in-memory
owner advances once beyond the requested number of steps during this check.

The original update threshold stays `atol=1e-6, rtol=2e-3`. Finite deviations are
printed as `finite_deviation` with parameter failures and optimizer differences;
they are not relabelled as a pass. Nonfinite values raise an error. Exact saved
data and next-forward checks do not imply a bitwise future optimizer trajectory.
The [precision policy and retained failures](precision.md) remain applicable.

`--resume` loads a trusted single-device checkpoint through the public API. In
training mode it requires a saved single-group AdamW state; forward-only modes
do not use its optimizer. Saved model options are authoritative. Explicit
`--preset`/`--config` asserts equality with the full selected model configuration;
individual model flags without either assert only their selected saved fields.
They cannot silently change a saved model. Placed checkpoints use the
[placement API](placement.md), outside this small entry.

Resume runtime mode, precision, B/L, thread count and seed come from the current
invocation. Use a batch size compatible with the saved private state. Saved
optimizer learning rate and foreach settings survive the runtime preset; only
explicit `--lr` and `--[no-]foreach` override those saved values. The printed
runtime and optimizer group show the resulting settings.

The checkpoint format does not save precision policy, RNG or a data sampler.
Each invocation generates a **new synthetic sequence** from its invocation seed;
resume restores model/state/AdamW, not the previous RNG/data position. The example
requires FP32 parameter storage for both FP32 and BF16 execution, following
[`tide.precision`](precision.md). It does not provide an FP64 training preset.

## Device and performance boundaries

Device selection is explicit. The small presets default to CPU, the measured
NPU presets require `--device`, and the entry never chooses an idle NPU. After
device admission under the [execution contract](execution-flows.md), an explicitly
selected NPU uses the same entry and disables HF32:

```sh
python examples/train.py --preset hb-settle --device npu:0 --precision fp32 \
  --batch 2 --length 8 --steps 3
```

This command demonstrates configuration, not a hardware optimum. Freeze the
source, configuration and command before a long job; use the existing detached
launcher, finite bounds and resource/admission records. Device-specific
recommendations must include actual allocated/reserved peaks, update timing,
continuation/restore outcomes and any sharing/contending workloads. CPU settings
cannot establish CUDA or NPU qualification. CUDA remains unqualified by this entry.

These combinations are different candidates and must remain distinct in
commands, preset provenance and performance evidence:

| Candidate | Explicit model options |
| --- | --- |
| Regional batching plus checkpointing | `region_batch_size=4`, `checkpoint_regions=true`; retain the chosen Attention backend |
| Historical three-option `combined` | The same two options **plus** `local_attention_backend="math"` |
| Selected HB training combination | `region_batch_size=4`, `checkpoint_reception=true`, `ssd_summary_backend="prefix"`; region checkpointing stays false and local Attention is inherited |

Local math Attention can change arithmetic and device kernel dispatch. A result for
`combined` cannot be attributed to just batching plus checkpointing. Inference,
grad-enabled forward and complete training need their own matched observations;
reducing peak memory may permit a larger batch while changing per-step cost.
The [regional performance evidence](evidence/native-regional-performance-20261118.md)
and [rounding investigation](evidence/capacity-rounding-20261118.md) retain their
original finite numerical failures. No template widens those thresholds or
claims downstream quality.
