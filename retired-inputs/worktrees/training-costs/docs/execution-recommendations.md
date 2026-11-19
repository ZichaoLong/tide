# Training entry and execution recommendations

[`examples/train.py`](../examples/train.py) is a finite example using the installed
public Tide API. With no arguments it trains FC-K on CPU for three updates and
prints the resolved model, clocks, runtime, precision policy and each update as
JSON lines. Synthetic normal inputs and the squared readout scalar demonstrate
the mechanism. Task heads, data, losses and training recipes remain downstream.
The entry performs diagnostic finite checks; its elapsed time is not a benchmark.

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
| `fc-k` | FC-K, n8, 4 layers, d128 | CPU template; hardware recommendation pending measurement |
| `hb-settle` | HB X, compact-9, n8/hubs4, d128, region sharing, Settle | CPU template; hardware recommendation pending measurement |
| `hb-timed` | Same HB geometry, Timed | CPU template; flush is included in every finite window |

Templates are representative choices, not an exhaustive configuration menu or a
claim that these values maximize throughput. A measured recommendation must name
its geometry, mode, precision, hardware, B/L, memory boundary and evidence.
The small CPU entry stays separate from capacity-oriented device recommendations.

Resolution order is:

1. Built-in safe execution defaults.
2. `--preset NAME` or `--preset /path/to/preset.json` supplies model and runtime
   settings; omission selects `default`.
3. `--config /path/to/model.json` replaces the **entire model configuration** and
   retains the selected preset's runtime settings.
4. Explicit CLI flags replace their individual model or runtime values, including
   `--no-checkpoint-regions` and `--no-foreach`.

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
  --region-batch-size 4 --checkpoint-regions --batch 2 --length 8 --steps 3
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

Device selection is explicit. Presets default to CPU and the entry never chooses
an idle NPU. After device admission under the [execution contract](execution-flows.md),
an explicitly selected NPU uses the same entry and disables HF32:

```sh
python examples/train.py --preset hb-settle --device npu:0 --precision bf16 \
  --batch 2 --length 8 --steps 3
```

This command demonstrates configuration, not a hardware optimum. Freeze the
source, configuration and command before a long job; use the existing detached
launcher, finite bounds and resource/admission records. Device-specific
recommendations must include actual allocated/reserved peaks, update timing,
continuation/restore outcomes and any sharing/contending workloads. CPU settings
cannot establish CUDA or NPU qualification. CUDA remains unqualified by this entry.

The two combinations below are different candidates and must remain distinct in
commands, preset provenance and performance evidence:

| Candidate | Explicit model options |
| --- | --- |
| Regional batching plus checkpointing | `region_batch_size=4`, `checkpoint_regions=true`; retain the chosen Attention backend |
| Historical three-option `combined` | The same two options **plus** `local_attention_backend="math"` |

The third option can change arithmetic and device kernel dispatch. A result for
`combined` cannot be attributed to just batching plus checkpointing. Inference,
grad-enabled forward and complete training need their own matched observations;
reducing peak memory may permit a larger batch while changing per-step cost.
The [regional performance evidence](evidence/native-regional-performance-20261118.md)
and [rounding investigation](evidence/capacity-rounding-20261118.md) retain their
original finite numerical failures. No template widens those thresholds or
claims downstream quality.
