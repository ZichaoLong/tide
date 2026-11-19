# Rebuilding the joint comparison

The implementation follows [the comparison contract](benchmark-contract.md).
Use an installed checkout and the environment/admission rules in
[development](development.md) and [execution](execution-flows.md). For example:

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m benchmarks.run \
  --config configs/fc-k.json --device cpu --precision bf16 \
  --batch 8 --length 128 --threads 8 --warmup 2 --steps 5 --output NEW.json
```

The default subjects are Tide and all four joint Transformer classes. Each
mode's initial audit fixes its own effective touched-parameter target, with
identical CPU-generated initial model and input. Inference disables autograd;
grad-forward and training enable it for the initial audit, without updating.
The baseline stays fixed through warmup and measured updates. A post-timing
audit records the resulting work instead of silently matching again.

`matching.py` fixes block count to node count or spatial depth, then searches
widths that are multiples of 16, at least 32, and FFN widths that are multiples
of 8 between 2d and 6d. Every selected weight participates in the baseline.
Head widths are 16, 32 or 64. The nearest legal parameter count, both target
dimensions, actual counts and signed relative deviations are stored under
`joint_matches_by_mode`; a large unavoidable mismatch remains visible.
Different widths use identical random values on their shared feature prefix,
with actual per-subject input hashes. Neither matching nor timing establishes
mathematical equivalence or model quality.

`--subjects tide ssd_layers ssd_parameters` retains the FC-S SSD references.
The old Transformer subjects `transformer_layers`, `transformer_nodes` and
`transformer_parameters` remain explicitly auxiliary single-axis comparisons.
They are never assigned A/B/C/D identities.

## Counting without perturbing timing

`statistics.py` partitions actual unique Parameter objects into disjoint
owner/group/source-class slices. It counts positive reception, memory
preparation, description/regional computation and positive Full separately.
Counts describe effective semantic touches, not nonzero gradients or physical
loads from padded matrices. The textbook base/D/H active proxy is retained,
excluding ingress as the textbook specifies; executable total and effective
counts include enabled ingress projections.

Settle deduplicates shared slices within each sample/input step across settled
layers. Timed deduplicates within each sample/logical tick across regions,
including flush ticks when enabled. B/D use the mean of those counts. The
cumulative count per actual input position and the union across time/batch are
reported separately. Timed's per-input accumulation can exceed unique model
parameters; it is repeated work, without an output causal attribution claim.

The audit also records positive messages, candidate/Full actions, attention
pairs, selected versus executed packed rows, and NodeLinear multiply-add
pairs including adapters/padding. These are not a complete FLOP model. A
separate ordinary forward records dispatch-visible operator/device/dtypes and
logical bytes materialized by copy/index/concatenate operations. It excludes
the semantic audit's diagnostic CPU copies. Logical output bytes do not measure
hardware traffic or all parameter/activation reads.

`--audit-only` produces these records and matches without timing. The independent
small checks are `tests/test_benchmark_contract.py`: mathematical shape formulas
for sparse FC, shared/private Full contributions, complete positive support,
real baseline tensors, and exhaustive small matching candidates.

## Synthetic support and timing boundaries

`--activity sparse03`, `sparse10` or `high` applies only to FC with legal quad.
The reproducible initialization sets score readouts to zero and descriptor
offsets to +16 for the first designated nodes and -16 for the others. It leaves
the legal mass budget unchanged. At n=32, k=1 this creates 1/32, 3/32 or full
support; actual support is recorded, and training may change it. These are
constructed pressure cases, separate from `--activity natural`, not learned
sparsity. Shannon/pool are never converted to sparse support fixtures.

Every measured iteration starts from zero private state. Timed includes its
whole flush unless `--no-flush` explicitly excludes it; input throughput always
uses B×L actual inputs. This is prefill throughput and whole-window latency,
not decode or first-output latency. Use `benchmarks.continuous` for incremental
state and real Transformer KV caches.

Training includes preparation, packing, readout/loss, backward, AdamW and
synchronization. Construction, transfers, audits and checkpoint are excluded.
BF16 uses the explicit library autocast policy and genuine BF16 matrix outputs;
actual parameter/gradient/optimizer storage is recorded. FP32 master parameters
and AdamW slots remain FP32, so the parameter memory floor is not halved.
Accelerator allocated/reserved peaks are reset after warmup and captured before
the post-timing audit. CPU RSS is explicitly a process-lifetime peak, not an
isolated per-subject allocation; use separate processes for capacity comparison.
Outer finite completion is not a numerical-equivalence qualification.

## Finite suite driver

`programme.py` runs explicit child commands serially, with independent process
logs/results, per-child timeout, preserved internal failures and an atomic
`programme.json`. Nested sessions remain tracked by PID/start identity; adopted
orphan descendants are included in completion and shutdown. It belongs inside the durable launcher and device admission;
it does not allocate hardware or establish its own RAM limit. A plan can be
reviewed without loading Torch:

```sh
python -m benchmarks.programme --suite representative \
  --configs configs/fc-k.json configs/fc-s.json \
  --precision fp32 bf16 --batch 8 --length 128 --threads 8 \
  --warmup 2 --steps 5 --child-timeout 1800 --plan-only --output-dir NEW-PLAN
```

After freezing source, use the same command with a new output directory and
without `--plan-only` inside the authorized service. `representative` defaults
to the seven registered points and appends `ssd_parameters` for FC-S. Explicit
`--configs`, `--subjects`, `--modes` and precision lists narrow a finite run.

`sparse`, `combinations` and `shape` take one base config. They respectively
create four support fixtures at FC n32/k1; a base plus three useful reception,
allocation and sharing combinations; or a base plus independent B×4, L×4 and
n×2 changes (n stops at 128). Generated configurations and hashes are preserved
before execution. `scale` requires explicit config paths and a prior capacity
budget. `hotspots` and `projections` invoke the bounded candidate tools.

`continuous` forwards `--prefill-chunk`, `--decode`, `--steps` as stream updates,
and optionally `--checkpoints` for Tide training. It defaults to Tide;
`--continuous-joint A B C D` adds baselines matched by each child's initial
audit of that exact prefill/decode/flush stream and mode. Optional
`--continuous-transformers FILE.json` supplies explicit lists of
`label/d/layers/ffn_width/head_dim` specifications and their reference identities;
the driver preserves them and does not invent a joint-match qualification for
different inputs. Keep the matching workload and reference scope with those
specifications.

## Continuous and capacity boundaries

`benchmarks.continuous --subject transformer --joint-match A` (likewise B/C/D)
audits the exact initial chunk/decode/flush schedule before choosing the baseline.
Input features use a shared seeded prefix across widths. Each measured stream
continues the preceding private state; train retains full BPTT within the stream
and detaches after the actual update. A Timed segment ends with flush, so its
successor starts at the advanced, flushed clock. First output means a backbone
output, without a language-model task head. Checkpoint time is separate, and the
restored update includes the complete next Timed flush in its loss.

The four `configs/continuous/*-small.json` cases are finite d64 CPU continuation
workloads: four FC nodes or 17 Multiscale nodes. They are distinct from the larger
NPU main configurations; their timings are not a same-shape CPU/NPU comparison.
Both retain the real 128-position window and use legal K/S/X state mechanisms.
Small independent whole/chunk VJP checks are available through
`python -m scripts.check_continuation --device cpu --precision fp64 --output NEW.json`.
The FP32/BF16 versions retain numerical, support, metadata and connectivity failures.

`python -m benchmarks.capacity --placement-partitions 2 --output NEW-budget.json`
counts scale parameters on the meta device without allocating their payloads.
It records FP32 master/gradient/AdamW floors, workspace allowances, private-state
bounds and exact HB layer partition ownership. The planning allowance is not an
allocation guarantee. `scripts.check_placement --mode inference` or `--mode train
--no-foreach` measures a large placed model without constructing a duplicate
one-device reference; see [placement](placement.md). Capacity execution does not
establish trajectory equivalence or large-checkpoint qualification.

`benchmarks.attention_sites` measures the three post-projection Attention families
with identical fixtures. Its forward/backward boundary has no optimizer and is
not complete training. Normal samples and the intrusive dispatch/dtype audit are
separate; composite device `matmul` is included in the audit. Primitive finite
execution does not replace independent formula/VJP qualification.
