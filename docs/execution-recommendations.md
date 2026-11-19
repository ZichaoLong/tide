# Training entry and execution recommendations

Actual training prioritizes **FP32**. The default entry, [`examples/train.py`](../examples/train.py),
runs three short CPU training updates. It uses the installed public `tide` API, synthetic inputs
and a squared-readout mechanism objective; task heads, data, losses and learning recipes remain downstream.
The named NPU presets select representative measured shapes and require an explicit admitted device.
They do not select a card automatically or claim a universal best configuration.

```sh
python examples/train.py
python examples/train.py --list-presets
python examples/train.py --preset npu-hb-settle-fp32 --device npu:0 --dry-run
python examples/train.py --preset npu-hb-light64-fp32 --device npu:0 --dry-run
```

Use an installed private environment, or `PYTHONPATH=src` in this checkout. The wheel contains
`tide`; examples, presets and benchmark drivers are checkout tools. Preset lookup is relative
to the script, so its absolute path also works from another directory.

## FP32 training choices

The [100-update evidence](evidence/reception-training-20261118.md) owns the following values.
Each run carries actual private history, detaches between windows, and includes forward/loss,
backward, AdamW and synchronization. Timed includes a final flush in each finite training window.
The stable rate is total input positions divided by complete update time for updates 3–100.
Constructor/input transfer, finite diagnostics and checkpoint verification are outside that timing.
The example uses its own synthetic sequence and diagnostic checks; its printed time is not this benchmark.

| Preset | Parameters | B/L | Input positions/s | Peak allocated/reserved GiB | Paired restored update |
|---|---:|---|---:|---:|---|
| `npu-fc-k-fp32` | 9.33M | 128/256 | 30333.2 | 37.361/39.576 | within_threshold |
| `npu-hb-0p5b-fp32` | 494.11M | 4/128 | 267.4 | 33.840/35.617 | within_threshold |
| `npu-hb-light64-fp32` | 17.88M | 24/256 | 1618.2 | 38.537/44.904 | within_threshold |
| `npu-hb-settle-fp32` | 15.69M | 24/256 | 2643.8 | 39.277/43.508 | within_threshold |
| `npu-hb-timed-fp32` | 15.69M | 12/128 | 369.7 | 42.193/47.248 | within_threshold |

These are this host's 64 GiB NPUs, admitted through the cooperative exclusive queue;
external jobs and the shared host are not controlled. The sampled 48 GiB allocated budget is not
hard memory isolation. Reserved memory is separate; earlier idle observations included roughly
3 GiB outside Torch. Recheck actual free memory after adding downstream heads/data/workspace.
The chosen points retain useful headroom, rather than selecting the last shape that just fits.

FC is K/n8/layers4/d128 with independent parameters. Small HB models are X/compact-9/n8+h4/d128
with region sharing; the 0.5B model uses profile9/n32+h4/d128 and independent parameters.
All HB FP32 choices use compatible region batch4, Reception-only recomputation, prefix vector
summaries and doubling matrix scan. The light64 choice additionally selects the incoming_light_64
mathematical binding, fused QKV and structural padding mask. The other presets retain source_residual.
Whole-region recomputation is off; Attention is auto; local Attention is inherited; indexing remains gather.
ModelConfig defaults and the small CPU examples remain explicit separate choices.

Light64 without Reception recomputation reached 1721.8 positions/s at B16/L256 in the short
zero-history search, but already used 44.1 GiB allocated. The selected B24 recomputed point has
more margin and a 100-update result. Both remain available. At 0.5B, B1→2→4 improved throughput;
the finite search stopped at B4, so this is not a proof of the largest feasible or fastest batch.

Each preset records its actual source, complete record SHA256, checkpoint SHA256, memory peaks,
numerical outcome and new-process restore observation. Exact checkpoint data/next forward,
a within-threshold next update and bitwise future optimizer trajectories are different claims.
[Cross-path finite deviations](evidence/reception-numerics-20261118.md) remain; BF16/placed
qualification is [separate](evidence/reception-validation-20261118.md).

## Training qualification boundaries

Training execution, numerical equivalence and downstream learning quality are separate claims.
Some retained failures involve gradients, updated parameters or discrete support; they have not
all been explained as rounding. A fixed path can complete training and restore while a change
of mode, batching, precision or placement produces a different trajectory.

| Path / setting | Evidence and practical boundary |
|---|---|
| Default CPU example | FP32 FC-K, gather/rectangular FFN and one thread have public training/restore checks. This is a small mechanism example, not large-model or task-quality qualification. |
| Recommended single-NPU FP32 presets | The five fixed-source/configuration 100-update runs above have paired actual restored updates within their original thresholds. Cross-path and inference/grad-forward differences remain, including baseline observations; keep measured settings fixed. |
| Narrow KV memory family | Medium training uses d192/KV32, with B8/B16 continuing training and paired restore evidence. The d768/KV128 7.268B instance has local formula/VJP and inference evidence, but no complete 9B-topology training/restore qualification. FFN-adjusted near-parameter variants also have inference-only evidence. |
| FFN grouped weights / power2 packing | Optional W/P/WP reduce measured medium NPU peak memory, but have no same-batch throughput gain and retain forward/update failures. CPU packing recovery did not finish within the artifact budget; medium NPU W qualification did not begin. Keep gather/rectangular as the training default. |
| BF16 | Some fixed HB workloads train and restore within threshold. Other cases retain support, memory-time, gradient and update differences, including FC/placed restore failures. BF16 is secondary and is not certified as a drop-in FP32 replacement. |
| Memory-only sharing | Small/medium training and candidate-local restore have evidence. The 9B region-batch1/4 inference pair changes support; keep the measured size4 setting for that instance, without claiming 9B training equivalence. |
| Timed and placement | The older Timed FP32 preset has its own 100-update evidence. New memory-candidate Timed/multi-device combinations lack qualification; Timed BF16 and some two-device restores retain failures. Four-device inference does not qualify four-device training. Direct NPU transport retains a backward failure; host-staged transport is a separate path. |
| Exact training resumption | Public checkpoints retain model/configuration, AdamW and private state. The example does not save RNG, sampler, scheduler or data-stream position; downstream training must preserve its own recipe state. Placed checkpoints use the placement API. |

See the [training-impact review and duplicate-code retirement](evidence/code-retirement-training-impact-20261119.md)
for source-scoped failures, fixed defects and cleanup decisions. Useful memory-saving candidates,
reference paths and diagnosis entry points remain available; removing duplicate frozen checkouts
changes storage availability, not the implementation or qualification of these paths.

## Mode-specific choices

Use the [resident inference measurements](evidence/reception-inference-20261118.md) and
[A–D three-mode tables](evidence/reception-training-20261118.md) for the actual numbers.
Inference returns backbone readouts, without a task head, token sampling or serving queue.

| Objective | Starting choice and boundary |
|---|---|
| Full training throughput | Select the matching FP32 preset above; keep actual history, backward/update and memory headroom in the comparison. |
| Grad-enabled forward | Measure graph construction and retained activations separately. It contains no backward or optimizer. B8/L128 has separate tables for all three incoming profiles. |
| First backbone output | Use B1 and a smaller prefill chunk where the first partial output is useful. The 9B short-prefix point uses chunk32; compare complete prompt latency separately. |
| Complete prompt readiness | Larger chunks can amortize scheduling. The 9B 512-position prompt uses chunk128; Timed readiness includes graph-depth delay and may require later calls or the final drain. |
| Single-request cached inference | B1, actual KV/SSD continuation; doubling uses the direct one-position SSD recurrence. The 256-call points report first increment, steady rate and p95 separately. Optional static_view gave about 9.5% higher measured 9B B1 decode throughput, with essentially unchanged allocated peak; it remains explicit. |
| Batched inference / offline stream | Short cached requests have B2/4/8 measurements: 9B on one NPU and 17B on two. Larger batches increase total throughput and step latency. For 128-position streams, use the separately recorded 1B B2 and Timed B4 points; short B8 results do not qualify longer streams. |
| Large multi-device inference | Whole-region placement provides capacity. The 17B short-request B4/B8 follow-up was faster on two devices than four; four devices roughly halved per-device allocated peaks. Keep batching, placement, concurrency and replicas as separate choices. |

The [fixed-depth and batch follow-up](evidence/transformer-depth-batch-20261118.md) uses FP32,
prompt128/chunk32/decode16. Tide 9B B8 reached 12.34 positions/s at 46.83/48.63 GiB
allocated/reserved on one NPU; 17B B8 reached 10.07 positions/s at 47.32/50.85 GiB per-card
peaks on two NPUs. Both allocated peaks approach the 48 GiB sampled budget; B4 retains
more room for longer history or downstream additions. B1 remains the lower-latency measured
choice. Four-card 17B B8 reached 8.52 positions/s at 22.93/24.75 GiB, trading throughput
for capacity in this comparison. The B4 placement pair retained a Full/support difference;
B8 work/support/time summaries matched while packing differed. This is not four-card
training/restore qualification, a maximum-throughput search, or a replica-parallel measurement.

The later [fixed-budget decode search](evidence/decode-capacity-20261118.md) formally
confirmed Tide B8 at 12.29 positions/s for 9B on one NPU,
B8 at 9.90 for 17B on two, and B34 at
24.28 for 17B on four. These are total batch throughput choices
within the fixed 48/56 GiB allocated/reserved budget and the same short request.
Four devices enable a larger batch and higher total throughput here; the earlier
same-B8 comparison still favors two devices. Keep B1 for the measured low-latency
objective and reserve more memory when adding a downstream head or changing context.
All points keep the measured gather/doubling/prefix/region-batch4 candidates; the search
did not tune all operators, Transformer shapes or replicas. The report gives each
point's latency, per-card rate, budget boundary and retained-state memory separately.

`benchmarks.inference` keeps real state across prefill, decode and stream chunks, consumes/releases
one output at a time, and drains Timed only once at the finite stream end. In contrast, the training
example and `benchmarks.continuous` use finite windows with their declared flush boundary.
First backbone output is **not** language-model TTFT. Cold invocation excludes model construction
and weight transfer; a later subject in the same process can inherit runtime caches.

A–D jointly match block count/depth and total/actual touched parameters. They are different
mathematical models; parameter matching does not match FLOPs, padding, scheduling or model quality.
The inference report preserves exact schedules, active targets, operator policy, device mapping,
samples and memory rather than ranking unmatched B/L or precision points together.

## Memory-binding candidates

The [five-family comparison](evidence/memory-candidates-20261119.md) adds explicit
model choices; it leaves the training entry and existing presets unchanged.
For the 9B graph (768 D, 80 H, d768, summary192, FFN3072), start downstream
quality trials with FP32 `9b-kv128.json` or the less narrow `9b-kv256.json` in
[`configs/memory-candidates`](../configs/memory-candidates). These jointly narrow
Q/K/V total width, not GQA. At B8/prompt128/chunk32/decode16 they reached
18.400/16.736 positions/s and 33.492/36.152 GiB allocated, versus X's
11.713 positions/s and 46.828 GiB. Parameters and mathematical functions differ.

For continuous medium training (d192, FFN768), the corresponding KV32/KV64
B16 points reached 949.08/890.78 positions/s in 20-update runs, measured over
updates 3–20, with paired
restored updates within threshold. B8 also has actual fresh-process updates;
B16 does not add that separate process check. These finite points provide a
starting batch, not a maximum-capacity or general optimum claim. They do not
qualify 9B training or the FFN3840/4016 near-parameter variants for training.

Full SSD is a useful lower-state inference candidate: B8 decode128 reached
16.539 positions/s. Its zero-state medium training diagnostic was slower and
used more memory than X, so select it by the intended workload. Memory-only
rank8 sharing mainly reduces parameter storage (6.936B with fixed FFN); it
retains private histories and independent FFNs, and did not improve medium
training throughput. Cross-region batching still repeats shared bases in BMMs.
Keep the measured `region_batch_size=4`: `region_batch_size=1` was slower at
the same B8 and introduced a 9B FP32 output-support difference, so the pair
is not strictly equivalent.

The report separately lists common B1/B8, each candidate's finite budget-best
batch, first backbone output, prefill, independent request throughput and
normal/cold memory. Small FP32 tolerance failures and BF16 support failures
remain. Use FP32 first, keep room for downstream state and heads, and validate
task quality before adopting a new binding.

## Retained execution candidates

| Model option | Meaning / selection limit |
|---|---|
| `region_batch_size=1/4` | Batch compatible same-layer regions within a parameter partition. Region Attention/allocation axes remain distinct. |
| `checkpoint_reception=true/false` | Recompute only Reception during training; no inference memory saving is implied. |
| `checkpoint_regions=true` | More aggressive whole-region activation saving, with additional recomputation; retained for tighter capacity. |
| `ssd_summary_backend=block/prefix` | Original block or parallel affine composition for vector summaries. |
| `ssd_scan_backend=block/incremental/doubling` | Original matrix scan; direct L1 recurrence; direct L1 plus short-block multiplication tree. Doubling can save time while retaining more backward intermediates. |
| `reception_qkv_backend=separate/fused` | Fuse compatible private/shared projections. Placed owner wrappers explicitly fall back to separate projections. |
| `reception_mask_padding=false/true` | Exclude nonexistent padded edges, preserving real zero-strength derivative roots. Native fusion can be slower or larger for some shapes. |
| `parameter_indexing=gather/static_view` | Optional exact static consecutive parameter-row views; dynamic Full and repeated/noncontiguous owners retain gather. Keep mode-specific evidence. |
| Attention `auto/math/tiled/sdpa/npu_fusion` | Choose for the actual weighted-prior, shape and backend. Explicit unsupported requests fail; a Flash-style API name is not a speed guarantee. |

All candidates remain configurable, including near-equal and memory-saving alternatives.
The [cost report](evidence/reception-performance-20261118.md) records positive and negative results,
actual operator paths, mathematical formulas and the remaining allocation/packing/indexing costs.
[Static indexing](parameter-indexing.md) has its own applicability and dynamic-selection boundary.
There is no complete C++ replacement: the full model is PyTorch; LibTorch remains a regional slice.
CPU/NPU evidence does not qualify CUDA, peer DMA/HCCL or a resident compiled scheduler.

## BF16 alternatives

BF16 remains secondary. Earlier measured presets remain available with their original 20-update
source and records:

| Preset | B/L | Stable positions/s | Allocated/reserved GiB | Original paired restore |
|---|---|---:|---:|---|
| `npu-fc-k-bf16` | 128/256 | 38248.5 | 25.692/29.936 | finite_deviation |
| `npu-hb-settle-bf16` | 32/256 | 2078.2 | 41.835/50.625 | within_threshold |
| `npu-hb-0p5b-bf16` | 1/128 | 90.61 | 13.180/14.641 | within_threshold |

These are not reruns of the new FP32 presets and do not establish a dtype-only speed ratio.
BF16 incoming Attention has retained output deviations; its combined path also has a support
difference after updates. Placed BF16 restored updates retain deviations. They are not selected
as new training defaults. Keep FP32 parameters/master weights, gradients and AdamW moments,
and use the explicit [precision context](precision.md), rather than casting the whole model to BF16.

## Resolve, save and resume

Resolution order is built-in safe runtime defaults, named/file preset, optional complete model
replacement by `--config`, then explicit flags. B/L, mode, device, optimizer and precision are
execution settings; they are not ModelConfig fields. Boolean `--no-*` overrides remain available.
The entry prints complete model/clock/runtime resolution, actual parameter dtype/device, optimizer
groups, preset/config paths and SHA256, plus explicit overrides. `--dry-run` validates without
constructing weights or initializing an accelerator.

```sh
python examples/train.py --preset hb-settle --aggregation incoming_light_64 \
  --region-batch-size 4 --checkpoint-reception --ssd-summary-backend prefix \
  --ssd-scan-backend doubling --reception-qkv-backend fused --reception-mask-padding
python examples/train.py --preset fc-k --mode grad_forward --batch 1 --length 16
python examples/train.py --preset fc-k --batch 1 --checkpoint artifacts/NEW-state.pt
python examples/train.py --resume artifacts/NEW-state.pt --batch 1 --steps 3
```

`--checkpoint` requires training and a new path. It saves the public model/state/AdamW checkpoint,
then compares exact saved data/next forward and one actual update of the original and restored
owners. The file remains **before** these verification updates; the in-memory original advances
one extra update. The update threshold remains atol 1e-6/rtol 2e-3 and finite deviations are printed.
Checkpoint verification can create two model owners and is outside measured update peaks.

`--resume` loads a trusted single-device checkpoint. Saved model options are authoritative:
explicit preset/config asserts full equality; explicit model flags assert only those saved fields.
Saved optimizer settings survive preset defaults; explicit `--lr`/`--[no-]foreach` can override them.
Use the saved batch-compatible state. Precision/mode/B/L/thread/seed belong to the current invocation.
RNG, sampler, scheduler and previous synthetic data position are not saved by this example.
Placed checkpoints use the [placement API](placement.md).

For long work, freeze source/config/command and use the [bounded execution workflow](execution-flows.md)
with quota, resource admission, background.slice, logs and terminal review. Changing preset values
does not transfer their recorded performance or numerical qualification to the new workload.
