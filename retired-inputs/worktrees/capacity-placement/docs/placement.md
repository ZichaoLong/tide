# HB layer and whole-region placement

`place_model` is a single-process layer split for the current strictly layered
HB instances. It partitions unique parameter rows, retains sharing within each
layer and leaves private memory with its owner. Actual predecessor tensor banks
move once per destination layer; their ordinary Torch copies retain autograd.
This path has no replicas, all-reduce or device-resident scheduler. A successful
copy does not establish a native peer-DMA implementation.

The default `transport="auto"` resolves to `host_stage` for multiple distinct NPU
devices and to `direct` for CPU/CUDA. Requested and resolved choices are recorded.
Explicit `transport="direct"` uses ordinary cross-device Torch copies;
`"host_stage"` uses synchronized accelerator→CPU→accelerator copies;
`"synchronized"` is a first-order custom copy with source/destination fences in
forward and backward. The latter is not a higher-order differentiation API.
These are bounded backend candidates, with hardware qualification attached to
its recorded source. The native direct NPU path has a recorded full-graph backward
event failure; isolated copy success does not qualify it. Host staging deliberately pays CPU transport and is never
reported as peer DMA. A device runtime error ends that process's qualification;
subsequent values from the same process cannot diagnose an independent failure.

Configure dtype and construct the placement before creating an optimizer:

```python
from tide import TideModel, place_model, placement_adamw

model = place_model(TideModel(config), ["npu:0", "npu:1"])
optimizer = placement_adamw(model, lr=1e-4, foreach=True)
result = model(inputs)  # inputs move to the first layer if necessary
loss = result.readout().square().mean()
loss.backward()
optimizer.step()
```

Use the normal `precision_context("npu:0", "bf16")` around forward/loss. It selects
the backend type's arithmetic for both devices; parameters/gradients/master
weights and AdamW slots remain FP32. Backward and optimizer run outside autocast.
`result` lives on the final layer's device; its state contains each layer's own
device tensors and original int64 times. Passing state retains the graph across
partitions and windows. Detach only at the declared optimizer boundary.

`placement_adamw` preserves a subtle ownership rule: if any shard of one original
Parameter has a gradient, disconnected shards receive zero gradients before the
update. Thus weight decay, moments and optimizer step counters cover the same
rows as the original dense Parameter. If every shard is disconnected, they all
remain `None`. Ordinary AdamW without this hook is not this training contract.
Closures are rejected; call backward explicitly. Checkpoint saving rejects an
ordinary optimizer on a placed model.

The default uses contiguous layer intervals chosen from node counts. Inspect
`placement_report(model)` for actual unique parameters and bytes per device.
An explicit `layer_partitions=[0,0,1,...]` can choose different contiguous cuts.
Every listed device must own a nonempty interval. CPU devices can repeat for a
logical partition check. Mixed CPU/accelerator execution and moving an already
placed model with `.to(...)` are outside this API; reconstruct and restore with a
new device mapping instead.

A placed checkpoint uses `tide-instances-2`, preserving its parameter and optimizer
shards and its layer partition. It does not gather all tensors onto one NPU:

```python
from tide import save_checkpoint, load_checkpoint

save_checkpoint(path, model, state=state, optimizer=optimizer)
restored, state, slots = load_checkpoint(
    path, placement_devices=["npu:0", "npu:1"])
optimizer = placement_adamw(restored, lr=1e-4, foreach=True)
optimizer.load_state_dict(slots)
```

Loading decodes tensors on CPU first, then maps each parameter/state shard to its
owner; optimizer loading moves slots to the matching parameter device. Without
`placement_devices`, saved logical partitions all map to `device` (CPU by default).
The number and layer boundaries of saved partitions remain fixed. Dense format-1
checkpoints retain their API. `canonical_state_dict(model)` exports the ordinary
dense parameter/buffer view on CPU for comparisons; this deliberately allocates
the full model and does not convert sharded optimizer slots.

Directed checks are `pytest -q tests/test_placement.py`. They compare logical CPU
partitions against ordinary execution, including all sharing profiles, mixed K/S,
Timed pending/flush, values, input/parameter gradients, three updates and a real
update after restore. Actual hardware qualification remains a separate record:

```sh
# Inside an admitted two-device job with remapped logical devices:
python -m scripts.check_placement --devices npu:0 npu:1 --precision bf16 \
  --output NEW-qualification.json
# Bounded inference at a large configuration; no duplicate one-card reference:
python -m scripts.check_placement --devices npu:0 npu:1 --precision bf16 \
  --mode inference --config LARGE.json --batch 1 --length 1 --steps 1 \
  --warmup 0 --output NEW-capacity.json
# Bounded continued AdamW capacity execution, without a duplicate reference:
python -m scripts.check_placement --devices npu:0 npu:1 npu:2 npu:3 \
  --precision bf16 --mode train --no-foreach --config HB-9B.json \
  --batch 1 --length 8 --steps 2 --output NEW-training-capacity.json
```

The inference mode measures construction/transfer separately, and then forward,
readout, necessary Timed flush and all-device synchronization. It records per-device
allocator peaks, process RSS, state storage, actual work and a separate operator
witness. No small check or one-step large execution qualifies all shapes, full
training capacity or sustained service. Optimizer bytes still follow the FP32
master/gradient/moment policy; a BF16 matrix policy does not halve that capacity.

Capacity `--mode train` defaults to B1/L8 and two real updates. It carries detached
private state between updates; any Timed flush contributes to the loss and its
advanced clock is the next segment's start. `--no-foreach`, available only in
this mode, avoids AdamW's tensor-list temporary allowance. It does not change
FP32 parameter, gradient or moment storage. There is one placed model and no
large duplicate reference, warmup update or checkpoint qualification.

Each update's complete wall time includes zeroing gradients, forward, necessary
flush, loss, backward, transport, AdamW, state detach and all-device synchronization.
It also includes intervening diagnostic stage persistence; this overhead is not
subtracted while asynchronous kernels may be running. Work counting replays the
identical pre-update parameters/input/state with gradients enabled outside timing;
the first update has an additional ordinary operator/dtype audit. Timed allocator
peaks start after those graphs are released, although reserved memory can retain
their allocator cache. Full parameter/gradient/optimizer/private-state finite scans,
actual payload/retained-storage counts and a bounded sampled update witness are
recorded outside timing. CPU construction, initial parameter hashing, placement
and optimizer/input preparation have separate costs. These records establish the
observed capacity and finite updates, not one-device trajectory equivalence.

To isolate transfer/backward failures, run each candidate in a separate admitted
process with `python -m scripts.probe_placement_copy --devices npu:0 npu:1
--transport host_stage --output NEW.json`. This probe is a diagnostic, not a
performance measurement. `scripts.check_placement --transport host_stage` then
checks the complete HB path. Its current phase and partial comparisons persist
before each forward/backward/optimizer stage, including the specific update and
reference or placed owner. Diagnostic stage record I/O is excluded from host phase
durations and included in complete wall time, along with state detach. Each complete update synchronizes all devices; `--phase-sync` adds
extra diagnostic fences before backward/optimizer and is recorded separately.
Individual phase durations measure device latency only with those extra fences. The checkpoint preserves the requested transport. An explicit transport remains
explicit after restore; `auto` resolves for the newly requested device mapping.
Older sharded checkpoints without a requested choice retain their saved resolved
transport. Both current choices are visible in `placement_report`.


## Whole-region ownership within a layer

Pass `region_partitions` to the same API to assign complete mathematical regions
within one layer to different devices. Omit it to retain the original contiguous
layer behavior. It is mutually exclusive with `layer_partitions`; every device
must own at least one complete region. The first implementation submits regions
serially, preserving layer dependencies. It establishes an ownership/transport
path, not simultaneous region execution or native cross-NPU event qualification.

```python
from benchmarks.capacity import placement_budget
from tide import TideModel, place_model, placement_adamw

# Meta tensors only; this does not allocate large weights or qualify capacity.
plan = placement_budget(config, mode="train", precision="bf16",
                        partition_kind="region", partitions=2)
model = place_model(TideModel(config), ["npu:0", "npu:1"],
                    region_partitions=plan["region_partitions"])
optimizer = placement_adamw(model, lr=1e-4, foreach=False)
```

The deterministic planner balances exact parameter bytes over complete regions
and inseparable shared groups. It does not optimize communication. An explicit
`region_partitions=[...]` can instead supply a reviewed layout. `independent` and
`region_shared` allow their complete regions to be assigned separately. A shared
base, source-embedding row or regional network may never span partitions. A
`layer_shared` proposal that crosses one of these groups is rejected before any
model module is mutated; no private replica or independent optimizer is created.
A legal layout that keeps all users of each base together can use the same
mechanism, but it does not establish shared-replica training. Regional placement
currently requires `region_batch_size=1` when that execution option is present;
combining cross-region batching with placement is a separate integration.

Input projections remain on partition zero, including under `independent` node
sharing. Their projected inputs are transported to first-layer consumers, and
autograd sums every contribution back to that single owner. Each destination
partition gathers the distinct predecessor nodes it needs, grouped by source
partition, then transports that compact bank once. The original edge ordering
and repeated edge consumers are reconstructed by static indexes. Values and
strengths move together; Timed pending ticks follow the same causal routes.
`placement_report` records the requested/resolved transport and static source-node
counts. These counts are not measured bus traffic or a communication speedup.

Node KV/SSD memories stay on their regional owner. Internally each layer bank is
a `RegionLayerOutput` containing the original regional order; Timed pending uses
compact differentiable last-tick clones, so a saved tail does not retain a full
output window. Consumed layer banks lose their Python reference after their last
static consumer; autograd still retains tensors needed by backward. Final output
nodes are collected in their original order on the last region's owner. Regions
with no output nodes are not connected to readout through a concatenate-and-slice
operation, preserving absent gradients for disconnected terminal owners.

`canonical_state(model, state, device="cpu")` gathers this private representation
into the original dense per-layer pending layout for comparisons. It is outside
timing and retains autograd copies; it is not a capacity-efficient checkpoint
format. `canonical_state_dict` continues to provide the dense parameter/gradient
view. Direct model continuation accepts its regional state, or a compatible dense
state which is split and moved to owners without detaching floating history.

Regional checkpoints use the existing `tide-instances-2` container with placement
kind `hb-region-shards-1`. Saving retains the actual parameter/optimizer layout,
regional memories and compact pending banks. Loading decodes on CPU, reconstructs
the same logical region layout, rebuilds static transport indexes, and remaps
saved partitions through `placement_devices`. The partition count and region
assignment remain fixed; arbitrary optimizer-state resharding is not implemented.
The same `placement_adamw` hook preserves original Parameter None-versus-zero,
weight decay, moment and step-counter meaning when only some shards have gradients.

The existing finite entry records regional mode in all three execution modes:

```sh
python -m scripts.check_placement --partition-kind region --devices cpu cpu \
  --precision fp32 --output NEW-region-qualification.json
# Actual NPU execution still requires an admitted, frozen whole-group job.
python -m scripts.check_placement --partition-kind region --devices npu:0 npu:1 \
  --precision bf16 --transport host_stage --output NEW-region-npu.json
```

`--region-partitions` supplies explicit integer owners; otherwise the exact-storage
planner chooses them. The JSON keeps original numerical thresholds and marks
`finite_deviation` separately from structural, nonfinite and execution failures.
CPU logical-partition tests cover exact initial values, FP64 VJPs, same and split
schedules, shared input gradients, Timed flush, three FP32/BF16 updates and actual
restored updates. Development tests retain finite FP32/BF16 trajectory deviations;
passing these mechanism tests does not turn those numerical failures into passes.
The existing direct/synchronized NPU event failures remain in the current evidence;
this implementation introduces no hardware qualification by itself.

## Exact region and shared-group planning

`benchmarks.capacity.region_ownership(config)` constructs only meta parameters and
returns each region's exclusive parameter count plus shared groups and their named
parameter components. Shared groups are counted once, and input projections are
reported separately. `placement_budget(..., partition_kind="region")` additionally
reports exact per-partition parameter rows, actual shard-Parameter counts, largest
Parameter shard, FP32 gradient/moment floors, separately listed CPU step scalars,
and cold/steady private-state bounds. Small tests compare those counts with actual
constructed shards; 9B/17B planner tests assert that all model parameters stay meta.

For 17B, region ownership can remove the indivisible widest-layer obstruction, but
not the total dense AdamW requirement: 16,707,220,176 FP32 parameters, gradients
and two moments require about 248.957 GiB before state and temporary allocations.
Two 60 GiB devices therefore cannot train this configuration under this policy.
Five or more devices require an actual per-region layout and remaining-margin
review. A single oversized region or inseparable shared group is still indivisible
here. The budget excludes activations/autograd, prepared banks, parameter casts,
static routing buffers, copies, allocator overhead and checkpoint host staging;
a positive planning margin is not proof that a hardware workload fits.
