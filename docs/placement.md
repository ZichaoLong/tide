# HB layer placement

`place_model` is a single-process layer split for the current strictly layered
HB instances. It partitions unique parameter rows, retains sharing within each
layer and leaves private memory with its owner. Actual predecessor tensor banks
move once per destination layer; their ordinary Torch copies retain autograd.
This path has no replicas, all-reduce or device-resident scheduler. A successful
copy does not establish a native peer-DMA implementation.

The transport is explicit: `transport="direct"` uses ordinary cross-device Torch
copies; `"host_stage"` uses synchronized accelerator→CPU→accelerator copies;
`"synchronized"` is a first-order custom copy with source/destination fences in
forward and backward. The latter is not a higher-order differentiation API.
These are bounded backend candidates, with hardware qualification attached to
its recorded source. Host staging deliberately pays CPU transport and is never
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
```

The inference mode measures construction/transfer separately, and then forward,
readout, necessary Timed flush and all-device synchronization. It records per-device
allocator peaks, process RSS, state storage, actual work and a separate operator
witness. No small check or one-step large execution qualifies all shapes, full
training capacity or sustained service. Optimizer bytes still follow the FP32
master/gradient/moment policy; a BF16 matrix policy does not halve that capacity.

To isolate transfer/backward failures, run each candidate in a separate admitted
process with `python -m scripts.probe_placement_copy --devices npu:0 npu:1
--transport host_stage --output NEW.json`. This probe is a diagnostic, not a
performance measurement. `scripts.check_placement --transport host_stage` then
checks the complete HB path. Its current phase and partial comparisons persist
before each forward/backward/optimizer stage, including the specific update and
reference or placed owner. Diagnostic stage record I/O is excluded from host phase
durations. Each complete update synchronizes all devices; `--phase-sync` adds
extra diagnostic fences before backward/optimizer and is recorded separately.
Individual phase durations measure device latency only with those extra fences. The checkpoint preserves the selected
transport so a restore does not silently change execution policy.
