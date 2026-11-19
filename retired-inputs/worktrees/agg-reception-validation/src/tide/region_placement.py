"""Whole-region placement using the ordinary regional tensor evaluator.

A layer bank keeps each region on its owner. Only the predecessor nodes needed
by a destination partition are packed and transported, once per source owner.
Optional per-device tasks join at each layer; transport remains caller-owned.
This is not a general graph scheduler, replica implementation or peer-DMA claim.
"""
from dataclasses import dataclass

import torch

from .model import LayerOutput, ModelState, Result
from .placement import move_tensors


@dataclass
class RegionLayerOutput:
    regions: tuple[LayerOutput, ...]


def gather_region_bank(bank, device, transport="direct"):
    chunks = [move_tensors(region, device, transport) for region in bank.regions]
    return LayerOutput(torch.cat([region.values for region in chunks], 0),
                       torch.cat([region.strengths for region in chunks], 0))



def _gather_outputs(model, bank, device, transport):
    # Terminal H-only regions may retain private state without contributing to
    # readout. Do not concatenate them and slice afterward: that manufactures
    # zero gradients for otherwise disconnected Parameters.
    wanted = set(model.graph.output_nodes)
    values, strengths = [], []
    for region, chunk in zip(model.layer_regions[-1], bank.regions):
        selected = [local for local, node in enumerate(model.graph.regions[region]) if node in wanted]
        if not selected:
            continue
        if len(selected) != len(model.graph.regions[region]):
            ids = torch.tensor(selected, device=chunk.values.device, dtype=torch.int64)
            chunk = LayerOutput(chunk.values.index_select(0, ids), chunk.strengths.index_select(0, ids))
        chunk = move_tensors(chunk, device, transport)
        values.append(chunk.values); strengths.append(chunk.strengths)
    return LayerOutput(torch.cat(values, 0), torch.cat(strengths, 0))

def _split_dense_bank(model, layer, bank):
    if isinstance(bank, RegionLayerOutput):
        if len(bank.regions) != len(model.layer_regions[layer]):
            raise ValueError("pending bank does not match regional layout")
        return bank
    offset, regions = 0, []
    for region in model.layer_regions[layer]:
        end = offset+len(model.graph.regions[region])
        regions.append(LayerOutput(bank.values[offset:end], bank.strengths[offset:end]))
        offset = end
    return RegionLayerOutput(tuple(regions))


def place_region_state(model, state):
    if state is None:
        return None
    if len(state.memories) != len(model.graph.regions):
        raise ValueError("private memories do not match regional layout")
    transport = model._placement_spec["transport"]
    memories = tuple(move_tensors(memory, model._placement_devices[model._region_partitions[region]], transport)
                     for region, memory in enumerate(state.memories))
    pending = []
    if state.pending and len(state.pending) != len(model.layer_regions):
        raise ValueError("pending layer count does not match regional layout")
    for layer, bank in enumerate(state.pending):
        bank = _split_dense_bank(model, layer, bank)
        pending.append(RegionLayerOutput(tuple(
            move_tensors(chunk, model._placement_devices[model._region_partitions[region]], transport)
            for region, chunk in zip(model.layer_regions[layer], bank.regions))))
    return ModelState(state.position, memories, tuple(pending))


def build_region_transfer_plan(model):
    """Static owner/index metadata only; rebuilt after checkpoint loading."""
    region_local = {region: local for regions in model.layer_regions for local, region in enumerate(regions)}
    node_local = {node: local for region in model.graph.regions for local, node in enumerate(region)}
    transfers, routes, summary = [], {}, []
    for layer, regions in enumerate(model.layer_regions):
        destinations = {}
        for region in regions:
            destination = model._region_partitions[region]
            origins = destinations.setdefault(destination, {})
            for node in model.graph.regions[region]:
                for edge_id in model.graph.incoming[node]:
                    source = model.graph.edges[edge_id].src
                    origin = model.graph.nodes[source].layer
                    origins.setdefault(origin, set()).add(source)
        plan = {}
        for destination, origins in destinations.items():
            device = model._placement_devices[destination]
            packed = {}
            for origin, needed in origins.items():
                nodes = tuple(sorted(needed))
                by_owner = {}
                for node in nodes:
                    source_region = model.graph.nodes[node].region
                    owner = model._region_partitions[source_region]
                    by_owner.setdefault(owner, {}).setdefault(source_region, []).append(node)
                chunks, packed_nodes = [], []
                for owner, source_regions in sorted(by_owner.items()):
                    selections, owner_nodes = [], []
                    for source_region, selected in sorted(source_regions.items()):
                        index = torch.tensor([node_local[node] for node in selected], dtype=torch.int64,
                                             device=model._placement_devices[owner])
                        selections.append((region_local[source_region], index))
                        owner_nodes.extend(selected)
                    chunks.append((owner, tuple(selections)))
                    packed_nodes.extend(owner_nodes)
                    summary.append(dict(layer=layer, origin=origin, source_partition=owner,
                                        destination_partition=destination, unique_source_nodes=len(owner_nodes),
                                        remote=model._placement_devices[owner] != device))
                positions = {node: index for index, node in enumerate(packed_nodes)}
                reorder = torch.tensor([positions[node] for node in nodes], device=device, dtype=torch.int64)
                packed[origin] = dict(chunks=tuple(chunks), reorder=reorder, nodes=nodes)
            plan[destination] = packed
        transfers.append(plan)
        for region in regions:
            destination = model._region_partitions[region]
            device = model._placement_devices[destination]
            classes, banks = model._routes[region]
            compact = {}
            for origin, (indices, mask) in banks.items():
                nodes = plan[destination][origin]["nodes"]
                positions = {node: index for index, node in enumerate(nodes)}
                # _routes is construction-owned CPU metadata, never a readback
                # from live device routing decisions.
                remap = torch.zeros_like(indices)
                for row, slots in enumerate(indices.tolist()):
                    for slot, old in enumerate(slots):
                        if bool(mask[row, slot]):
                            remap[row, slot] = positions[model.layer_nodes[origin][old]]
                compact[origin] = remap.to(device), mask.to(device)
            routes[region] = classes.to(device), compact
    last_use = list(range(len(model.layer_regions)))
    for row in summary:
        last_use[row["origin"]] = max(last_use[row["origin"]], row["layer"])
    model._region_last_use = tuple(last_use)
    model._region_transfers, model._region_routes = tuple(transfers), routes
    model._region_transfer_summary = summary
    # Wrappers are shared Python objects, but every tensor shard belongs to one
    # task. Populate their shared dispatch dictionary before workers start.
    # Each shard's uniform-group cache is touched only by its device task.
    for region in range(len(model.graph.regions)):
        _prepare_owner_dispatch(model, (region,))


def _prepare_owner_dispatch(model, regions):
    members = tuple(node for region in regions for node in model.graph.regions[region])
    for name, wrapper in model._shard_registry.items():
        if name.startswith("receive."):
            possible = members
        elif name.startswith("describe."):
            possible = tuple(model.description_ids[node] for node in members
                             if node in model.description_ids)
        else:
            kind = name.split(".")[1]
            possible = tuple(model.memory_ids[kind][node] for node in members
                             if node in model.memory_ids[kind])
        if possible:
            wrapper._prepare_dispatch(possible)


def region_batches(model, regions, state):
    """Compatible batches stay within a parameter partition, even on one device.

    Private-state shapes can change after continuation. Recompute the grouping
    from host metadata, and prepare shared wrapper dispatch before workers run.
    No device-valued routing decision is copied to the host here.
    """
    if model.config.region_batch_size == 1:
        return [(region,) for region in regions]
    partitions = {}
    for region in regions:
        partitions.setdefault(model._region_partitions[region], []).append(region)
    batches = [batch for members in partitions.values()
               for batch in model._region_batches(members, state)]
    for batch in batches:
        if len(batch) > 1:
            _prepare_owner_dispatch(model, batch)
    return batches


def _transport_bank(bank, plan, device, transport):
    values, strengths = [], []
    for _, selections in plan["chunks"]:
        parts = [LayerOutput(bank.regions[region].values.index_select(0, ids),
                             bank.regions[region].strengths.index_select(0, ids))
                 for region, ids in selections]
        packed = LayerOutput(torch.cat([part.values for part in parts], 0),
                             torch.cat([part.strengths for part in parts], 0))
        packed = move_tensors(packed, device, transport)
        values.append(packed.values); strengths.append(packed.strengths)
    return LayerOutput(torch.cat(values, 0).index_select(0, plan["reorder"]),
                       torch.cat(strengths, 0).index_select(0, plan["reorder"]))


def _layer_inputs(model, layer, layers, state, embedded):
    result = {}
    transport = model._placement_spec["transport"]
    for destination, plans in model._region_transfers[layer].items():
        device = model._placement_devices[destination]
        banks = {origin: _transport_bank(layers[origin], plan, device, transport) for origin, plan in plans.items()}
        pending = ({origin: _transport_bank(state.pending[origin], plan, device, transport)
                    for origin, plan in plans.items()} if state is not None and state.pending else {})
        # Projection remains a single Parameter owner, including independent
        # node sharing. Cross-device consumers contribute through these copies.
        inputs = move_tensors(embedded, device, transport) if layer == 0 else embedded
        result[destination] = banks, pending, inputs
    return result


def run_region_model(model, x, state=None, *, record_work=False, inject=True):
    transport = model._placement_spec["transport"]
    x = move_tensors(x, model._placement_devices[model._placement_spec["input_partition"]], transport)
    state = place_region_state(model, state)
    batch, positions, _ = x.shape
    start = 0 if state is None else state.position
    if type(start) is not int or start < 0:
        raise OverflowError("state.position must be a nonnegative Python int")
    maximum = ((start+positions-1)*model.graph.period+len(model.layer_nodes) if model.settle else
               (start+positions)*model.graph.period)
    if maximum > torch.iinfo(torch.int64).max:
        raise OverflowError("continuation logical time exceeds int64")
    length = positions if model.settle else positions*model.graph.period
    embedded = {role: projection(x) for role, projection in model.inputs.items()}
    if not model.settle:
        loc = torch.arange(positions, device=x.device)*model.graph.period
        embedded = {role: value.new_zeros((batch, length, model.config.d)).index_copy(1, loc, value)
                    for role, value in embedded.items()}
    layers, memories, work, pending_tails = [], [], [], []
    from .region_execution import RegionExecutor
    with RegionExecutor(model) as executor:
        for layer, regions in enumerate(model.layer_regions):
            inputs = _layer_inputs(model, layer, layers, state, embedded)
            batches = region_batches(model, regions, state)
            by_first = {members[0]: members for members in batches}
            times = {}
            for partition in inputs:
                device = model._placement_devices[partition]
                base = torch.arange(length, device=device, dtype=torch.int64)
                times[partition] = ((base+start)*model.graph.period+layer+1 if model.settle else
                                    base+start*model.graph.period+1)

            def evaluate(first):
                members = by_first[first]
                partition = model._region_partitions[first]
                device = model._placement_devices[partition]
                banks, pending, projected = inputs[partition]
                incoming = [model._incoming(region, banks, pending, projected, batch, length,
                                             device, inject, routes=model._region_routes[region])
                            for region in members]
                time = times[partition]
                if len(members) == 1:
                    contents, masses, classes = incoming[0]
                    old = None if state is None else state.memories[first]
                    function = model._evaluate_region
                    args = (first, contents, masses, classes, time, old, record_work)
                else:
                    from .region_batch import evaluate_regions
                    contents, masses, classes = (torch.cat([row[i] for row in incoming], 0) for i in range(3))
                    old = None if state is None else [state.memories[region] for region in members]
                    function = evaluate_regions
                    args = (model, members, contents, masses, classes, time, old, record_work)
                if model.config.checkpoint_regions and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    full, control, saved, measured = checkpoint(function, *args, use_reentrant=False)
                else:
                    full, control, saved, measured = function(*args)
                if len(members) == 1:
                    full, control, saved = (full,), (control,), (saved,)
                return dict(zip(members, zip(full, control, saved))), measured

            results = {}
            for grouped, measured in executor.map(tuple(by_first), evaluate):
                results.update(grouped)
                if measured is not None:
                    work.append(measured)
            chunks = [LayerOutput(results[region][0], results[region][1]) for region in regions]
            memories.extend(results[region][2] for region in regions)
            time = times[model._region_partitions[regions[-1]]]
            bank = RegionLayerOutput(tuple(chunks))
            layers.append(bank)
            if not model.settle:
                pending_tails.append(RegionLayerOutput(tuple(LayerOutput(
                    chunk.values[:, :, -1:].clone(), chunk.strengths[:, :, -1:].clone()) for chunk in chunks)))
            # Python-owned banks expire after their final consumer; autograd
            # keeps saved tensors. The compact tails remain differentiable.
            for origin, last_use in enumerate(model._region_last_use[:layer+1]):
                if last_use == layer and origin != len(model.layer_regions)-1:
                    layers[origin] = None
    pending = tuple(pending_tails)
    output_device = model._placement_devices[model._placement_spec["output_partition"]]
    final = _gather_outputs(model, bank, output_device, transport)
    measured = None
    if record_work:
        measured = dict(zip(("candidates", "full", "positive_messages", "candidate_slots"),
                            torch.stack([move_tensors(item, output_device, transport) for item in work]).sum(0).tolist()))
    return Result(final.values, final.strengths, time.to(output_device),
                  ModelState(start+positions, tuple(memories), pending), measured)
