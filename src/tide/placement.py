"""Finite layer placement for the current strictly layered HB instances.

This is a single-process split of parameter ownership and private state. It is
not a general graph scheduler or distributed replica implementation. Shared bases
never span these layer partitions. Cross-partition tensor copies retain autograd.
"""
from collections import OrderedDict
from dataclasses import fields, is_dataclass

import torch
from torch import nn

from .parameters import NodeLinear, NodeOwners, NodeVector, owner_ids
from .reception import SourceEmbedding


def move_tensors(value, device):
    """Move private state without detaching floating history or changing int64."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value):
        return type(value)(**{f.name: move_tensors(getattr(value, f.name), device) for f in fields(value)})
    if isinstance(value, dict):
        return {k: move_tensors(v, device) for k, v in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(move_tensors(v, device) for v in value))
    if isinstance(value, (tuple, list)):
        return type(value)(move_tensors(v, device) for v in value)
    return value


def _parameter_rows(parameter, indices, device):
    index = torch.tensor(indices, device=parameter.device, dtype=torch.int64)
    return nn.Parameter(parameter.detach().index_select(0, index).to(device),
                        requires_grad=parameter.requires_grad)


class _ShardedOwners(nn.Module):
    def __init__(self, source, partitions, devices):
        super().__init__()
        self.partitions = tuple(partitions)
        self.devices = tuple(torch.device(device) for device in devices)
        self.shards = nn.ModuleList()
        self._maps, self._shapes = {}, {}
        self._original_buffers = {name: value.detach().cpu().clone()
                                  for name, value in source.named_buffers(recurse=False)}
        self._dispatch = {}
        self._live_partitions = []

    def _new_shard(self, cls, partition, owners):
        shard = cls.__new__(cls)
        nn.Module.__init__(shard)
        lookup = torch.full((len(self.partitions),), -1, dtype=torch.int64)
        lookup[list(owners)] = torch.arange(len(owners))
        shard.register_buffer("owner_lookup", lookup.to(self.devices[partition]))
        self._live_partitions.append(partition)
        self.shards.append(shard)
        return shard

    def _slice(self, source, shard, leaf, indices, partition):
        parameter = getattr(source, leaf)
        if parameter is None:
            setattr(shard, leaf, None)
            return
        setattr(shard, leaf, _parameter_rows(parameter, indices, self.devices[partition]))
        self._maps.setdefault(leaf, []).append(tuple(indices))
        self._shapes[leaf] = tuple(parameter.shape)

    def _local(self, owners):
        if not isinstance(owners, NodeOwners):
            raise TypeError("placed owner modules require static NodeOwners")
        possible = owners.possible_ids
        if possible not in self._dispatch:
            partitions = {self.partitions[index] for index in possible}
            if len(partitions) != 1:
                raise ValueError("one region call cannot span layer partitions")
            partition = next(iter(partitions))
            shard_index = self._live_partitions.index(partition)
            local = tuple(i for i, owner in enumerate(
                index for index, part in enumerate(self.partitions) if part == partition) if owner in possible)
            self._dispatch[possible] = shard_index, local
        index, possible_local = self._dispatch[possible]
        shard = self.shards[index]
        ids = owner_ids(owners)
        if ids.device != shard.owner_lookup.device:
            raise ValueError("owner IDs must be on their layer partition device")
        return shard, NodeOwners(shard.owner_lookup[ids], possible_local)

    def canonical(self, *, gradients=False, device="cpu"):
        result = OrderedDict()
        for leaf, shape in self._shapes.items():
            tensors = [(getattr(shard, leaf).grad if gradients else getattr(shard, leaf))
                       for shard in self.shards]
            if gradients and all(value is None for value in tensors):
                result[leaf] = None
                continue
            template = next(value for value in tensors if value is not None)
            value = torch.zeros(shape, dtype=template.dtype, device=device)
            for indices, tensor in zip(self._maps[leaf], tensors):
                if tensor is not None:
                    value.index_copy_(0, torch.tensor(indices, dtype=torch.int64, device=device),
                                      tensor.detach().to(device))
            result[leaf] = value
        if not gradients:
            result.update((name, value.to(device)) for name, value in self._original_buffers.items())
        return result

    def gradient_groups(self):
        return [[getattr(shard, leaf) for shard in self.shards] for leaf in self._shapes]


class _ShardedLinear(_ShardedOwners):
    def __init__(self, source, partitions, devices):
        super().__init__(source, partitions, devices)
        self.nodes, self.in_features, self.out_features = source.nodes, source.in_features, source.out_features
        self.shared = source.shared
        groups = source._host_groups
        _check_group_ownership(groups, partitions)
        for partition in range(len(devices)):
            owners = tuple(i for i, part in enumerate(partitions) if part == partition)
            if not owners:
                continue
            original_groups = tuple(sorted({groups[i] for i in owners}))
            mapping = {group: i for i, group in enumerate(original_groups)}
            shard = self._new_shard(NodeLinear, partition, owners)
            shard.nodes, shard.in_features, shard.out_features = len(owners), source.in_features, source.out_features
            shard.shared = source.shared
            shard._set_groups(torch.tensor([mapping[groups[i]] for i in owners], dtype=torch.int64))
            shard.groups = shard.groups.to(devices[partition])
            self._slice(source, shard, "weight", original_groups, partition)
            self._slice(source, shard, "bias", original_groups, partition)
            if source.shared:
                self._slice(source, shard, "left", owners, partition)
                self._slice(source, shard, "right", owners, partition)

    def forward(self, x, owners=None):
        shard, owners = self._local(owners)
        return shard(x, owners)


class _ShardedVector(_ShardedOwners):
    def __init__(self, source, partitions, devices):
        super().__init__(source, partitions, devices)
        groups = source._host_groups
        _check_group_ownership(groups, partitions)
        for partition in range(len(devices)):
            owners = tuple(i for i, part in enumerate(partitions) if part == partition)
            if not owners:
                continue
            original_groups = tuple(sorted({groups[i] for i in owners}))
            mapping = {group: i for i, group in enumerate(original_groups)}
            shard = self._new_shard(NodeVector, partition, owners)
            shard._set_groups(torch.tensor([mapping[groups[i]] for i in owners], dtype=torch.int64))
            shard.groups = shard.groups.to(devices[partition])
            self._slice(source, shard, "value", original_groups, partition)

    def forward(self, owners=None):
        shard, owners = self._local(owners)
        return shard(owners)


class _ShardedEmbedding(_ShardedOwners):
    def __init__(self, source, partitions, devices):
        super().__init__(source, partitions, devices)
        self.labels = source.labels
        lookup = source.lookup.detach().cpu()
        ownership = {}
        for owner, rows in enumerate(lookup.tolist()):
            for row in rows:
                if row >= 0 and ownership.setdefault(row, partitions[owner]) != partitions[owner]:
                    raise ValueError("a source embedding parameter cannot span layer partitions")
        for partition in range(len(devices)):
            owners = tuple(i for i, part in enumerate(partitions) if part == partition)
            if not owners:
                continue
            original_rows = tuple(sorted({row for owner in owners for row in lookup[owner].tolist() if row >= 0}))
            mapping = {row: index for index, row in enumerate(original_rows)}
            local_lookup = torch.tensor([[mapping[row] if row >= 0 else -1 for row in lookup[owner].tolist()]
                                         for owner in owners], dtype=torch.int64)
            shard = self._new_shard(SourceEmbedding, partition, owners)
            shard.labels = source.labels
            shard.register_buffer("lookup", local_lookup.to(devices[partition]))
            self._slice(source, shard, "weight", original_rows, partition)

    def forward(self, owners, classes):
        shard, owners = self._local(owners)
        return shard(owners, classes)


def _check_group_ownership(groups, partitions):
    ownership = {}
    for group, partition in zip(groups, partitions):
        if ownership.setdefault(group, partition) != partition:
            raise ValueError("shared parameter base cannot span layer partitions")


def _default_partitions(model, count):
    if count > len(model.layer_nodes):
        raise ValueError("more devices than HB layers")
    # Contiguous cuts approximately balance actual node counts. Return the exact
    # resulting parameter capacity below; this does not promise equal memory.
    weights = [len(layer) for layer in model.layer_nodes]
    partitions, completed, partition = [], 0, 0
    for layer, weight in enumerate(weights):
        remaining_layers = len(weights)-layer
        if partition < count-1 and partitions and (
                completed >= sum(weights)*(partition+1)/count or remaining_layers == count-partition-1):
            partition += 1
        partitions.append(partition)
        completed += weight
    return tuple(partitions)


def place_model(model, devices, *, layer_partitions=None):
    """Split HB parameters in place, before optimizer construction.

    ``devices`` can repeat CPU for logical-partition formula checks. Real devices
    must share one backend type. Returned outputs live on the last layer's device;
    private state stays on each owner's device. Use ``placement_adamw`` so zero
    gradient rows retain the original unsplit Parameter's update semantics.
    """
    if hasattr(model, "_placement_spec"):
        raise ValueError("model is already placed; restore into a new owner to change placement")
    if model.graph.config.kind not in {"lattice", "exchange", "multiscale"}:
        raise ValueError("layer placement currently supports the strictly layered HB instances")
    devices = tuple(torch.device(device) for device in devices)
    if not devices or any(device.type not in {"cpu", "npu", "cuda"} for device in devices):
        raise ValueError("placement needs explicit CPU, NPU or CUDA devices")
    if len({device.type for device in devices}) != 1:
        raise ValueError("mixed CPU/accelerator model placement is not implemented")
    partitions = _default_partitions(model, len(devices)) if layer_partitions is None else tuple(layer_partitions)
    if (len(partitions) != len(model.layer_nodes) or any(type(index) is not int for index in partitions)
            or tuple(sorted(partitions)) != partitions or set(partitions) != set(range(len(devices)))):
        raise ValueError("layer_partitions must assign every device a contiguous nonempty layer interval")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("place a fresh model before computing gradients/constructing the optimizer")
    registry = {}
    def replace(module, owner_partitions, prefix):
        for name, child in list(module.named_children()):
            path = prefix+"."+name
            wrapper = (_ShardedLinear if isinstance(child, NodeLinear) else
                       _ShardedVector if isinstance(child, NodeVector) else
                       _ShardedEmbedding if isinstance(child, SourceEmbedding) else None)
            if wrapper is not None:
                replacement = wrapper(child, owner_partitions, devices)
                setattr(module, name, replacement)
                registry[path] = replacement
            else:
                replace(child, owner_partitions, path)
    node_partitions = tuple(partitions[node.layer] for node in model.graph.nodes)
    replace(model.receive, node_partitions, "receive")
    described = [node for node, _ in sorted(model.description_ids.items(), key=lambda row: row[1])]
    replace(model.describe, tuple(node_partitions[node] for node in described), "describe")
    for kind, memory in model.memories.items():
        owners = [node for node, _ in sorted(model.memory_ids[kind].items(), key=lambda row: row[1])]
        replace(memory, tuple(node_partitions[node] for node in owners), "memories."+kind)
    regional_partitions = {}
    for region, group in enumerate(model.region_group):
        if group < 0:
            continue
        partition = partitions[model.graph.nodes[model.graph.regions[region][0]].layer]
        if regional_partitions.setdefault(group, partition) != partition:
            raise ValueError("regional sharing cannot span layer partitions")
    for group, partition in regional_partitions.items():
        model.regional[group].to(devices[partition])
    model.inputs.to(devices[partitions[0]])
    model._placement_devices, model._layer_partitions = devices, partitions
    model._shard_registry = registry
    model._placement_spec = dict(kind="hb-layer-shards-1", devices=[str(device) for device in devices],
                                 layer_partitions=list(partitions),
                                 optimizer="placement_adamw; align disconnected shards per original Parameter before step",
                                 communication="single-process autograd tensor.to copies of actual predecessor banks; no HCCL replicas")
    return model


def placement_adamw(model, **kwargs):
    """AdamW with original-Parameter zero/None/weight-decay and slot-step meaning."""
    optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
    if not hasattr(model, "_shard_registry"):
        return optimizer
    groups = [group for wrapper in model._shard_registry.values() for group in wrapper.gradient_groups()]
    def prepare(optimizer, args, kwargs):
        if kwargs.get("closure") is not None or len(args) > 1 and args[1] is not None:
            raise ValueError("placed AdamW requires explicit backward before step; closures are unsupported")
        for group in groups:
            if any(parameter.grad is not None for parameter in group):
                for parameter in group:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
    optimizer.register_step_pre_hook(prepare)
    optimizer._tide_placement = model._placement_spec
    return optimizer


def canonical_state_dict(model, *, gradients=False, device="cpu"):
    """Export an ordinary dense state/gradient view for portable comparison.

    This gathers full weights and is intentionally outside timed execution. A
    placed checkpoint instead retains its shards and never gathers onto one NPU.
    """
    registry = getattr(model, "_shard_registry", {})
    source = dict(model.named_parameters()) if gradients else model.state_dict()
    result = OrderedDict()
    for name, value in source.items():
        if any(name.startswith(prefix+".") for prefix in registry):
            continue
        value = value.grad if gradients else value
        result[name] = None if value is None else value.detach().to(device).clone()
    for prefix, wrapper in registry.items():
        for name, value in wrapper.canonical(gradients=gradients, device=device).items():
            result[prefix+"."+name] = value
    return result


def place_state(model, state):
    if state is None or not hasattr(model, "_placement_spec"):
        return state
    from .model import ModelState
    memories = tuple(move_tensors(memory, model._placement_devices[
        model._layer_partitions[model.graph.nodes[model.graph.regions[region][0]].layer]])
        for region, memory in enumerate(state.memories))
    pending = tuple(move_tensors(layer, model._placement_devices[model._layer_partitions[index]])
                    for index, layer in enumerate(state.pending))
    return ModelState(state.position, memories, pending)


def placement_report(model):
    if not hasattr(model, "_placement_spec"):
        raise ValueError("model is not placed")
    devices = {}
    for parameter in model.parameters():
        row = devices.setdefault(str(parameter.device), dict(parameters=0, parameter_bytes=0))
        row["parameters"] += parameter.numel()
        row["parameter_bytes"] += parameter.numel()*parameter.element_size()
    return dict(**model._placement_spec, actual_devices=devices,
                total_unique_parameters=sum(parameter.numel() for parameter in model.parameters()))
