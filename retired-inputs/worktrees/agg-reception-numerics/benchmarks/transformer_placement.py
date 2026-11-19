"""Contiguous Transformer layer owners for finite backbone comparisons.

Single-process tensor transport only. This is not a serving framework, replica
implementation or communication-backend qualification. A layer's real KV cache
stays with its weights; only its new hidden rows cross an ownership boundary.
"""
import torch
from torch import nn

from benchmarks.baselines import TransformerBaseline, TransformerState, parameter_count
from tide.placement import transfer_tensor


class LayerPlacedTransformer(nn.Module):
    def __init__(self, backbone, devices, *, partitions=None, transport="auto"):
        super().__init__()
        if not isinstance(backbone, TransformerBaseline):
            raise TypeError("placement needs a TransformerBaseline")
        if hasattr(backbone, "_benchmark_layer_owner"):
            raise ValueError("Transformer is already placed")
        devices = tuple(torch.device(device) for device in devices)
        if not devices or len({device.type for device in devices}) != 1 or devices[0].type not in {"cpu", "cuda", "npu"}:
            raise ValueError("placement needs explicit devices from one CPU/CUDA/NPU backend")
        devices = tuple(torch.device(d.type, getattr(torch, d.type).current_device())
                        if d.type != "cpu" and d.index is None else d for d in devices)
        count = len(backbone.blocks)
        partitions = tuple(i*len(devices)//count for i in range(count)) if partitions is None else tuple(partitions)
        if (len(partitions) != count or any(type(p) is not int for p in partitions)
                or partitions != tuple(sorted(partitions)) or set(partitions) != set(range(len(devices)))):
            raise ValueError("every device must own a contiguous nonempty Transformer layer interval")
        if transport not in {"auto", "direct", "host_stage", "synchronized"}:
            raise ValueError("unknown Transformer transport")
        requested = transport
        if transport == "auto":
            transport = "host_stage" if devices[0].type == "npu" and len(set(devices)) > 1 else "direct"
        self.backbone = backbone
        self.d, self.window, self.head_dim = backbone.d, backbone.window, backbone.head_dim
        self.devices, self.partitions, self.transport = devices, partitions, transport
        self._placement_devices = devices
        self._layouts = {}
        self._placement_spec = dict(kind="transformer-contiguous-layers-1", devices=[str(d) for d in devices],
                                    layer_partitions=list(partitions), requested_transport=requested, transport=transport,
                                    input_partition=partitions[0], output_partition=partitions[-1],
                                    communication="single-process new hidden rows only; no replicas or peer-DMA claim",
                                    private_state="bounded original-position KV per layer, retained on that layer owner")
        # This moves the actual modules, never copies a second dense backbone.
        for layer, block in enumerate(backbone.blocks):
            block.to(devices[partitions[layer]])
        backbone.final_norm.to(devices[partitions[-1]])
        backbone._layout_cache = None
        backbone._benchmark_layer_owner = True

    @property
    def parameter_count(self):
        return parameter_count(self)

    def configuration(self):
        return dict(self.backbone.configuration(), placement=self.placement_report())

    def placement_report(self):
        actual = {}
        for parameter in self.parameters():
            row = actual.setdefault(str(parameter.device), dict(parameters=0, parameter_bytes=0))
            row["parameters"] += parameter.numel()
            row["parameter_bytes"] += parameter.numel()*parameter.element_size()
        return dict(self._placement_spec, actual_devices=actual, total_unique_parameters=self.parameter_count)

    def forward(self, x, state=None, *, return_state=False):
        self.backbone._check_input(x)
        position, history = 0, 0
        if state is not None:
            position = state.position
            if type(position) is not int or position < 0:
                raise ValueError("Transformer state.position must be a nonnegative int")
            if len(state.keys) != len(self.backbone.blocks) or len(state.values) != len(self.backbone.blocks):
                raise ValueError("Transformer cache layer count differs from model")
            history = min(position, self.window-1)
            if any(t.ndim != 4 or t.shape[-2] != history for t in (*state.keys, *state.values)):
                raise ValueError("Transformer cache length differs from its original position/window")
        if position+x.shape[1] > torch.iinfo(torch.int64).max:
            raise OverflowError("Transformer position exceeds int64")
        length = x.shape[1]
        keys, values = [], []
        retained = min(position+length, self.window-1) if return_state else None
        for layer, block in enumerate(self.backbone.blocks):
            device = self.devices[self.partitions[layer]]
            x = transfer_tensor(x, device, self.transport)
            if not length:
                if state is None:
                    dtype = (torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type)
                             and x.dtype == torch.float32 else x.dtype)
                    keys.append(x.new_empty((x.shape[0], self.d//self.head_dim, 0, self.head_dim), dtype=dtype))
                    values.append(keys[-1].clone())
                continue
            # At most one layout per actual device, not one per layer or step.
            key = (length, position, history, x.device, x.dtype)
            cached_layout = self._layouts.get(x.device)
            if cached_layout is None or cached_layout[0] != key:
                layout = self.backbone._layout(x, position, history)
                self._layouts[x.device] = key, layout
            else:
                layout = cached_layout[1]
            cached = None if state is None else (state.keys[layer], state.values[layer])
            x, saved = block(x, layout, cached, retained)
            if saved is not None:
                keys.append(saved[0]); values.append(saved[1])
        result = self.backbone.final_norm(x)
        following = state if not length and state is not None else TransformerState(position+length, tuple(keys), tuple(values))
        return (result, following) if return_state else result
