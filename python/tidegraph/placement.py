"""Explicit execution placement; independent Python and native implementations."""
from dataclasses import asdict, dataclass
import copy
import torch


@dataclass(frozen=True)
class ExecutionPlacement:
    preset: str = "native"
    read: str = "auto"
    control: str = "auto"
    selection: str = "auto"
    events: str = "auto"
    scoring_dtype: str = "profile"

    def __post_init__(self):
        if self.preset not in {"native", "cpu", "mixed-a", "mixed-b", "mixed-c", "resident"}:
            raise ValueError("unknown execution placement preset")
        if self.scoring_dtype not in {"profile", "payload", "float32", "float64"}:
            raise ValueError("unknown scoring precision")
        if any(not isinstance(getattr(self, key), str) for key in ("read", "control", "selection", "events")):
            raise ValueError("placement switches must be device strings")

    def resolve(self, payload):
        payload = torch.device(payload)
        if payload.type not in {"cpu", "cuda", "npu"}:
            raise ValueError("placement requires CPU, CUDA or NPU payload tensors")
        defaults = dict(read="payload", control="payload", selection="cpu", events="cpu")
        if self.preset == "cpu":
            if payload.type != "cpu":
                raise ValueError("CPU preset requires CPU payload tensors")
            defaults.update(read="cpu", control="cpu")
        elif self.preset not in {"native", "cpu"}:
            if payload.type == "cpu":
                raise ValueError("accelerator preset requires accelerator payload tensors")
            if self.preset == "mixed-a":
                defaults.update(read="cpu", control="cpu")
            if self.preset in {"mixed-c", "resident"}:
                defaults["selection"] = "payload"
            if self.preset == "resident":
                defaults["events"] = "payload"
        resolved = dict(payload=payload)
        for key, default in defaults.items():
            requested = getattr(self, key)
            value = default if requested == "auto" else requested
            device = payload if value == "payload" else torch.device(value)
            if device.type == "cpu":
                device = torch.device("cpu")
            elif device.type == payload.type and device.index in {None, payload.index}:
                device = payload
            else:
                raise ValueError("placement must use CPU or the model payload device")
            resolved[key] = device
        return dict(resolved, scoring_dtype=self.scoring_dtype)

    def to_dict(self):
        return asdict(self)


def request(value):
    if value is None:
        return ExecutionPlacement()
    return value if isinstance(value, ExecutionPlacement) else ExecutionPlacement(**value)


def read_dtype(spec, payload, placement):
    declared = {"norm-fp32-v1": torch.float32, "norm-fp64-v1": torch.float64}.get(spec.readout, payload)
    precision = placement["scoring_dtype"]
    dtype = declared if precision == "profile" else payload if precision == "payload" else getattr(torch, precision)
    if spec.readout != "linear-v1" and dtype != declared:
        raise ValueError("scoring precision conflicts with the named norm Read contract")
    if dtype == torch.float64 and any(placement[key].type == "npu" for key in ("read", "control", "selection")):
        raise ValueError("NPU Read/control/selection cannot consume FP64 descriptors; select CPU or explicit FP32 Read")
    return dtype


def validate(graph, dtype, placement):
    if placement["events"].type != "cpu":
        raise ValueError("host model placement cannot provide device-resident event progression")
    for spec in graph.nodes:
        read_dtype(spec, dtype, placement)


def _shallow_module(module):
    result = copy.copy(module)
    result._modules = dict(module._modules)
    result._parameters = dict(module._parameters)
    result._buffers = dict(module._buffers)
    return result


def place_model(graph, model, placement):
    """A Python execution view sharing the original parameter leaves and names.

    Use Native(..., placement=...) for the independent native implementation.
    The caller owns optimizer/checkpoint state. No input, state or route is run
    in advance; only built-in Read/Region programs can be adapted safely.
    """
    from .placement_read import PlacedRead
    from .placement_region import PlacedRegion
    from .ownership import region_reference
    config = request(placement)
    if len(model.nodes) != len(graph.nodes) or len(model.regions) != len(graph.regions):
        raise ValueError("placement model/graph size mismatch")
    result = _shallow_module(model)
    nodes = []
    for spec, weights in zip(graph.nodes, model.nodes):
        p = config.resolve(weights.bias.device)
        if p["events"].type != "cpu":
            raise ValueError("host model placement cannot provide device-resident event progression")
        node = _shallow_module(weights)
        node.read_program = PlacedRead(spec, weights.read_program, read_dtype(spec, weights.bias.dtype, p), p["read"])
        nodes.append(node)
    result.nodes = torch.nn.ModuleList(nodes)
    result.regions = torch.nn.ModuleList(PlacedRegion(spec, program, config.resolve(
        region_reference(graph, model, r).device)) for r, (spec, program) in enumerate(zip(graph.regions, model.regions)))
    return result
