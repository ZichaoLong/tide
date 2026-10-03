"""Eager payload ownership, separate from the online scheduling policy."""
import torch


def region_reference(graph, model, region):
    program = model.regions[region]
    values = [*program.parameters(), *program.buffers()]
    if values:
        return values[0]
    members = graph.region_layouts[region].members
    return model.nodes[members[0] if members else 0].bias


def payload_devices(model):
    return tuple(dict.fromkeys(v.device for v in model.state_dict().values()))


def place_payloads(graph, model, node_devices):
    """Place a newly constructed model in place, before creating its optimizer.

    Each node's weights/state stay together. A shared node parameter constrains
    those nodes to one owner; conflicting explicit maps fail before mutation.
    Region and port aliases retain a single canonical leaf. Port uses copy that
    leaf differentiably when their consumer is remote. No graph is executed.
    """
    devices = tuple(torch.device(d) for d in node_devices)
    if len(devices) != len(graph.nodes) or len(model.nodes) != len(graph.nodes):
        raise ValueError("payload owner map must cover every node")
    if any(d.type not in {"cpu", "cuda", "npu"} or
           (d.type != "cpu" and d.index is None) for d in devices):
        raise ValueError("payload owners require CPU or indexed CUDA/NPU devices")
    families = {d.type for d in devices if d.type != "cpu"}
    if len(families) > 1:
        raise ValueError("payload owners cannot mix accelerator backends")
    devices = tuple(torch.device("cpu") if d.type == "cpu" else d for d in devices)
    destinations = {}

    def assign(values, device, required=False):
        for value in values:
            old = destinations.setdefault(id(value), device)
            if required and old != device:
                raise ValueError("shared node parameter requires co-located payload owners")

    for node, device in zip(model.nodes, devices):
        assign((*node.parameters(), *node.buffers()), device, True)
    for region, program in enumerate(model.regions):
        values = (*program.parameters(), *program.buffers())
        members = graph.region_layouts[region].members
        device = next((destinations[id(v)] for v in values if id(v) in destinations),
                      devices[members[0] if members else 0])
        assign(values, device, True)
    for field, owners in (("input_scale", graph.inputs), ("output_scale", graph.outputs),
                          ("agg_scale", [e.target for e in graph.edges]),
                          ("edge_scale", [e.source for e in graph.edges])):
        for value, owner in zip(getattr(model, field), owners):
            assign((value,), devices[owner])
    # Stage all copies first; allocation errors leave the caller's model intact.
    copied = {}
    for module in model.modules():
        for field in ("_parameters", "_buffers"):
            for value in getattr(module, field).values():
                if value is None or id(value) in copied:
                    continue
                device = destinations[id(value)]
                moved = value
                if value.device != device:
                    moved = value.detach().to(device)
                    if isinstance(value, torch.nn.Parameter):
                        moved = torch.nn.Parameter(moved, requires_grad=value.requires_grad)
                    else:
                        moved.requires_grad_(value.requires_grad)
                copied[id(value)] = moved
    for module in model.modules():
        for field in ("_parameters", "_buffers"):
            table = getattr(module, field)
            for name, value in table.items():
                if value is not None:
                    table[name] = copied[id(value)]
    return model
