"""Temporary Read/control column views; raw state and parameter owners stay put."""
from dataclasses import replace
import torch
from .config import KVConfig
from .parameter_tree import rebind
from .placement_tensors import mapper, move


def read_parameter(name, config):
    if name.startswith("read_program.") or name == "kernel.summary":
        return True
    if name.startswith("kernel.read_"):
        return True
    return (not isinstance(config, KVConfig) and not config.basic and config.memory_read == "state"
            and (name in {"kernel.projection", "kernel.delta_bias"} or name.startswith("kernel.conv.")))


def _read_inputs(node, window, initial, mode):
    """Collect borrowed roles before any copy; shared roots keep their identity."""
    c = node.kernel.config
    parameters = [p for n, p in node.named_parameters(remove_duplicate=False) if read_parameter(n, c)]
    parameters += [p for n, p in node.named_buffers(remove_duplicate=False) if read_parameter(n, c)]
    values = [window.mass, window.content, *parameters]
    if isinstance(c, KVConfig):
        names = ("summaries", "weights") if c.memory_read == "sum" else ("keys", "values", "weights")
        if mode == "0":
            # Read0 validates the complete bound initial state (usually empty).
            names = ("keys", "values", "summaries", "weights", "inert")
            source = initial
        else:
            source = window.bank
        fields = {name: getattr(source, name) for name in names}
        values += list(fields.values())
    else:
        source = initial if mode == "0" else window
        fields = {n: getattr(initial, n) for n in ("H", "Z", "W")} if mode == "0" else window.bank
        values += list(fields.values())
        if source.cache_values is not None:
            values.append(source.cache_values)
    return values, parameters, source, fields


def _read_view(node, window, initial, mode, device, dtype, *, mapped=None, inputs=None):
    if (window.content.device, window.content.dtype) == (device, dtype):
        return node, window, initial
    values, parameters, source, fields = inputs if inputs is not None else _read_inputs(node, window, initial, mode)
    mapped = mapper(values, device, dtype) if mapped is None else mapped
    ids = {id(x) for x in parameters}
    view = rebind(node, lambda t: mapped(t) if id(t) in ids else t)
    query = replace(window, mass=mapped(window.mass), content=mapped(window.content))
    if isinstance(node.kernel.config, KVConfig):
        copied = replace(source, **{name: mapped(value) for name, value in fields.items()})
        if mode == "0":
            initial = copied
        else:
            query = replace(query, bank=copied)
    else:
        cache = None if source.cache_values is None else mapped(source.cache_values)
        if mode == "0":
            initial = replace(initial, **{n: mapped(t) for n, t in fields.items()}, cache_values=cache)
        else:
            query = replace(query, bank={n: mapped(t) for n, t in fields.items()}, cache_values=cache)
    return view, query, initial


def read(module, node, window, initial, mode, device, dtype):
    node, window, initial = _read_view(node, window, initial, mode, device, dtype)
    return module.read(node, window, read_mode=mode, initial=initial)


def read_many(requests, *, max_workspace_bytes):
    """Copy each real root once per destination, then preserve node/block roles.

    Request-local copies would turn shared parameter owners into unrelated
    temporary tensors, forcing matrix stacking and duplicate transfers. Input
    order remains stable and different destination/dtype groups stay separate.
    """
    from .column_modules import read_many as evaluate
    prepared, groups = {}, {}
    for key, (module, node, window, initial, mode, device, dtype) in requests.items():
        if (window.content.device, window.content.dtype) == (device, dtype):
            prepared[key] = module, node, window, mode, initial
        else:
            inputs = _read_inputs(node, window, initial, mode)
            groups.setdefault((device, dtype), []).append((key, inputs))
    for (device, dtype), members in groups.items():
        mapped = mapper([t for _, inputs in members for t in inputs[0]], device, dtype)
        for key, inputs in members:
            module, node, window, initial, mode, _, _ = requests[key]
            node, window, initial = _read_view(node, window, initial, mode, device, dtype,
                                               mapped=mapped, inputs=inputs)
            prepared[key] = module, node, window, mode, initial
    return evaluate({key: prepared[key] for key in requests}, max_workspace_bytes=max_workspace_bytes)


def region(program, fields, nodes, control, selection):
    from .regional_columns import region_columns
    ref = fields["score"]
    tensors = [*program.parameters(), *program.buffers()]
    if control != ref.device or any(p.device != control or p.dtype != ref.dtype for p in tensors):
        mapped = mapper([*tensors, *fields.values()], control, ref.dtype)
        program = rebind(program, mapped)
        fields = {n: mapped(t) for n, t in fields.items()}
        nodes = nodes.to(control)
    return region_columns(program, fields, nodes, selection=selection)


def controls(value, like):
    result = move(value, like.device, like.dtype)
    if value.dtype != result.dtype and not torch.equal((value.detach() > 0).cpu(), (result.detach() > 0).cpu()):
        raise ValueError("column control cast changes positive support")
    return result
