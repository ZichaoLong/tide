"""Finite built-in column capabilities, with typed private state per memory.

One schedule calls prepare/Read/Full. New memory algorithms may add a private
state and scan here without changing message or regional column interfaces.
"""
from dataclasses import dataclass
from .kv_columns_state import KVStateColumns, prepare_kv_columns, prepare_kv_columns_many
from .kv_columns import read_kv_columns, full_kv_columns, supported
from .kv_column_groups import read_kv_columns_many, full_kv_columns_many
from .recurrent_column_state import RecurrentStateColumns, prepare_recurrent_columns
from .recurrent_columns import read_recurrent_columns, full_recurrent_columns


@dataclass(frozen=True)
class ColumnModule:
    state_type: type
    prepare: object
    read: object
    full: object
    reference_field: str
    prepare_many: object = None
    read_many: object = None
    full_many: object = None


def module_for(node):
    from .recurrent import BasicMemory
    from .gated_delta import GatedDeltaMemory
    from .ssd import SSDMemory
    if supported(node.kernel):
        return ColumnModule(KVStateColumns, prepare_kv_columns, read_kv_columns, full_kv_columns, "keys",
                            prepare_kv_columns_many, read_kv_columns_many, full_kv_columns_many)
    if type(node.kernel) in {BasicMemory, GatedDeltaMemory, SSDMemory} and node.bias.dtype.is_floating_point:
        return ColumnModule(RecurrentStateColumns, prepare_recurrent_columns, read_recurrent_columns, full_recurrent_columns, "H")
    raise ValueError("local memory has no supported column implementation")


def prepare_many(requests, *, max_workspace_bytes):
    """Map node IDs to typed windows; modules own batching and private layouts."""
    result, groups = {}, {}
    for node, (module, *args) in requests.items():
        if module.prepare_many is None:
            result[node] = module.prepare(*args)
        else:
            groups.setdefault(module.prepare_many, []).append((node, args))
    for function, rows in groups.items():
        windows = function([args for _, args in rows], max_workspace_bytes=max_workspace_bytes)
        result.update((node, window) for (node, _), window in zip(rows, windows))
    return {node: result[node] for node in requests}


def read_many(requests, *, max_workspace_bytes):
    """(module,node,window,mode,initial); modules own private query grouping."""
    result, groups = {}, {}
    for key, (module, node, window, mode, initial) in requests.items():
        if module.read_many is None:
            result[key] = module.read(node, window, read_mode=mode, initial=initial)
        else:
            groups.setdefault(module.read_many, []).append((key, (node, window, mode, initial)))
    for function, rows in groups.items():
        values = function([args for _, args in rows], max_workspace_bytes=max_workspace_bytes)
        result.update((key, value) for (key, _), value in zip(rows, values))
    return {key: result[key] for key in requests}


def full_many(requests, *, max_workspace_bytes):
    """(module,node,window,controls); recurrent and custom capability fallback."""
    result, groups = {}, {}
    for key, (module, *args) in requests.items():
        if module.full_many is None:
            result[key] = module.full(*args)
        else:
            groups.setdefault(module.full_many, []).append((key, args))
    for function, rows in groups.items():
        values = function([args for _, args in rows], max_workspace_bytes=max_workspace_bytes)
        result.update((key, value) for (key, _), value in zip(rows, values))
    return {key: result[key] for key in requests}
