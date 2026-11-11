"""Finite built-in column capabilities, with typed private state per memory.

One schedule calls prepare/Read/Full. New memory algorithms may add a private
state and scan here without changing message or regional column interfaces.
"""
from dataclasses import dataclass
from .kv_columns_state import KVStateColumns, prepare_kv_columns
from .kv_columns import read_kv_columns, full_kv_columns, supported
from .recurrent_column_state import RecurrentStateColumns, prepare_recurrent_columns
from .recurrent_columns import read_recurrent_columns, full_recurrent_columns


@dataclass(frozen=True)
class ColumnModule:
    state_type: type
    prepare: object
    read: object
    full: object
    reference_field: str


def module_for(node):
    from .recurrent import BasicMemory
    from .gated_delta import GatedDeltaMemory
    from .ssd import SSDMemory
    if supported(node.kernel):
        return ColumnModule(KVStateColumns, prepare_kv_columns, read_kv_columns, full_kv_columns, "keys")
    if type(node.kernel) in {BasicMemory, GatedDeltaMemory, SSDMemory} and node.bias.dtype.is_floating_point:
        return ColumnModule(RecurrentStateColumns, prepare_recurrent_columns, read_recurrent_columns, full_recurrent_columns, "H")
    raise ValueError("local memory has no supported column implementation")
