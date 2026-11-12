"""Public API; load Torch and native/vendor adapters only when requested."""
from importlib import import_module

_EXPORTS = {'Allocation': '.allocation',
 'eager_runner': '.execution',
 'WindowResult': '.window_execution',
 'SequenceLayout': '.column_layout',
 'MessageColumns': '.message_columns',
 'KVStateColumns': '.kv_columns_state',
 'RecurrentStateColumns': '.recurrent_column_state',
 'SettleColumnPlan': '.settle_columns',
 'allocate_quad': '.allocation',
 'allocate': '.entropy_allocation',
 'allocate_batch': '.allocation_rows',
 'allocate_columns': '.allocation_columns',
 'AllocationColumns': '.allocation_columns',
 'region_columns': '.regional_columns',
 'RegionColumns': '.regional_columns',
 'observe_support': '.support_policy',
 'attention_shares': '.sending',
 'attention_send': '.sending',
 'proportional': '.sending',
 'total_zero': '.sending',
 'hard_selection': '.sending',
 'Binding': '.binding',
 'bind': '.binding',
 'KVConfig': '.config',
 'RegionalConfig': '.config',
 'RecurrentConfig': '.recurrent_config',
 'ConvCache': '.convolution',
 'FixedTimeConv': '.convolution',
 'Payload': '.tensors',
 'Received': '.tensors',
 'SourceBinding': '.source_config',
 'fully_connected': '.topology',
 'hb_lattice': '.topology',
 'bind_layered': '.instances',
 'layered_inputs': '.instances',
 'readout': '.instances',
 'WeightedResident': '.resident',
 'WeightedResidentTraining': '.resident_training',
 'WeightedBlockResidentTraining': '.resident_blocks',
 'block_resident_capability': '.resident_blocks'}

__all__ = ['Allocation',
 'eager_runner', 'WindowResult', 'SequenceLayout', 'MessageColumns',
 'KVStateColumns', 'RecurrentStateColumns', 'SettleColumnPlan',
 'allocate',
 'allocate_batch',
 'allocate_columns',
 'AllocationColumns',
 'region_columns',
 'RegionColumns',
 'observe_support',
 'allocate_quad',
 'Binding',
 'bind',
 'KVConfig',
 'RegionalConfig',
 'SourceBinding',
 'attention_shares',
 'attention_send',
 'proportional',
 'total_zero',
 'hard_selection',
 'RecurrentConfig',
 'ConvCache',
 'FixedTimeConv',
 'Payload',
 'Received',
 'fully_connected',
 'hb_lattice',
 'bind_layered',
 'layered_inputs',
 'readout',
 'WeightedResident',
 'WeightedResidentTraining',
 'WeightedBlockResidentTraining',
 'block_resident_capability']


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
