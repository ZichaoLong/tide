"""Public API; load Torch and native/vendor adapters only when requested."""
from importlib import import_module

_EXPORTS = {'Allocation': '.allocation',
 'allocate_quad': '.allocation',
 'allocate': '.entropy_allocation',
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
 'WeightedResidentTraining': '.resident_training'}

__all__ = ['Allocation',
 'allocate',
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
 'WeightedResidentTraining']


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
