"""Public API; load Torch and native/vendor adapters only when requested."""
from importlib import import_module

_EXPORTS = {'Edge': '.graph',
 'Graph': '.graph',
 'Node': '.graph',
 'Region': '.graph',
 'Atom': '.records',
 'Continuation': '.records',
 'External': '.records',
 'State': '.records',
 'PortLayout': '.ports',
 'History': '.history',
 'SourceDomain': '.source_domain',
 'StateClock': '.clocks',
 'GraphConfig': '.config',
 'ExecutionOptions': '.execution_options',
 'ExecutionPlacement': '.placement',
 'place_payloads': '.ownership',
 'ResidentLimits': '.resident_options',
 'ResidentTrainingLimits': '.resident_training_options',
 'ResidentPlacement': '.resident_training_options',
 'GraphRuntime': '.library',
 'FP32MasterOptimizer': '.precision',
 '__version__': '.version'}

__all__ = ['Edge',
 'Graph',
 'Node',
 'Region',
 'Atom',
 'Continuation',
 'External',
 'State',
 'PortLayout',
 'History',
 'SourceDomain',
 'StateClock',
 'GraphConfig',
 'ExecutionOptions',
 'ExecutionPlacement',
 'place_payloads',
 'ResidentLimits',
 'ResidentTrainingLimits',
 'ResidentPlacement',
 'GraphRuntime',
 'FP32MasterOptimizer',
 '__version__']


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
