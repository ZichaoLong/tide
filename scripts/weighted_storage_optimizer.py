"""Compatibility alias to the installed public implementation."""
import sys
from tidegraph.experiment import storage_optimizer as _implementation
sys.modules[__name__] = _implementation
