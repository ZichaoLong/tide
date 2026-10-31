"""Compatibility alias to the installed public implementation."""
import sys
from tidegraph.experiment import training_layout as _implementation
sys.modules[__name__] = _implementation
