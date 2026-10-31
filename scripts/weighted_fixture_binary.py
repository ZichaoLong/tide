"""Compatibility alias to the installed public implementation."""
import sys
from tidegraph.experiment import binary_material as _implementation
sys.modules[__name__] = _implementation
