"""Compatibility alias to the installed public implementation."""
import sys
from tidegraph.experiment import returns as _implementation
sys.modules[__name__] = _implementation
