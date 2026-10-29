"""Compatibility import; public implementation lives in tidegraph.planning.precision."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"python"))
from tidegraph.planning.precision import *  # noqa: F401,F403
