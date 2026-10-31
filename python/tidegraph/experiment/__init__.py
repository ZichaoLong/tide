"""Installed, configuration-driven experiment tools.

Planning and report inspection do not import Torch or initialize a backend.
Execution, independent checks and task-owned training remain separate APIs.
"""
from .config import resolve

__all__ = ["resolve"]
