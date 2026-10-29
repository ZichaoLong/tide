"""Allocation-free core-5 configuration and workload planning.

Runtime construction imports Torch only inside materialize().
"""
from .inventory import inventory
from .workload import make_plan, validate_plan

__all__ = ["inventory", "make_plan", "validate_plan"]

from .model import resolve_model, validate_model, workload
from .execution import ExecutionSpec, MeasurementSpec

__all__ += ["resolve_model", "validate_model", "workload", "ExecutionSpec", "MeasurementSpec"]
