"""Explicit consumer capacities, independent of values or an execution trace."""
import hashlib
import json

FIELDS = frozenset(("queue", "stages", "cache", "tensor_bytes", "operator_bytes", "timeout_ms",
                    "windows", "retained_bytes", "backward_bytes", "optimizer_bytes", "total_bytes"))


def load(path):
    raw = path.read_bytes()
    values = json.loads(raw)
    if not isinstance(values, dict) or not values or set(values)-FIELDS:
        raise ValueError("resident limits require known capacity fields")
    if any(type(x) is not int or not 1 <= x < 2**63 for x in values.values()):
        raise ValueError("resident limits require positive int64 values")
    return values, hashlib.sha256(raw).hexdigest()
