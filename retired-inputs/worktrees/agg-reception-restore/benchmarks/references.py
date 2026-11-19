"""Frozen initial-workload records shared by independent comparison children."""
import hashlib
import json
from pathlib import Path


def file_identity(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))


def load_reference(path, config, arguments, *, same_shape):
    record = json.loads(Path(path).read_text())
    if record.get("state") not in {"audit_complete", "completed_finite"}:
        raise ValueError("reference must be a completed initial-workload audit")
    if record.get("config") != config.to_dict():
        raise ValueError("reference configuration differs from the measured model")
    required = ("precision", "seed", "activity", "no_flush")
    if same_shape:
        required += ("batch", "length")
    for key in required:
        if record["arguments"][key] != getattr(arguments, key):
            raise ValueError("reference workload differs: "+key)
    if not all(mode in record["joint_matches_by_mode"] and mode in record["initial_audits"]
               for mode in arguments.modes):
        raise ValueError("reference does not cover the requested modes")
    return record
