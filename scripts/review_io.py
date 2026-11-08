"""Restrict human-review JSON writes to review files, never job inputs/outputs.

An earlier ad-hoc handoff loop shadowed its map path with a future run's
state.json path. Keep the validation before mkdir/atomic_json so a status
update cannot reserve a workload directory or overwrite a live run record.
"""
from pathlib import Path

from tide_lab.records import atomic_json

ROOT = Path(__file__).resolve().parents[1]
GRAPH_SCHEMA = 'tide-research-map-prototype-v1'


def write_review_json(path, value, *, root=ROOT):
    root = Path(root).resolve()
    requested = Path(path)
    target = (requested if requested.is_absolute() else root/requested).resolve()
    if not isinstance(value, dict):
        raise ValueError('review records must be objects')
    if value.get('schema') == GRAPH_SCHEMA:
        if target != root/'research/map.json':
            raise ValueError('research graph can only be written to research/map.json')
    elif target.parent != root/'docs/evidence' or target.suffix != '.json':
        raise ValueError('review evidence must stay in docs/evidence; workload paths are read-only')
    atomic_json(target, value)

