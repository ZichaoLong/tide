"""Strict Python index types before scheduling or pybind integer coercion.

Range, graph ownership and temporal consistency remain executor validations.
Owned cursors validate imported state once and only new inputs per advance.
"""
from .history import History, int64


def integers(label, *values):
    if any(not int64(value) for value in values):
        raise ValueError(f"{label} requires Python int64 values (excluding bool)")


def _pair(label, values):
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(f"invalid {label} pair")
    integers(label, *values)


def window_inputs(external, stop, sealed_until):
    integers("window coordinates", stop, sealed_until)
    # Materialize an iterable once; all metadata is checked before any execution.
    rows = list(external)
    for x in rows:
        integers("external coordinates", x.batch, x.port, x.position, x.time)
    return rows


def continuation(q):
    integers("continuation cut/batch", q.cut, q.batch_size)
    for owner, state in q.states.items():
        _pair("state owner", owner)
        integers("state clock", state.last_time, state.observations)
    for owner, history in q.history.items():
        _pair("region history owner", owner)
        if not isinstance(history, History):
            raise ValueError("invalid region history record")
        integers("region history clock", history.last_time)
        if not isinstance(history.scalars, dict) or not isinstance(history.node_maps, dict):
            raise ValueError("invalid region history fields")
        integers("region history scalars", *history.scalars.values())
        for counts in history.node_maps.values():
            if not isinstance(counts, dict):
                raise ValueError("invalid region history node map")
            for node, count in counts.items():
                integers("region history node map", node, count)
    for owner, last in q.ledger.items():
        _pair("input ledger owner", owner)
        _pair("input ledger position/time", last)
    for a in q.pending:
        integers("pending coordinates", a.batch, a.node, a.time, a.kind, a.source, a.position)
