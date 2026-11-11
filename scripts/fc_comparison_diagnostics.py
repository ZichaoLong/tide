"""Scoped FC cost diagnostics; these are not public execution alternatives.

Direct-boundary trials keep local numerical functions but omit their independent
intermediate-root VJP wrappers. Only an explicitly checked task projection may
be compared. Never install these patches in a consumer or qualify general roots.
"""
from contextlib import contextmanager, ExitStack
from functools import wraps
from unittest.mock import patch
from contextvars import ContextVar


BOUNDARIES = ("public", "query-direct", "region-direct", "both-direct")
_calls = ContextVar("fc_diagnostic_boundary_calls", default=None)


def boundary_calls():
    return dict(_calls.get() or {})


@contextmanager
def boundary_scope(mode):
    if mode not in BOUNDARIES:
        raise ValueError("unknown diagnostic boundary")
    with ExitStack() as stack:
        calls = {}
        token = _calls.set(calls)
        stack.callback(_calls.reset, token)
        if mode in {"query-direct", "both-direct"}:
            from tidegraph.weighted import query_vjp, read_batch, full_batch

            def direct(program, weights, requests, *, kind="read", slots=0, mode="hard", zeta=1.):
                calls[kind+"_calls"] = calls.get(kind+"_calls", 0)+1
                calls[kind+"_rows"] = calls.get(kind+"_rows", 0)+len(requests)
                if kind == "read":
                    return read_batch.evaluate(program, weights, requests)
                if kind == "full":
                    return full_batch.evaluate(program, weights, requests, slots, mode, zeta)
                raise ValueError("unknown query kind")

            stack.enter_context(patch.object(query_vjp, "evaluate", direct))
        if mode in {"region-direct", "both-direct"}:
            from tidegraph.weighted import region_vjp, region_batch
            def direct_region(program, requests):
                calls["region_calls"] = calls.get("region_calls", 0)+1
                calls["region_rows"] = calls.get("region_rows", 0)+len(requests)
                return region_batch.evaluate(program, requests)
            stack.enter_context(patch.object(region_vjp, "evaluate", direct_region))
        yield


@contextmanager
def formula_ranges():
    """Label the actual formula implementation only during a profiling pass."""
    import torch
    import fc_tensor_baseline as formula

    def labelled(name, function):
        @wraps(function)
        def call(*args, **kwargs):
            with torch.profiler.record_function("formula::"+name):
                return function(*args, **kwargs)
        return call

    with ExitStack() as stack:
        for name in ("aggregate", "prepare", "full", "quadratic", "head", "kv_terminal"):
            stack.enter_context(patch.object(formula, name, labelled(name, getattr(formula, name))))
        stack.enter_context(patch.object(formula.TransformerLayers, "forward",
                                        labelled("transformer", formula.TransformerLayers.forward)))
        yield


def stage_operators(profiler):
    """CPU dispatcher observations by nearest named range, not wall-time shares."""
    from collections import defaultdict
    groups = defaultdict(lambda: dict(count=0, self_cpu_ms=0., total_cpu_ms=0., shapes={}))
    for event in profiler.events():
        if not event.name.startswith("aten::"):
            continue
        parent, stage = event.cpu_parent, "unattributed"
        while parent is not None:
            if parent.name.startswith(("formula::", "tide::stage::", "task::")):
                stage = parent.name
                break
            parent = parent.cpu_parent
        row = groups[stage, event.name]
        row["count"] += 1
        row["self_cpu_ms"] += event.self_cpu_time_total/1000
        row["total_cpu_ms"] += event.cpu_time_total/1000
        shape = str(event.input_shapes)
        row["shapes"][shape] = row["shapes"].get(shape, 0)+1
    return dict(scope="Instrumented CPU dispatcher self times, grouped by closest calling-thread range. "
                      "Workers may be unattributed; nested totals overlap, and these are not device kernel times.",
                operators=[dict(stage=stage, operator=name, **row)
                           for (stage, name), row in sorted(groups.items())])
