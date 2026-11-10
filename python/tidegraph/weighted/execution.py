"""Public eager dispatch for weighted bindings; independent scheduler ownership.

Standard scalar and packed ATen programs expose their joint kernels through the
existing LocalProgram capabilities. New optimized kernels attach there; this
adapter never chooses a different mathematical configuration.
"""
from ..placement import place_model


def eager_runner(binding, implementation, placement, native_library=None, *, streaming=False,
                 workers=1, packing="auto", query_execution="auto", prefill_schedule="auto"):
    if prefill_schedule not in {"auto", "greedy", "window"}:
        raise ValueError("unknown prefill_schedule")
    if streaming and prefill_schedule != "auto":
        raise ValueError("explicit prefill_schedule requires prefill")
    algorithm = "window" if prefill_schedule == "auto" else prefill_schedule
    if implementation not in {"python", "native"} or packing not in {"auto", "packed", "scalar"}:
        raise ValueError("unknown eager implementation/packing")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    packed = packing != "scalar"
    if query_execution == "auto" and (not packed or streaming or workers != 1):
        query_execution = "legacy"
    if query_execution not in {"legacy", "auto"} and (not packed or streaming or workers != 1):
        raise ValueError("dense-v1 requires packed prefill with one worker")
    from .query_execution import execution_view
    model = execution_view(binding.model, query_execution)
    options = dict(packed=packed, aggregate_autograd="batched" if packed else "replay",
                   full_autograd="batched" if packed else "replay")
    if implementation == "native":
        from ..native import Native
        from ..native_loader import load_native
        load_native(native_library, backend=binding.model.nodes[0].bias.device.type)
        return Native(binding.graph, binding.model, workers=workers,
                      algorithm="streaming" if streaming else algorithm, placement=placement,
                      query_execution=query_execution, **options).run
    if workers != 1 or streaming and packing == "packed":
        raise ValueError("Python eager has one host worker and scalar streaming dispatch")
    from ..greedy import run as greedy
    from ..streaming import run as stream
    view = place_model(binding.graph, model, placement)
    schedule = stream if streaming else greedy
    if streaming:
        options = {}
    else:
        options["component_window"] = algorithm == "window"
    modes = [getattr(n.read_program, "query_execution", "legacy") for n in model.nodes]
    def run(*a, **kw):
        result = schedule(binding.graph, view, *a, **options, **kw)
        for mode in ("legacy", "dense-v1", "packed-v2"):
            result.stats["query_"+mode+"_nodes"] = modes.count(mode)
        return result
    return run
