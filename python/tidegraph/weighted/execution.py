"""Public eager dispatch for weighted bindings; independent scheduler ownership.

Standard scalar and packed ATen programs expose their joint kernels through the
existing LocalProgram capabilities. New optimized kernels attach there; this
adapter never chooses a different mathematical configuration.
"""
from ..placement import place_model


def eager_runner(binding, implementation, placement, native_library=None, *, streaming=False,
                 workers=1, packing="auto"):
    if implementation not in {"python", "native"} or packing not in {"auto", "packed", "scalar"}:
        raise ValueError("unknown eager implementation/packing")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    packed = packing != "scalar"
    options = dict(packed=packed, aggregate_autograd="batched" if packed else "replay",
                   full_autograd="batched" if packed else "replay")
    if implementation == "native":
        from ..native import Native
        from ..native_loader import load_native
        load_native(native_library, backend=binding.model.nodes[0].bias.device.type)
        return Native(binding.graph, binding.model, workers=workers,
                      algorithm="streaming" if streaming else "greedy", placement=placement, **options).run
    if workers != 1 or streaming and packing == "packed":
        raise ValueError("Python eager has one host worker and scalar streaming dispatch")
    from ..greedy import run as greedy
    from ..streaming import run as stream
    view = place_model(binding.graph, binding.model, placement)
    schedule = stream if streaming else greedy
    if streaming:
        options = {}
    return lambda *a, **kw: schedule(binding.graph, view, *a, **options, **kw)
