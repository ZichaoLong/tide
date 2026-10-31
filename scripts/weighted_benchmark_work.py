"""Untimed/profile-only packing counters for the existing benchmark clients."""
from contextlib import contextmanager
from tidegraph.weighted.packing_observation import observe

SCHEMA = "tide-weighted-packing-observation-v1"
SCOPE = ("eager local forward calls; training includes private forward replay; "
         "host metadata only, no additional tensor reads or device synchronization")


@contextmanager
def observe_work(implementation):
    """Warmup, reference and checkpoint checks must stay outside this scope."""
    result = dict(schema=SCHEMA, scope=SCOPE, throughput_sample=False, completed=False)
    if implementation == "python":
        with observe() as counters:
            try:
                yield result
                result["completed"] = True
            finally:
                result["metrics"] = counters.metrics()
    elif implementation == "native":
        import _tide_native as core
        if core.work_metrics().get("packing/schema_version") != 1:
            raise RuntimeError("native packing observation requires a matching rebuilt core")
        core.reset_work(True)
        try:
            yield result
            result["completed"] = True
        finally:
            result["metrics"] = {k: v for k, v in core.work_metrics().items() if k.startswith("packing/")}
            core.reset_work(False)
    elif implementation == "resident":
        result.update(scope="resident physical rows/stages are reported by actual_stats; eager bucket counters are inapplicable",
                      applicable=False)
        yield result
        result["completed"] = True
    else:
        raise ValueError("unknown packing observation implementation")
