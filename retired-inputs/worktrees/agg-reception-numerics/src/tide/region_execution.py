"""Bounded same-layer device tasks; caller owns transport and publication.

Only grad, inference, autocast, device and current-stream state is propagated.
Intrusive caller-thread instrumentation must use the explicit serial witness.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar

import torch


_serial_audits = ContextVar("tide_serial_region_audits", default=frozenset())


@contextmanager
def serial_region_audit(model):
    """Run an untimed, explicitly serial instrumentation witness for this model.

    Dispatch modes, saved-tensor hooks and profilers are caller-thread facilities;
    they are not replicated into region workers. The returned identity belongs in
    the audit record, separately from the requested timed execution policy.
    """
    token = _serial_audits.set(_serial_audits.get() | {id(model)})
    try:
        yield dict(requested_region_concurrency=getattr(model, "_placement_spec", {}).get(
                       "region_concurrency", "serial"),
                   effective_region_concurrency="serial", workers=0,
                   scope="caller-thread instrumentation witness; excluded from concurrent timing")
    finally:
        _serial_audits.reset(token)


def _check_caller_modes():
    # These read-only stack queries are version-sensitive; fail explicitly if
    # unavailable instead of silently omitting an unknown caller's mode.
    try:
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
        from torch.overrides import _get_current_function_mode_stack
    except ImportError as error:
        raise RuntimeError("per_device mode inspection is unavailable in this Torch; use serial") from error
    if _get_current_dispatch_mode_stack() or _get_current_function_mode_stack():
        raise RuntimeError("per_device does not propagate caller dispatch/function modes; "
                           "use serial_region_audit(model) for an explicitly serial witness")


def device_groups(model, regions):
    groups = {}
    for region in regions:
        device = model._placement_devices[model._region_partitions[region]]
        groups.setdefault(device, []).append(region)
    return groups


def execution_report(model):
    widths = [len(device_groups(model, regions)) for regions in model.layer_regions]
    policy = model._placement_spec.get("region_concurrency", "serial")
    return dict(policy=policy, maximum_layer_devices=max(widths),
                maximum_workers=max(widths) if policy == "per_device" and max(widths) > 1 else 0,
                device_identity="CPU indices are logical simulation groups; repeated devices share one task",
                region_batch_size=model.config.region_batch_size,
                ownership="caller projects and transports; compatible regions batch within each parameter partition; one task per device; join each layer",
                streams="caller current stream per accelerator, captured before worker creation; no extra streams",
                backward="ordinary autograd engine, without a separate region thread scheduler",
                instrumentation="record_work aggregates returned per-region counters; caller dispatch/function modes require explicit serial_region_audit; hooks/profilers are not propagated")


class RegionExecutor:
    """One finite pool per forward/flush, bounded by simultaneous device groups."""
    def __init__(self, model):
        self.model, self.pool = model, None
        report = execution_report(model)
        self.workers = 0 if id(model) in _serial_audits.get() else report["maximum_workers"]
        self.contexts = {}

    def __enter__(self):
        if not self.workers:
            return self
        _check_caller_modes()
        self.grad = torch.is_grad_enabled()
        self.inference = torch.is_inference_mode_enabled()
        self.cache = torch.is_autocast_cache_enabled()
        for device in dict.fromkeys(self.model._placement_devices):
            stream = None
            if device.type != "cpu":
                backend = getattr(torch, device.type)
                # Public initialization and stream lookup happen on the caller.
                # Workers enter that same stream, preserving transport order.
                backend.init()
                stream = backend.current_stream(device)
            self.contexts[device] = (torch.is_autocast_enabled(device.type),
                                     torch.get_autocast_dtype(device.type), stream)
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="tide-region")
        return self

    def _run(self, device, regions, evaluate):
        enabled, dtype, stream = self.contexts[device]
        with ExitStack() as stack:
            if device.type != "cpu":
                backend = getattr(torch, device.type)
                stack.enter_context(backend.device(device))
                stack.enter_context(backend.stream(stream))
            stack.enter_context(torch.inference_mode(self.inference))
            stack.enter_context(torch.set_grad_enabled(self.grad))
            stack.enter_context(torch.autocast(device.type, enabled=enabled, dtype=dtype,
                                               cache_enabled=self.cache))
            return [(region, evaluate(region)) for region in regions]

    def map(self, regions, evaluate):
        groups = device_groups(self.model, regions)
        if self.pool is None or len(groups) == 1:
            return [evaluate(region) for region in regions]
        futures = [self.pool.submit(self._run, device, members, evaluate)
                   for device, members in groups.items()]
        results = dict(item for future in futures for item in future.result())
        return [results[region] for region in regions]

    def __exit__(self, *exc):
        if self.pool is not None:
            # Also join sibling tasks when a worker fails, before inputs leave
            # scope or the caller is allowed to retry/mutate model parameters.
            self.pool.shutdown(wait=True, cancel_futures=True)
