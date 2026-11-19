"""Intrusive stage and operator diagnoses, separate from normal timing passes.

These observers serialize the selected device. They include host dispatch and
completion synchronization, perturb scheduling, and do not measure native kernel
time. Run stage and operator observers in separate replays of the same workload.
Hook installation changes global aliases: use one exclusive worker, without
WorkAudit or another hook session in the same interpreter.
"""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import importlib
import time

from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from benchmarks.run import synchronize
from benchmarks.stages import (Paths, StageTimes, _descriptor, _descriptor_json,
                               _tensors, stage_hooks)


class DetailedStageTimes(StageTimes):
    """Nested synchronized wall times with the entry wait outside the scope.

In a nested observation, an entry wait for previously submitted work remains in
the enclosing scope's exclusive time. It is never charged to the entering scope.
Completion waits remain inside elapsed time. Hook and bookkeeping overhead still
perturb the execution and can remain in the parent.
"""
    def __init__(self, device, *, clock=time.perf_counter, sync=synchronize):
        super().__init__(device, clock=clock, sync=sync)
        self.boundary_failures = []

    @contextmanager
    def scope(self, name):
        try:
            self.sync(self.device)
        except BaseException as error:
            self.boundary_failures.append(dict(path="/".join([*self.stack, name]),
                                               boundary="entry", error=repr(error)))
            raise
        start = self.clock()
        self.stack.append(name)
        self.children.append(0.)
        body_failed = False
        try:
            yield
        except BaseException:
            body_failed = True
            raise
        finally:
            path, elapsed = self.path, None
            try:
                self.sync(self.device)
                elapsed = self.clock() - start
            except BaseException as error:
                self.boundary_failures.append(dict(path=path, boundary="completion", error=repr(error)))
                # Preserve an exception from the observed code. The additional
                # synchronization failure remains explicit in the report.
                if not body_failed:
                    raise
            finally:
                children = self.children.pop()
                self.stack.pop()
                if elapsed is not None:
                    row = self.rows[path]
                    row["calls"] += 1
                    row["inclusive_seconds"] += elapsed
                    row["exclusive_seconds"] += elapsed - children
                    if self.children:
                        self.children[-1] += elapsed

    def report(self):
        return dict(
            rows=[dict(path=path, **row) for path, row in sorted(self.rows.items())],
            boundary_failures=list(self.boundary_failures),
            boundary="entry synchronize completes before the timer starts; completion synchronize is inside elapsed time",
            interpretation="separate intrusive stage pass; inclusive rows overlap and must not be summed; exclusive rows partition only this observation, not normal wall time or kernel time; a nested entry wait for prior unwrapped work stays in its parent; Python, synchronization and hook overhead perturb execution",
        )


@contextmanager
def detailed_stage_hooks(model, observer):
    """Extend the established Tide/Transformer stages without double wrapping.

FFN, node projections, vectors and memory helpers retain their calling hierarchy.
Canonical module names distinguish parameter owners. Batched regional linears
and inline memory contractions are visible in OperatorTimes even when they do
not call a module's forward. No tensors or computation are replaced.
"""
    from tide.parameters import FeedForward, HeadQuery, NodeLinear, NodeVector

    modules = list(model.named_modules())
    before = {id(module): module.forward for _, module in modules}
    with stage_hooks(model, observer):
        originals = []
        seen = {(id(module), "forward") for _, module in modules
                if module.forward != before[id(module)]}

        def wrap(obj, name, label):
            key = id(obj), name
            if key in seen:
                return
            original, owned = getattr(obj, name), name in vars(obj)
            seen.add(key)

            @wraps(original)
            def run(*args, **kwargs):
                with observer.scope(label):
                    return original(*args, **kwargs)

            originals.append((obj, name, original, owned))
            setattr(obj, name, run)

        try:
            for name, module in modules:
                label = None
                if isinstance(module, FeedForward):
                    label = "ffn"
                elif isinstance(module, HeadQuery):
                    label = "head_query"
                elif isinstance(module, NodeLinear):
                    label = "node_linear." + name
                elif isinstance(module, NodeVector):
                    label = "node_vector." + name
                elif isinstance(module, nn.Linear):
                    label = "linear." + name
                elif isinstance(module, nn.RMSNorm):
                    label = "rms_norm." + name
                if label is not None:
                    wrap(module, "forward", label)

            if hasattr(model, "graph"):
                for module_name in ("tide.parameters", "tide.reception"):
                    wrap(importlib.import_module(module_name), "fused_node_linears", "node_linears.fused")
                for module_name in ("tide.parameters", "tide.kv", "tide.ssd_memory",
                                    "tide.reception", "tide.region", "tide.region_batch"):
                    wrap(importlib.import_module(module_name), "norm", "normalization")
                kv = importlib.import_module("tide.kv")
                for name, label in (("cache_order", "kv.history.order"),
                                    ("chronological_compaction", "kv.history.compaction"),
                                    ("_take", "kv.history.gather"), ("rope", "kv.rope")):
                    wrap(kv, name, label)
                for memory in model.memories.values():
                    for name, label in (("_finish_tile", "kv.query_tile"),
                                        ("_summaries", "ssd.summaries"),
                                        ("_initial", "ssd.initial_state")):
                        if hasattr(memory, name):
                            wrap(memory, name, label)
            yield
        finally:
            for obj, name, original, owned in reversed(originals):
                if owned:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)


class OperatorTimes(TorchDispatchMode, Paths):
    """Synchronized dispatcher timings grouped by path, op and tensor metadata.

Use ``with observer:`` to intercept forward, backward or optimizer operations;
``observer.scope(name)`` supplies path labels without any stage timing. Each
operator waits for prior device work before starting its timer. Descriptors and
aggregation are outside that timer. Only shapes, strides, dtypes and devices are
retained, never tensor payloads. Non-tensor argument values are not grouping keys.
"""
    def __init__(self, device, *, clock=time.perf_counter, sync=synchronize):
        TorchDispatchMode.__init__(self)
        Paths.__init__(self)
        self.device, self.clock, self.sync = device, clock, sync
        self.groups = defaultdict(lambda: dict(calls=0, seconds=0.))
        self.failures = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        path, name = self.path, str(func)
        before = tuple(_descriptor(tensor) for tensor in _tensors((args, kwargs)))
        phase = "entry"
        try:
            self.sync(self.device)
            start = self.clock()
            phase = "dispatch"
            result = func(*args, **kwargs)
            phase = "completion"
            self.sync(self.device)
            elapsed = self.clock() - start
        except BaseException as error:
            self.failures.append(dict(path=path, operator=name, boundary=phase, error=repr(error)))
            raise
        after = tuple(_descriptor(tensor) for tensor in _tensors(result))
        row = self.groups[path, name, before, after]
        row["calls"] += 1
        row["seconds"] += elapsed
        return result

    def report(self):
        rows = [dict(path=path, operator=op, **row,
                     inputs=[_descriptor_json(tensor) for tensor in before],
                     outputs=[_descriptor_json(tensor) for tensor in after])
                for (path, op, before, after), row in sorted(self.groups.items())]
        return dict(
            rows=rows,
            dispatched_calls=sum(row["calls"] for row in rows),
            operator_seconds=sum(row["seconds"] for row in rows),
            failed_calls=len(self.failures),
            failures=list(self.failures),
            boundary="entry synchronize completes before the operator timer starts; dispatch and completion synchronize are timed; descriptor collection and aggregation are excluded",
            interpretation="separate intrusive operator pass on the selected device; rows group tensor shapes, strides, dtypes and devices, not scalar argument values; totals include completed dispatches only and exclude Python gaps, descriptor bookkeeping and entry waits; summed seconds are neither normal wall time nor hardware kernel time; dispatcher operations can contain multiple native kernels or no device kernel",
        )
