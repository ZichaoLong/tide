"""Exclusive diagnostic hooks; never installed during normal throughput timing.

Paths preserve nesting and distinguish forward/flush/backward recomputation.
The synchronized wall observer changes scheduling. Its exclusive values are a
partition of that intrusive observation, not shares of uninstrumented latency.
"""
from collections import defaultdict
from contextlib import contextmanager
import importlib
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from benchmarks.run import synchronize


class Paths:
    def __init__(self):
        self.stack = []

    @contextmanager
    def scope(self, name):
        self.stack.append(name)
        try:
            yield
        finally:
            self.stack.pop()

    @property
    def path(self):
        return "/".join(self.stack) or "outside"


class StageTimes(Paths):
    def __init__(self, device, *, clock=time.perf_counter, sync=synchronize):
        super().__init__()
        self.device, self.clock, self.sync = device, clock, sync
        self.children = []
        self.rows = defaultdict(lambda: dict(calls=0, inclusive_seconds=0., exclusive_seconds=0.))

    @contextmanager
    def scope(self, name):
        self.stack.append(name)
        self.children.append(0.)
        start = self.clock()
        try:
            self.sync(self.device)
            yield
        finally:
            self.sync(self.device)
            elapsed = self.clock()-start
            row = self.rows[self.path]
            row["calls"] += 1
            row["inclusive_seconds"] += elapsed
            row["exclusive_seconds"] += elapsed-self.children.pop()
            self.stack.pop()
            if self.children:
                self.children[-1] += elapsed

    def report(self):
        return dict(rows=[dict(path=path, **row) for path, row in sorted(self.rows.items())],
                    boundary="synchronize before and after every nested scope; synchronization and hook overhead included",
                    interpretation="inclusive rows overlap and must not be summed; exclusive rows partition only this intrusive observation, not normal wall time; unwrapped Python/dispatch/publication remains in its parent")


@contextmanager
def stage_hooks(model, observer):
    """Cover both serial and region-batched Tide plus complete Transformer blocks.

    Global function aliases are temporarily replaced. Call only in an exclusive
    worker, without WorkAudit or another hook session in the same interpreter.
    """
    originals, seen = [], set()

    def wrap(obj, name, label):
        key = id(obj), name
        if key in seen:
            return
        seen.add(key)
        original, owned = getattr(obj, name), name in vars(obj)
        def run(*args, **kwargs):
            with observer.scope(label):
                return original(*args, **kwargs)
        originals.append((obj, name, original, owned))
        setattr(obj, name, run)

    try:
        if hasattr(model, "graph"):
            for name, label in (("_incoming", "messages.gather"),
                                ("_region_batches", "regions.group"),
                                ("_evaluate_region", "region.evaluate"),
                                ("_score_regions", "regional.scoring")):
                wrap(model, name, label)
            for projection in model.inputs.values():
                wrap(projection, "forward", "input.projection")
            wrap(model.receive, "forward", "reception")
            wrap(model.describe, "forward", "description")
            for regional in model.regional:
                wrap(regional, "forward", "regional.scoring")
            for kind, memory in model.memories.items():
                wrap(memory, "prepare", "memory."+kind+".prepare")
                wrap(memory, "finish", "memory."+kind+".full")
            for module_name in ("tide.model", "tide.region_batch"):
                wrap(importlib.import_module(module_name), "region_control", "allocation")
            batch = importlib.import_module("tide.region_batch")
            for name, label in (("evaluate_regions", "regions.evaluate"),
                                ("join_states", "state.join"), ("split_state", "state.split")):
                wrap(batch, name, label)
            for module_name in ("tide.kv", "tide.ssd_memory"):
                wrap(importlib.import_module(module_name), "select_queries", "queries.pack")
            wrap(importlib.import_module("tide.layout").Queries, "publish", "queries.publish")
            for module_name in ("tide.kv", "tide.reception", "tide.region", "tide.region_batch"):
                wrap(importlib.import_module(module_name), "weighted_attention", "attention")
            ssd = importlib.import_module("tide.ssd_memory")
            wrap(ssd, "ssd_scan", "ssd.scan")
            wrap(ssd, "logical_conv", "ssd.convolution")
        else:
            wrap(model, "_layout", "layout.rope_masks")
            wrap(model.final_norm, "forward", "final_norm")
            for block in model.blocks:
                wrap(block, "forward", "block")
                for name in ("attention_norm", "qkv", "output", "ffn_norm", "ffn"):
                    wrap(getattr(block, name), "forward", name)
                wrap(block.ffn.gate_up, "forward", "gate_up")
                wrap(block.ffn.down, "forward", "down")
            baseline = importlib.import_module("benchmarks.baselines")
            wrap(baseline, "_rope", "rope")
            wrap(baseline, "weighted_attention", "attention")
        yield
    finally:
        for obj, name, original, owned in reversed(originals):
            if owned:
                setattr(obj, name, original)
            else:
                delattr(obj, name)


def _tensors(value):
    return [tensor for tensor in tree_flatten(value)[0] if isinstance(tensor, torch.Tensor)]


def _descriptor(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))


def _descriptor_json(value):
    shape, stride, dtype, device = value
    return dict(shape=list(shape), stride=list(stride), dtype=dtype, device=device)


class ShapeAudit(TorchDispatchMode, Paths):
    """Actual dispatched operators, including backward and optimizer, by phase.

    This records logical shapes/strides, not opaque native kernel algorithms.
    WorkAudit support copies are deliberately absent from this pass.
    """
    def __init__(self):
        TorchDispatchMode.__init__(self)
        Paths.__init__(self)
        self.groups = defaultdict(int)
        self.calls = defaultdict(int)
        self.movement = defaultdict(lambda: dict(calls=0, logical_output_bytes=0))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name, path = str(func), self.path
        self.calls[path, name] += 1
        if any(token in name for token in ("mm.", "bmm.", "addmm.", "matmul.", "linear.", "attention", "convolution")):
            before = tuple(_descriptor(t) for t in _tensors((args, kwargs)))
            after = tuple(_descriptor(t) for t in _tensors(result))
            self.groups[path, name, before, after] += 1
        if any(name.startswith("aten."+token+".") for token in
               ("_to_copy", "copy_", "clone", "cat", "index", "index_put", "index_put_",
                "index_select", "gather", "scatter", "scatter_", "index_copy", "index_copy_")):
            row = self.movement[path, name]
            row["calls"] += 1
            row["logical_output_bytes"] += sum(t.numel()*t.element_size() for t in _tensors(result))
        return result

    def report(self):
        return dict(matrix_groups=[dict(path=path, operator=op, calls=count,
                                        inputs=[_descriptor_json(t) for t in before],
                                        outputs=[_descriptor_json(t) for t in after])
                                   for (path, op, before, after), count in sorted(self.groups.items())],
                    operators=[dict(path=path, operator=op, calls=count)
                               for (path, op), count in sorted(self.calls.items())],
                    materializations=[dict(path=path, operator=op, **row)
                                      for (path, op), row in sorted(self.movement.items())],
                    dispatched_calls=sum(self.calls.values()),
                    bf16_matrix_observed=any(t[2] == "torch.bfloat16" for (_, _, _, after) in self.groups for t in after),
                    interpretation="separate intrusive complete-training dispatch pass; shapes include padding; logical materialized bytes are not hardware traffic; native attention/scan work is opaque here")


class SavedAudit(Paths):
    """Saved tensor lifetimes by innermost stage; aliases overlap stages."""
    def __init__(self, model):
        super().__init__()
        from benchmarks.memory_ledger import SavedStorageAudit
        self.total = SavedStorageAudit(list(model.parameters()))
        def make_row():
            row = SavedStorageAudit()
            row.existing = self.total.existing
            return row
        self.rows = defaultdict(make_row)

    def pack(self, tensor):
        # Both owners hold detached aliases until autograd destroys this tuple.
        # Destruction, not unpack, releases the ledger's retained storage.
        return self.rows[self.path].pack(tensor), self.total.pack(tensor)

    @staticmethod
    def unpack(saved):
        row, total = saved
        row.owner.unpack_calls += 1
        total.owner.unpack_calls += 1
        return row.tensor

    def report(self):
        result = dict(rows=[dict(path=path, **row.report()) for path, row in sorted(self.rows.items())],
                      total=self.total.report(),
                      interpretation="separate saved_tensors_hooks pass; events count every save; each row is peak live capacity attributed to its innermost save site, not allocator peak; aliases and lifetimes overlap between rows, so their peaks must not be summed; additional storage here excludes parameters only; checkpoint internal hooks can hide recomputed saves")
        # SavedStorageAudit's generic wording mentions a broader possible
        # existing inventory. This pass deliberately excludes parameters only.
        for row in [result["total"], *result["rows"]]:
            row["additional_scope"] = "excludes parameter storages only"
        return result
