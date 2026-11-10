"""Opt-in eager graph census; snapshots contain metadata, never graph owners.

Register all task roots before backward, and register hidden local graphs where
they are constructed. Traversing ``next_functions`` cannot discover graphs held
inside a custom Function. Counts are diagnostic observations, not timing or a
correctness certificate. This module never installs saved-tensor pack hooks:
normal SavedVariable version checks and ownership remain in force.
"""
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
import itertools
import threading
import weakref

import torch

_current = ContextVar("tide_autograd_census", default=None)
_sessions = itertools.count(1)
_tag = "_tide_autograd_census_id"


def current():
    """The active caller scope; pass it explicitly across worker threads."""
    value = _current.get()
    return value if value is not None and value.active else None


def register_private_graph(label, roots):
    """Register already computed private roots; disabled calls do no traversal."""
    owner = current()
    if owner is not None:
        owner.register(roots, label="private/"+label)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensors(child)


def tensor_roots(value):
    """Visit returned Tensor fields without creating projections or objectives.

    Dataclass records, mappings and tuple/list containers are supported. No
    arbitrary object attributes are followed; the caller chooses the scope.
    Repeated references are allowed and deduplicated by ``Census.register``.
    """
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from tensor_roots(getattr(value, field.name))
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensor_roots(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from tensor_roots(child)


class _Storage:
    """An intrusive weak identity survives Python storage-wrapper collection."""
    def __init__(self, storage):
        self.ref = storage._weak_ref()
        self.bytes = storage.nbytes()

    def alive(self):
        return not torch.UntypedStorage._expired(self.ref)

    def __del__(self):
        torch.UntypedStorage._free_weak_ref(self.ref)


@dataclass
class _Node:
    kind: str
    edges: Counter
    null_edges: int
    saved_occurrences: int = 0
    saved_bytes: int = 0
    inaccessible_saved_fields: int = 0
    executions: int = 0


class Census:
    """Observe one eager run without retaining Tensor/Node/Storage owners.

    Python-visible saved fields are sampled at registration, before backward
    releases them. Native opaque nodes require native explicit instrumentation;
    their unexposed internals must not be described as covered by this census.
    Saved bytes are cumulative over unique observed nodes, not allocator peaks.
    Re-registering overlapping roots deduplicates nodes and saved references.
    Hook execution counts include repeated retained backward executions.
    """
    def __init__(self, *, executions=True, saved_tensors=True):
        self.session = next(_sessions)
        self.executions = executions
        self.saved_tensors = saved_tensors
        self.active = False
        self._entered = False
        self._lock = threading.RLock()
        self._nodes, self._groups, self._storage, self._impls = {}, {}, {}, {}
        self._handles = []
        self._next_tensor = itertools.count(1)
        self._saved_impls, self._saved_storage = set(), set()
        self._saved_without_storage = 0

    def __enter__(self):
        if self._entered or current() is not None:
            raise RuntimeError("census scopes are single-use and cannot nest")
        self._entered = self.active = True
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc):
        self.active = False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        _current.reset(self._token)

    def _save(self, value, node):
        node.saved_occurrences += 1
        node.saved_bytes += value.numel()*value.element_size()
        key = value._cdata
        previous = self._impls.get(key)
        if previous is None or previous[0].expired():
            previous = (torch._C._WeakTensorRef(value), next(self._next_tensor))
            self._impls[key] = previous
        self._saved_impls.add(previous[1])
        if value.layout != torch.strided:
            self._saved_without_storage += 1
            return
        storage = value.untyped_storage()
        key = storage._cdata
        # A weak StorageImpl keeps its address distinct even after payload free.
        if key not in self._storage:
            self._storage[key] = _Storage(storage)
        self._saved_storage.add(key)

    def _saved(self, fn, node):
        names = [name for name in dir(fn) if name.startswith("_saved_")]
        if hasattr(type(fn), "saved_tensors"):
            names.append("saved_tensors")
        for name in names:
            try:
                value = getattr(fn, name)
            except RuntimeError:
                node.inaccessible_saved_fields += 1
                continue
            for tensor in _tensors(value):
                self._save(tensor, node)

    def register(self, roots, *, label="outer"):
        """Union all specified Tensor roots, including leaf and unused roots.

        No backward, value reads, synchronization, detach, clone, or saved-tensor
        unpack override is performed. Call on a quiescent graph before backward;
        execution hooks may subsequently run on autograd worker threads.
        """
        if not self.active:
            raise RuntimeError("register requires an active census")
        if not isinstance(label, str) or not label:
            raise ValueError("census label must be a nonempty string")
        if isinstance(roots, torch.Tensor):
            roots = (roots,)
        roots = tuple(roots)
        if not all(isinstance(root, torch.Tensor) for root in roots):
            raise TypeError("census roots must be tensors")
        with self._lock:
            group = self._groups.setdefault(label, dict(nodes=set(), registrations=0,
                                                        roots=0, nondifferentiable_roots=0))
            group["registrations"] += 1
            group["roots"] += len(roots)
            queue = []
            for root in roots:
                if root.requires_grad:
                    queue.append(torch.autograd.graph.get_gradient_edge(root).node)
                else:
                    group["nondifferentiable_roots"] += 1
            visited = set()
            while queue:
                fn = queue.pop()
                if fn in visited:
                    continue
                visited.add(fn)  # Only this traversal temporarily owns wrappers.
                tag = fn.metadata.get(_tag)
                fresh = tag is None or tag[0] != self.session
                if fresh:
                    ident = len(self._nodes)
                    fn.metadata[_tag] = (self.session, ident)
                    edges = Counter()
                    nulls = 0
                    for child, _ in fn.next_functions:
                        if child is None:
                            nulls += 1
                        else:
                            edges[child.name()] += 1
                    node = _Node(fn.name(), edges, nulls)
                    self._nodes[ident] = node
                    if self.saved_tensors:
                        self._saved(fn, node)
                    if self.executions:
                        ref = weakref.ref(self)
                        def executed(grads, *, ref=ref, ident=ident):
                            owner = ref()
                            if owner is not None and owner.active:
                                with owner._lock:
                                    owner._nodes[ident].executions += 1
                            return None
                        self._handles.append(fn.register_prehook(executed))
                else:
                    ident = tag[1]
                group["nodes"].add(ident)
                queue.extend(child for child, _ in fn.next_functions if child is not None)

    def _counts(self, ids):
        types, edges, executed = Counter(), Counter(), Counter()
        nulls = saved = logical = inaccessible = 0
        for ident in ids:
            node = self._nodes[ident]
            types[node.kind] += 1
            executed[node.kind] += node.executions
            edges.update({node.kind+" -> "+kind: count for kind, count in node.edges.items()})
            nulls += node.null_edges
            saved += node.saved_occurrences
            logical += node.saved_bytes
            inaccessible += node.inaccessible_saved_fields
        return dict(unique_nodes=sum(types.values()), connected_edges=sum(edges.values()),
                    null_edges=nulls, node_types=dict(sorted(types.items())),
                    edge_types=dict(sorted(edges.items())),
                    node_executions=sum(executed.values()),
                    execution_types=dict(sorted(executed.items())), saved_tensor_occurrences=saved,
                    saved_logical_bytes=logical, inaccessible_saved_fields=inaccessible)

    def snapshot(self):
        """Return JSON-compatible metadata; overlapping groups are not additive."""
        with self._lock:
            groups = {label: {**{k: v for k, v in g.items() if k != "nodes"},
                              **self._counts(g["nodes"])} for label, g in self._groups.items()}
            total = self._counts(self._nodes)
            total.update(saved_unique_tensor_impls=len(self._saved_impls),
                saved_unique_storages=len(self._saved_storage),
                saved_unique_storage_bytes=sum(self._storage[k].bytes for k in self._saved_storage),
                observed_storage_bytes_still_alive=sum(v.bytes for v in self._storage.values() if v.alive()),
                saved_occurrences_without_storage=self._saved_without_storage)
            return dict(schema="tide-autograd-census-v1", total=total, groups=groups,
                scope=dict(graphs="specified outer roots and explicitly registered private graphs",
                           saved_tensors="Python-visible saved fields at first registration; native opaque internals require explicit instrumentation",
                           storage="StorageImpl identity and allocated capacity, not physical overlap or allocator peak",
                           lifetime="metadata and weak identities only; no retained graph/tensor/storage owners",
                           executions="registered eager nodes only; counts are not timings",
                           enabled_execution_hooks=self.executions,
                           enabled_saved_inspection=self.saved_tensors))
