"""Diagnostic CPU outer reverse graph with FP64 join buffers (Torch 2.12 only).

Local VJPs receive original dtype. Private autograd.grad calls inside custom
Functions still use the normal engine. This is neither a production backward
engine nor an independent mathematical oracle. Only the companion workload
driver's first-order scalar losses, without hooks/retained gradients, are used.
"""
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
import sys
from types import MethodType
import torch


class JoinProbe:
    def __init__(self):
        self.counts = Counter()

    @contextmanager
    def install(self):
        if not torch.__version__.startswith("2.12."):
            raise ValueError("private reverse-node diagnostic requires tested Torch 2.12")
        if any(n.startswith("tidegraph.weighted") for n in sys.modules):
            raise ValueError("join diagnostic must start before importing weighted modules")
        original_apply = torch.autograd.Function.apply.__func__
        original_backward = torch.Tensor.backward

        def capture(cls, *args, **kwargs):
            outputs = original_apply(cls, *args, **kwargs)
            for out in outputs if isinstance(outputs, tuple) else (outputs,):
                if isinstance(out, torch.Tensor) and out.grad_fn is not None:
                    out.grad_fn._join_tensor_args = tuple(i for i, x in enumerate(args)
                                                          if isinstance(x, torch.Tensor))
            return outputs

        torch.autograd.Function.apply = classmethod(capture)
        probe = self
        def backward(value, gradient=None, retain_graph=None, create_graph=False, inputs=None):
            return probe.backward(value, gradient, retain_graph, create_graph, inputs)
        torch.Tensor.backward = backward
        try:
            yield self
        finally:
            torch.autograd.Function.apply = classmethod(original_apply)
            torch.Tensor.backward = original_backward
            # Formula modules may bind `norm = _Normalize.apply` on import.
            # Restore those aliases as well as the base class method.
            for module in tuple(sys.modules.values()):
                if module is None:
                    continue
                for name, item in tuple(vars(module).items()):
                    if isinstance(item, MethodType) and item.__func__ is capture:
                        setattr(module, name, MethodType(original_apply, item.__self__))

    def backward(self, value, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        if create_graph or inputs is not None or retain_graph is not None:
            raise ValueError("join diagnostic accepts whole first-order scalar losses only")
        if value.device.type != "cpu" or value.numel() != 1 or value.grad_fn is None:
            raise ValueError("join diagnostic requires a differentiable CPU scalar")
        pending, edges, todo = Counter(), {}, [value.grad_fn]
        while todo:
            node = todo.pop()
            if node in edges:
                continue
            edges[node] = list(node.next_functions)
            for child, _ in edges[node]:
                if child is not None:
                    pending[child] += 1
                    todo.append(child)
        bars = defaultdict(lambda: defaultdict(list))
        bars[value.grad_fn][value.output_nr].append(torch.ones_like(value) if gradient is None else gradient)
        ready, visited = deque([value.grad_fn]), 0
        with torch.no_grad():
            while ready:
                node = ready.popleft()
                slots = bars.pop(node, {})
                gs = []
                for i, metadata in enumerate(node._input_metadata):
                    terms = slots.get(i, [])
                    if not terms:
                        gs.append(None)
                        continue
                    if any(g.device.type != "cpu" or g.dtype not in (torch.float32, torch.float64) for g in terms):
                        raise ValueError("join diagnostic supports CPU FP32/FP64 only")
                    g = terms[0]
                    if len(terms) > 1:
                        g = g.double()
                        for term in terms[1:]:
                            g = g+term.double()
                        self.counts["joined_slots"] += 1
                        self.counts["joined_terms"] += len(terms)
                    gs.append(g.to(dtype=metadata.dtype))
                if type(node).__name__ == "AccumulateGrad":
                    if len(gs) != 1 or node.variable.grad is not None:
                        raise ValueError("join diagnostic requires zeroed leaf gradients")
                    if gs[0] is not None:
                        node.variable.grad = gs[0]
                    outputs = ()
                elif not any(g is not None for g in gs):
                    outputs = (None,)*len(edges[node])
                elif callable(node):
                    values = node(*gs)
                    outputs = values if isinstance(values, tuple) else (values,)
                else:
                    values = node.apply(*gs)
                    values = values if isinstance(values, tuple) else (values,)
                    outputs = tuple(values[i] for i in node._join_tensor_args)
                if len(outputs) != len(edges[node]):
                    raise RuntimeError("private reverse-node output mapping changed")
                for (child, slot), g in zip(edges[node], outputs):
                    if child is None:
                        continue
                    if g is not None:
                        shape = tuple(child._input_metadata[slot].shape)
                        if tuple(g.shape) != shape:
                            g = g.sum_to_size(shape)
                            self.counts["broadcast_reductions"] += 1
                        bars[child][slot].append(g)
                    pending[child] -= 1
                    if pending[child] == 0:
                        ready.append(child)
                visited += 1
        if visited != len(edges):
            raise RuntimeError("reverse-node dependency count mismatch")
        self.counts["nodes"] += visited
