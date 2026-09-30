"""Independent Python autograd inputs and explicit resident consumer roots."""
import os
import torch
from tidegraph import (Edge, Graph, Node, Region, GraphConfig, GraphRuntime, External,
                       ExecutionOptions, ExecutionPlacement, ResidentLimits)


def configuration(family, full="tanh"):
    nodes = tuple(Node(i // 2, memory="ema", full=full) for i in range(4))
    edges = (Edge(0, 2, 1), Edge(0, 2, 1), Edge(1, 2, 1), Edge(1, 3, 1))
    if family != "settle":
        edges += (Edge(0, 3, 2),)
    if family == "pdg":
        edges += (Edge(3, 0, 3),)
    return GraphConfig(family, Graph(nodes, edges, (Region(1), Region(1)), (0, 1), (2, 3)),
                       width=4, ranks=(1, 2) if family == "settle" else ())


def runtime(family, device, schedule="greedy", full="tanh"):
    cfg = configuration(family, full)
    if device == "cpu":
        r = GraphRuntime(cfg, device=device, options=ExecutionOptions(schedule="reference", packed=False, trace=True))
    else:
        r = GraphRuntime(cfg, device=device, native_library=os.environ["TIDE_BUILD_DIR"],
            resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],
            options=ExecutionOptions(implementation="native", schedule=schedule, trace=True,
                placement=ExecutionPlacement(preset="resident"),
                resident_limits=ResidentLimits(queue=96, arrivals=128, outputs=128, trace=512)))
    r.model.nodes[2].weight = r.model.nodes[0].weight
    r.model.nodes[3].read = r.model.nodes[0].bias
    r.model.input_scale[1] = r.model.agg_scale[0]
    return r


def inputs(session, values, start, stop):
    if session.runtime.spec:
        return values[:, start:stop], {}
    return [External(b, p, t, t, values[b, t]) for b in range(len(values))
            for p in range(len(session.runtime.graph.inputs)) for t in range(start, stop)], dict(stop=stop, sealed_until=stop)


def roots(session, window, mode):
    if mode == "none":
        return session.cotangents(window)
    with torch.enable_grad():
        leaf = window.outputs.values.detach().requires_grad_(True)
        safe = torch.where(window.outputs.valid[:, None], leaf, torch.zeros_like(leaf))
        loss = safe.square().sum() * (0 if mode == "zero" else .0625)
        gradient, = torch.autograd.grad(loss, (leaf,))
    factor = 0 if mode == "zero" else 1
    return session.cotangents(window, outputs=gradient.detach(),
        pending=torch.full_like(window.pending_values, factor * .015625),
        final=torch.full_like(window.state_values, factor * .03125))


def terms(result, continuation, mode):
    if mode == "none":
        return []
    factor = 0 if mode == "zero" else 1
    return ([x.square().sum() * (factor * .0625) for _, _, _, x in result.outputs]
            + [s.value.sum() * (factor * .03125) for s in continuation.states.values()]
            + [a.value.sum() * (factor * .015625) for a in continuation.pending])


def compare_parameters(actual, model):
    expected = model.state_dict(keep_vars=True)
    for name, value in actual["parameters"].items():
        torch.testing.assert_close(value, expected[name].detach(), atol=1e-6, rtol=1e-5)


def compare_gradients(gradient, model, expected, input_leaf, input_gradient):
    named = model.state_dict(keep_vars=True)
    packed, connected = gradient.values.cpu(), gradient.connected.cpu()
    for i, name in enumerate(gradient.names):
        ref = expected[id(named[name])]
        assert connected[i].item() == (ref is not None), name
        if ref is not None:
            value = packed.narrow(0, gradient.offsets[i], ref.numel()).reshape(ref.shape)
            torch.testing.assert_close(value, ref, atol=1e-6, rtol=1e-5)
    actual = torch.zeros_like(input_leaf)
    for boundary in gradient.boundaries:
        coordinates, valid = boundary.coordinates.cpu(), boundary.valid.cpu()
        values, on = boundary.values.cpu(), boundary.connected.cpu()
        for row in torch.nonzero(valid & on, as_tuple=False).flatten().tolist():
            b, node, time, kind, source, position = coordinates[row].tolist()
            if kind == 0:
                actual[b, position].add_(values[row])
    if input_gradient is None:
        assert not actual.any()
    else:
        torch.testing.assert_close(actual, input_gradient, atol=1e-6, rtol=1e-5)


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            tree_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            tree_equal(x, y)
    else:
        assert a == b
