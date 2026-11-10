"""Additive anchors: raw left state, sharing, true training and disk restore.

This checks the declared output/terminal-state numeric projection. It does not
replace complete-record, intermediate-root or disconnected-gradient gates.
"""
import subprocess
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from tensor_baseline_state import detached
from layer_tensor_baseline import fixture as layers, run as layer_run
from fc_tensor_baseline import run as fc_run
from tidegraph import Continuation
from tidegraph.records import State
from tidegraph.reference import run as scalar
from tidegraph.compare import equivalent
from tidegraph.weighted import layered_inputs, readout


def fixture(family, dtype):
    if family == "fc":
        from test_fc_tensor_baseline import fixture as fc_fixture
        binding, spec = fc_fixture(dtype)
    else:
        binding, spec = layers(family, width=4, depth=2, heads=2, dtype=dtype)
    # Same parameter, different nodes and private state; both evaluators must
    # accumulate all consumers. FC's regional KV bases are already shared.
    binding.model.nodes[1].full_program.content.down = binding.model.nodes[0].full_program.content.down
    return binding, spec


def initial(binding, spec, batch, start):
    result = {}
    for v, node in enumerate(binding.model.nodes):
        c, ref = node.kernel.config, node.bias
        stamps = torch.tensor([(start-2)*spec.stride+spec.rank(v), (start-1)*spec.stride+spec.rank(v)], dtype=torch.int64)
        def values(shape, offset):
            n = 1
            for size in shape:
                n *= size
            return torch.linspace(.01+offset, .06+offset, n, dtype=ref.dtype).reshape(shape).requires_grad_()
        if hasattr(c, "window"):
            result[v] = dict(times=stamps, keys=values((batch, 2, c.width), .01),
                             values=values((batch, 2, c.width), -.03),
                             summaries=values((batch, 2, c.summary_width), .02),
                             weights=values((batch, 2), .3))
        else:
            result[v] = dict(time=stamps[-1], H=values((batch, *c.matrix_shape), -.02),
                             Z=values((batch, 2, c.summary_width), .02), W=values((batch, 2), .3),
                             conv_times=stamps, conv_values=values((batch, 2, c.conv_width), -.01))
    return result


def clone_roots(states):
    return {v: {name: x.detach().clone().requires_grad_(x.requires_grad) for name, x in row.items()}
            for v, row in states.items()}


def candidate(family, binding, spec, x, left, start):
    if family == "fc":
        result = fc_run(binding, spec, x, start=start, left=left)
        return result["hidden"], result["states"]
    output, states = layer_run(binding, spec, x, start=start,
                               left=None if left is None else [left[v] for v in range(len(left))])
    return output, dict(enumerate(states))


def reference(binding, spec, x, left, start):
    batch = x.shape[0]
    q = Continuation(binding.graph.identity, batch, cut=start*spec.stride)
    for v, row in left.items():
        last = int(row["times"][-1] if "times" in row else row["time"])
        for b in range(batch):
            slots = {name: value if name in {"time", "times", "conv_times"} else value[b]
                     for name, value in row.items()}
            q.states[b, v] = State(torch.zeros_like(binding.model.nodes[v].bias), last, start, slots)
    q.ledger = {(b, p): (start-1, (start-1)*spec.stride+spec.rank(v))
                for b in range(batch) for p, v in enumerate(binding.graph.inputs)}
    stop = (start+x.shape[1])*spec.stride
    result = scalar(binding.graph, binding.model, q, layered_inputs(spec, x, start_position=start), stop, sealed_until=stop)
    output = torch.stack([torch.stack([readout(result.outputs,
        [((start+t)*spec.stride+spec.rank(v), p) for p, v in enumerate(binding.graph.outputs)], x[b, t], batch=b)
        for t in range(x.shape[1])]) for b in range(batch)])
    states = {}
    for v in range(len(binding.graph.nodes)):
        states[v] = {name: value if name in {"time", "times", "conv_times"}
                     else torch.stack([result.continuation.states[b, v].slots[name] for b in range(batch)])
                     for name, value in result.continuation.states[0, v].slots.items()}
    return output, states


def objective(output, states):
    return output.square().sum()+.01*sum(value.square().sum() for row in states.values()
                                        for value in row.values() if value.is_floating_point())


def floating(states):
    return [value for row in states.values() for value in row.values() if value.is_floating_point()]


def train_step(family, binding, spec, optimizer, material, old, position):
    optimizer.zero_grad(set_to_none=True)
    output, state = candidate(family, binding, spec, material, old, position)
    objective(output, state).backward()
    grads = [None if p.grad is None else p.grad.clone() for p in binding.model.parameters()]
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    optimizer.step()
    return output.detach(), detached(state), grads


@pytest.mark.parametrize("family", ["kv", "ssd", "fc"])
def test_three_sgd_steps_against_independent_scalar_schedule(dtype, family):
    binding, spec = fixture(family, dtype)
    oracle, ospec = fixture(family, dtype)
    optimizer = torch.optim.SGD(binding.model.parameters(), lr=.001, momentum=.8)
    scalar_optimizer = torch.optim.SGD(oracle.model.parameters(), lr=.001, momentum=.8)
    left = initial(binding, spec, 1, 3)
    old = clone_roots(left)
    material = torch.linspace(-.2, .4, 8, dtype=dtype).reshape(1, 2, 4)
    for position in (3, 5, 7):
        actual, left, grads = train_step(family, binding, spec, optimizer, material, left, position)
        scalar_optimizer.zero_grad(set_to_none=True)
        expected, state = reference(oracle, ospec, material, old, position)
        objective(expected, state).backward()
        old = detached(state)
        scalar_optimizer.step()
        equivalent(actual, expected)
        equivalent(left, old)
        for value, other, grad in zip(binding.model.parameters(), oracle.model.parameters(), grads):
            equivalent(value, other)
            equivalent(torch.zeros_like(value) if grad is None else grad,
                       torch.zeros_like(other) if other.grad is None else other.grad)
            # Numeric projection: the eliminated dead fields may be None where
            # scalar execution has connected zero. Do not certify slot identity.
            equivalent(optimizer.state[value].get("momentum_buffer", torch.zeros_like(value)),
                       scalar_optimizer.state[other].get("momentum_buffer", torch.zeros_like(other)))


@pytest.mark.parametrize("family", ["kv", "ssd", "fc"])
def test_nonzero_raw_left_shared_parameters_and_independent_roots(dtype, family):
    binding, spec = fixture(family, dtype)
    oracle, ospec = fixture(family, dtype)
    left = initial(binding, spec, 2, 3)
    old = clone_roots(left)
    x = torch.linspace(-.2, .4, 16, dtype=dtype).reshape(2, 2, 4).requires_grad_()
    ox = x.detach().clone().requires_grad_()
    actual, states = candidate(family, binding, spec, x, left, 3)
    expected, estate = reference(oracle, ospec, ox, old, 3)
    equivalent(actual, expected)
    equivalent(states, estate)
    roots = [x, *floating(left), *(p for p in binding.model.parameters() if p.requires_grad)]
    oroots = [ox, *floating(old), *(p for p in oracle.model.parameters() if p.requires_grad)]
    grads = torch.autograd.grad(objective(actual, states), roots, allow_unused=True)
    ograd = torch.autograd.grad(objective(expected, estate), oroots, allow_unused=True)
    for a, e, root in zip(grads, ograd, roots):
        equivalent(torch.zeros_like(root) if a is None else a, torch.zeros_like(root) if e is None else e)


@pytest.mark.parametrize("family", ["kv", "ssd", "fc"])
def test_cross_window_retained_graph_and_original_time(dtype, family):
    binding, spec = fixture(family, dtype)
    # Above float64's exact integer range. The old/new time subtraction and
    # fixed convolution slots must not be computed in floating coordinates.
    start = (2**53)//spec.stride+2
    x = torch.linspace(-.2, .4, 24, dtype=dtype).reshape(2, 3, 4).requires_grad_()
    left = initial(binding, spec, 2, start)
    a, state_a = candidate(family, binding, spec, x, left, start)
    first, mid = candidate(family, binding, spec, x[:, :1], left, start)
    second, state_b = candidate(family, binding, spec, x[:, 1:], mid, start+1)
    joined = torch.cat((first, second), 1)
    equivalent(joined, a)
    equivalent(state_b, state_a)
    roots = [x, *floating(left), *(p for p in binding.model.parameters() if p.requires_grad)]
    ga = torch.autograd.grad(objective(a, state_a), roots, allow_unused=True)
    gb = torch.autograd.grad(objective(joined, state_b), roots, allow_unused=True)
    for g, h, root in zip(ga, gb, roots):
        equivalent(torch.zeros_like(root) if g is None else g, torch.zeros_like(root) if h is None else h)


@pytest.mark.parametrize("family", ["kv", "ssd", "fc"])
def test_three_optimizer_steps_and_exact_projection_checkpoint(tmp_path, dtype, family):
    binding, spec = fixture(family, dtype)
    optimizer = torch.optim.SGD(binding.model.parameters(), lr=.001, momentum=.8)
    left = initial(binding, spec, 1, 3)
    material = torch.linspace(-.2, .4, 8, dtype=dtype).reshape(1, 2, 4)
    _, left, _ = train_step(family, binding, spec, optimizer, material, left, 3)
    path = tmp_path/"projection.pt"
    torch.save(dict(schema="tensor-baseline-projection-v1", graph=binding.graph.identity, next_position=5,
                    states=left, material=material, model=binding.model.state_dict(), optimizer=optimizer.state_dict()), path)
    result_path = tmp_path/"restored.pt"
    # A new process/owner decodes actual raw tensors, without a reference run.
    subprocess.run([sys.executable, __file__, family, str(path), str(result_path)], check=True, timeout=45)
    restored = torch.load(result_path, weights_only=True)
    for index, position in enumerate((5, 7)):
        a, left, ga = train_step(family, binding, spec, optimizer, material, left, position)
        b, right, gb, model, optim = restored[index]
        for value, expected in ((a, b), (left, right), (ga, gb),
                                (binding.model.state_dict(), model), (optimizer.state_dict(), optim)):
            equivalent(value, expected, atol=0, rtol=0)


@pytest.mark.parametrize("family", ["kv", "ssd", "fc"])
def test_left_metadata_and_start_rejected(dtype, family):
    binding, spec = fixture(family, dtype)
    x = torch.ones(1, 1, 4, dtype=dtype)
    left = initial(binding, spec, 1, 3)
    key = "times" if family != "ssd" else "time"
    left[0][key] = left[0][key].to(dtype)
    with pytest.raises(ValueError, match="dtype"):
        candidate(family, binding, spec, x, left, 3)
    for start in (True, -1, 1.5, 2**63):
        with pytest.raises(ValueError, match="position|overflows"):
            candidate(family, binding, spec, x, None, start)


@pytest.mark.parametrize("family", ["kv", "ssd"])
def test_expired_left_entries_and_fixed_slot_gaps(dtype, family):
    from dataclasses import replace
    from tidegraph.weighted import bind
    binding, spec = fixture(family, dtype)
    if family == "kv":
        binding = bind(spec.graph, memories=tuple(replace(c, window=2*spec.stride) for c in binding.memories),
                       regions=binding.regions, dtype=dtype)
        spec = replace(spec, graph=binding.graph)
    left = initial(binding, spec, 2, 3)
    # Advance original position by four without compacting the old timestamps.
    x = torch.linspace(-.2, .4, 16, dtype=dtype).reshape(2, 2, 4).requires_grad_()
    actual, states = candidate(family, binding, spec, x, left, 7)
    expected, estate = reference(binding, spec, x, left, 7)
    equivalent(actual, expected)
    equivalent(states, estate)


if __name__ == "__main__":
    import copy
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    family, source, destination = sys.argv[1:]
    loaded = torch.load(source, weights_only=True)
    binding, spec = fixture(family, loaded["material"].dtype)
    assert loaded["schema"] == "tensor-baseline-projection-v1" and loaded["graph"] == binding.graph.identity
    optimizer = torch.optim.SGD(binding.model.parameters(), lr=.001, momentum=.8)
    binding.model.load_state_dict(loaded["model"])
    optimizer.load_state_dict(loaded["optimizer"])
    state, records = loaded["states"], []
    for position in (loaded["next_position"], loaded["next_position"]+2):
        out, state, grads = train_step(family, binding, spec, optimizer, loaded["material"], state, position)
        records.append(copy.deepcopy((out, state, grads, binding.model.state_dict(), optimizer.state_dict())))
    torch.save(records, destination)
