"""Independent literal SSD/state binding, gradients and continuation checks."""
import copy
import math

import pytest
import torch
from torch.nn import functional as F

from tide.ssd import ConvState
from tide.ssd_memory import SSDMemory, SSDState


def linear(layer, x, owner):
    weight = layer.weight[layer.groups[owner]]
    if layer.shared:
        weight = weight + layer.left[owner] @ layer.right[owner]
    return weight @ x


def vector(layer, owner):
    return layer.value[layer.groups[owner]]


def normalized(x):
    return x / torch.sqrt(x.square().sum() / x.numel() + 1e-5)


def literal(module, u, mass, times, owners, state, control):
    """One Python dictionary and matrix recurrence per node/sample.

    No production projection, scan, convolution, packing or FFN helper is used.
    Matrix weights are materialized independently, including node adapters.
    """
    descriptions, outputs, states, caches = [], [], [], []
    beta = module.short_decay
    for node in range(u.shape[0]):
        node_desc, node_out, node_state, node_cache = [], [], [], []
        owner = int(owners[node])
        kernel = vector(module.conv_kernel, owner).reshape(4, module.d + 64)
        bias = vector(module.conv_bias, owner)
        for batch in range(u.shape[1]):
            matrix = state.matrix[node, batch]
            short, long = state.short[node, batch], state.long[node, batch]
            ws, wl = state.short_weight[node, batch], state.long_weight[node, batch]
            saved_time = int(state.times[node, batch])
            cache = {int(t): v for t, v in zip(state.conv.times[node, batch], state.conv.values[node, batch])
                     if int(t) >= 0}
            desc, full = [], []
            for i in range(u.shape[2]):
                content, time = u[node, batch, i], int(times[node, batch, i])
                x0 = normalized(content)
                p = linear(module.summary, x0, owner)
                projection = linear(module.input, x0, owner)
                gate, conv, dt_raw = projection.split((module.d, module.d + 64, module.heads))
                convolved = kernel[0] * conv + bias
                for lag in range(1, 4):
                    convolved = convolved + kernel[lag] * cache.get(time - lag * module.stride, 0)
                value, B, C = F.silu(convolved).split((module.d, 32, 32))
                if mass[node, batch, i] > 0:
                    weight = mass[node, batch, i].clamp(max=1)
                    dt = weight * F.softplus(dt_raw + vector(module.dt_bias, owner))
                    decay = torch.exp(-vector(module.a_log, owner).exp() * dt)
                    matrix = (decay[:, None, None] * matrix
                              + dt[:, None, None] * value.reshape(module.heads, 32, 1) * B[None, None, :])
                    decay_short = math.exp(-beta * max(0, time - saved_time))
                    short, long = decay_short * short + weight * p, long + weight * p
                    ws, wl = decay_short * ws + weight, wl + weight
                    saved_time = time
                    cache = {t: v for t, v in cache.items() if time - 3 * module.stride <= t < time}
                    cache[time] = weight * conv
                effective = math.exp(-beta * max(0, time - saved_time))
                short_now, ws_now = effective * short, effective * ws
                read = torch.stack([head @ C for head in matrix]).flatten()
                denominator = ws_now if ws_now > 1 else torch.ones_like(ws_now)
                desc.append((p, short_now / denominator, linear(module.read, read, owner), ws_now, wl))
                if control[node, batch, i] > 0:
                    skip = vector(module.skip, owner).repeat_interleave(32)
                    gated = (read + skip * value) * F.silu(gate)
                    mix = linear(module.output, vector(module.rho, owner) * normalized(gated), owner)
                    x = content + mix
                    z = normalized(x)
                    hidden = F.silu(linear(module.ffn.gate, z, owner)) * linear(module.ffn.up, z, owner)
                    full.append(x + linear(module.ffn.down, hidden, owner))
                else:
                    full.append(torch.zeros_like(content))
            node_desc.append(tuple(torch.stack([row[j] for row in desc]) for j in range(5)))
            node_out.append(torch.stack(full))
            node_state.append((times.new_tensor(saved_time), matrix, short, long, ws, wl))
            node_cache.append(cache)
        descriptions.append(tuple(torch.stack([row[j] for row in node_desc]) for j in range(5)))
        outputs.append(torch.stack(node_out))
        states.append(tuple(torch.stack([row[j] for row in node_state]) for j in range(6)))
        caches.append(node_cache)
    return (tuple(torch.stack([node[j] for node in descriptions]) for j in range(5)),
            torch.stack(outputs), tuple(torch.stack([node[j] for node in states]) for j in range(6)), caches)


def fixture(shared=False, settle=True, **clock):
    torch.manual_seed(417)
    module = SSDMemory(3, 64, [0, 0, 1] if shared else None, 3, 2,
                       settle=settle, chunk=3, **clock).double()
    with torch.no_grad():
        for parameter in (module.a_log.value, module.dt_bias.value, module.skip.value, module.rho.value):
            parameter.add_(torch.randn_like(parameter) * .2)
    u = torch.randn(3, 2, 7, 64, dtype=torch.float64, requires_grad=True)
    mass = torch.tensor([[[.2, 0, .8, .1, 0, .5, 0], [.2, .6, 0, 0, .7, 0, 0]],
                         [[0, .5, .3, -.2, .4, 0, 0], [.2, .4, .1, 0, 0, .8, 0]],
                         [[.1, .3, 0, .2, 1.4, 0, 0], [0, 0, .6, 0, 0, 0, 0]]],
                        dtype=torch.float64, requires_grad=True)
    times = torch.tensor([5, 6, 8, 9, 12, 14, 18]).expand(3, 2, 7)
    owners = torch.tensor([2, 0, 1])
    control = torch.tensor([[[1, 0, 0, .2, 0, .8, 0], [0, 1, 0, 0, 1, 0, 0]],
                            [[0, 0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
                            [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]], dtype=u.dtype)

    def random(*shape):
        return (.1 * torch.randn(*shape, dtype=u.dtype)).requires_grad_()

    initial = SSDState(torch.full((3, 2), 3, dtype=torch.int64), random(3, 2, 2, 32, 32),
                       random(3, 2, 16), random(3, 2, 16),
                       torch.full((3, 2), .3, dtype=u.dtype, requires_grad=True),
                       torch.full((3, 2), 1.2, dtype=u.dtype, requires_grad=True),
                       ConvState(torch.tensor([0, 1, 3]).expand(3, 2, 3), random(3, 2, 3, 128)))
    return module, u, mass, times, owners, initial, control


def state_values(state):
    return (state.times, state.matrix, state.short, state.long, state.short_weight, state.long_weight)


def decode_cache(state):
    return [[{int(t): v for t, v in zip(ts, vs) if int(t) >= 0}
             for ts, vs in zip(row_t, row_v)] for row_t, row_v in zip(state.times, state.values)]


def assert_result(actual, expected, atol=2e-11, rtol=2e-11):
    ad, ag, ast, acache = actual
    ed, eg, est, ecache = expected
    for x, y in zip((*ad, ag, *ast), (*ed, eg, *est)):
        torch.testing.assert_close(x, y, rtol=rtol, atol=atol)
    for arow, erow in zip(acache, ecache):
        for a, e in zip(arow, erow):
            assert sorted(a) == sorted(e)
            for t in e:
                torch.testing.assert_close(a[t], e[t], rtol=rtol, atol=atol)


def loss(result):
    desc, full, state, cache = result
    tensors = (*desc, full, *state[1:], *(v for row in cache for sample in row for v in sample.values()))
    return sum(x.square().sum() for x in tensors)


def inputs(module, u, mass, state):
    return [u, mass, *state_values(state)[1:], state.conv.values, *module.parameters()]


def evaluated(module, u, mass, times, owners, initial, control):
    description, context, state = module.prepare(u, mass, times, owners, initial)
    return description, module.finish(context, control), state_values(state), decode_cache(state.conv)


@pytest.mark.parametrize("shared,settle", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize("summary_backend", ["block", "prefix"])
@pytest.mark.parametrize("scan_backend", ["block", "doubling"])
def test_full_binding_matches_independent_values_and_all_gradients(shared, settle, summary_backend, scan_backend):
    args = fixture(shared, settle, summary_backend=summary_backend, scan_backend=scan_backend)
    module, u, mass, _, _, initial, _ = args
    actual, expected = evaluated(*args), literal(*args)
    assert_result(actual, expected)
    roots = inputs(module, u, mass, initial)
    actual_grad = torch.autograd.grad(loss(actual), roots, retain_graph=True)
    expected_grad = torch.autograd.grad(loss(expected), roots)
    for x, y in zip(actual_grad, expected_grad):
        torch.testing.assert_close(x, y, atol=2e-9, rtol=2e-10)


@pytest.mark.parametrize("summary_backend", ["block", "prefix"])
@pytest.mark.parametrize("scan_backend", ["block", "incremental", "doubling"])
def test_full_binding_split_continuation_values_raw_state_and_all_gradients(summary_backend, scan_backend):
    module, u, mass, times, owners, initial, control = fixture(True, False, summary_backend=summary_backend,
                                                              scan_backend=scan_backend)
    whole = evaluated(module, u, mass, times, owners, initial, control)
    left, ctx, state = module.prepare(u[:, :, :1], mass[:, :, :1], times[:, :, :1], owners, initial)
    left_g = module.finish(ctx, control[:, :, :1])
    right, ctx, state = module.prepare(u[:, :, 1:], mass[:, :, 1:], times[:, :, 1:], owners, state)
    split = (tuple(torch.cat((a, b), 2) for a, b in zip(left, right)),
             torch.cat((left_g, module.finish(ctx, control[:, :, 1:])), 2),
             state_values(state), decode_cache(state.conv))
    assert_result(split, whole)
    roots = inputs(module, u, mass, initial)
    whole_grad = torch.autograd.grad(loss(whole), roots, retain_graph=True)
    split_grad = torch.autograd.grad(loss(split), roots)
    for x, y in zip(split_grad, whole_grad):
        torch.testing.assert_close(x, y, atol=2e-9, rtol=2e-10)


def test_full_uses_one_selected_pack_and_skips_output_work_without_control(monkeypatch):
    import tide.ssd_memory as implementation
    module, u, mass, times, owners, initial, control = fixture()
    _, context, state = module.prepare(u, mass, times, owners, initial)
    calls = []
    original_select = implementation.select_queries

    def select(*args):
        calls.append("select")
        return original_select(*args)

    monkeypatch.setattr(implementation, "select_queries", select)
    handles = [layer.register_forward_pre_hook(lambda _, args: calls.append(tuple(args[0].shape)))
               for layer in (module.output, module.ffn)]
    output = module.finish(context, control)
    assert calls == ["select", (2, 2, 3, 64), (2, 2, 3, 64)]
    assert torch.count_nonzero(output[control <= 0]) == 0
    calls.clear()
    assert torch.count_nonzero(module.finish(context, torch.zeros_like(control))) == 0
    assert calls == ["select"]
    # Positive unselected receptions still advanced the third node's state.
    assert state.times[2].tolist() == [12, 8]
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("summary_backend,short_decay", [("block", None), ("prefix", None), ("prefix", 0.)])
@pytest.mark.parametrize("scan_backend", ["block", "incremental", "doubling"])
def test_idle_and_empty_windows_preserve_raw_state_with_large_logical_clock(summary_backend, short_decay, scan_backend):
    module, u, mass, times, owners, initial, control = fixture(summary_backend=summary_backend, short_decay=short_decay,
                                                              scan_backend=scan_backend)
    offset = 2**55
    initial.times = initial.times + offset
    initial.conv = ConvState(initial.conv.times + offset, initial.conv.values)
    times = times + offset + 1_000_000
    description, context, idle = module.prepare(u, torch.zeros_like(mass), times, owners, initial)
    for got, want in zip(state_values(idle), state_values(initial)):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    for row, expected in zip(decode_cache(idle.conv), decode_cache(initial.conv)):
        for got, want in zip(row, expected):
            assert got.keys() == want.keys()
            for t in got:
                torch.testing.assert_close(got[t], want[t], atol=0, rtol=0)
    if module.short_decay:
        assert torch.count_nonzero(description[1]) == 0
        assert torch.count_nonzero(description[3]) == 0
    assert all(torch.isfinite(x).all() for x in description)
    empty, context, unchanged = module.prepare(u[:, :, :0], mass[:, :, :0], times[:, :, :0], owners, initial)
    assert unchanged is initial and all(x.shape[2] == 0 for x in empty)
    assert module.finish(context, control[:, :, :0]).shape == u[:, :, :0].shape
    # A real reception after the all-idle gap must resume the raw state, not
    # an effective decayed summary accidentally committed by that idle window.
    actual = evaluated(module, u, mass, times+128, owners, idle, control)
    expected = literal(module, u, mass, times+128, owners, initial, control)
    assert_result(actual, expected)
    roots = inputs(module, u, mass, initial)
    observed = torch.autograd.grad(loss(actual), roots, retain_graph=True, allow_unused=True)
    reference = torch.autograd.grad(loss(expected), roots, allow_unused=True)
    for root, x, y in zip(roots, observed, reference):
        if y is None:
            # The dictionary oracle never reads the now-expired convolution
            # cache; rectangular gather can retain a connected zero VJP.
            assert root is initial.conv.values
            assert x is None or torch.count_nonzero(x) == 0
        else:
            torch.testing.assert_close(x, y, atol=2e-9, rtol=2e-10)


@pytest.mark.parametrize("scan_backend,length", [("block", 5), ("doubling", 5), ("incremental", 1)])
def test_fp32_training_from_zero_matches_literal_for_three_optimizer_steps(scan_backend, length):
    torch.manual_seed(91)
    actual_module = SSDMemory(1, 32, None, 2, 1, scan_backend=scan_backend)
    expected_module = copy.deepcopy(actual_module)
    actual_optimizer = torch.optim.SGD(actual_module.parameters(), lr=.001, momentum=.8)
    expected_optimizer = torch.optim.SGD(expected_module.parameters(), lr=.001, momentum=.8)
    u = torch.randn(1, 2, 5, 32)
    mass = torch.tensor([[[.4, 0, .7, .1, 0], [.3, .5, 0, .6, 0]]])
    times = torch.tensor([1, 2, 4, 5, 8]).expand(1, 2, 5)
    u, mass, times = u[:, :, :length], mass[:, :, :length], times[:, :, :length]
    owners, control = torch.tensor([0]), (mass > 0).to(u.dtype)
    initial = SSDState(torch.zeros(1, 2, dtype=torch.int64), torch.zeros(1, 2, 1, 32, 32),
                       torch.zeros(1, 2, 8), torch.zeros(1, 2, 8), torch.zeros(1, 2), torch.zeros(1, 2),
                       ConvState(torch.empty(1, 2, 0, dtype=torch.int64), torch.empty(1, 2, 0, 96)))
    for _ in range(3):
        actual = evaluated(actual_module, u, mass, times, owners, None, control)
        expected = literal(expected_module, u, mass, times, owners, initial, control)
        assert_result(actual, expected, atol=2e-5, rtol=2e-5)
        actual_optimizer.zero_grad()
        expected_optimizer.zero_grad()
        (loss(actual) / u.numel()).backward()
        (loss(expected) / u.numel()).backward()
        for x, y in zip(actual_module.parameters(), expected_module.parameters()):
            torch.testing.assert_close(x.grad, y.grad, atol=3e-5, rtol=3e-5)
        actual_optimizer.step()
        expected_optimizer.step()
        for x, y in zip(actual_module.parameters(), expected_module.parameters()):
            torch.testing.assert_close(x, y, atol=2e-7, rtol=2e-7)
