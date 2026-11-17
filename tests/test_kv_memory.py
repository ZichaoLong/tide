"""Independent KV/sum formulas on original clocks and sparse Full queries."""
import math

import pytest
import torch
from torch.nn import functional as F

from tide.kv import KVMemory, KVState


def matrix(layer, owner):
    weight = layer.weight[layer.groups[owner]]
    return weight + layer.left[owner] @ layer.right[owner] if layer.shared else weight


def normal(x):
    return x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-5)


def decoded(state):
    return [[{int(state.times[n, b, i]): tuple(field[n, b, i] for field in
              (state.weight, state.keys, state.values, state.features))
              for i in range(state.times.shape[2]) if state.weight[n, b, i] > 0}
             for b in range(state.times.shape[1])] for n in range(state.times.shape[0])]


def formula(module, u, mass, times, owners, initial, control, queries=None):
    """Direct window sums and relative-RoPE dot products, without KV helpers.

    The optional query subset bounds long-window reference work. State and
    projections still include every positive reception, selected or otherwise.
    """
    queries = list(range(u.shape[2])) if queries is None else queries
    descriptions, full_nodes, caches = [], [], []
    frequencies = u.new_tensor([10000**(-2*i/32) for i in range(16)])
    for node in range(u.shape[0]):
        owner = int(owners[node])
        normalized = normal(u[node])
        keys, values = (normalized @ matrix(module.kv, owner).T).chunk(2, -1)
        features = normalized @ matrix(module.summary, owner).T
        if initial is None:
            bank = (times[node], mass[node], keys, values, features)
        else:
            bank = tuple(torch.cat((old[node], new), 1) for old, new in zip(
                (initial.times, initial.weight, initial.keys, initial.values, initial.features),
                (times[node], mass[node], keys, values, features)))
        bt, bw, bk, bv, bp = bank
        delta = times[node, :, queries, None] - bt[:, None, :]
        visible = (delta >= 0) & (delta < module.window) & (bw[:, None, :] > 0)
        long_weight = torch.where(visible, bw[:, None, :], 0)
        short_weight = long_weight * torch.exp(delta.clamp_min(0).to(u.dtype)
                                              * (-math.log(2)/(8*module.period)))
        ws, wl = short_weight.sum(-1), long_weight.sum(-1)
        ms = short_weight @ bp / torch.where(ws > 1, ws, 1)[..., None]
        ml = long_weight @ bp / torch.where(wl > 1, wl, 1)[..., None]
        descriptions.append((features[:, queries], ms, ml, ws, wl))
        batch_full, batch_caches = [], []
        for batch in range(u.shape[1]):
            selected = (control[node, batch] > 0).nonzero().flatten().tolist()
            outputs = []
            for i in selected:
                current = u[node, batch, i]
                keep = (bt[batch] <= times[node, batch, i]) & ((times[node, batch, i]-bt[batch]) < module.window)
                keep = keep & (bw[batch] > 0)
                weight = bw[batch, keep]
                assert weight.numel()  # Selected fixture rows have positive reception.
                key = bk[batch, keep].reshape(-1, module.d//32, 16, 2)
                relative = (bt[batch, keep]-times[node, batch, i]).to(u.dtype) / module.period
                phase = relative[:, None] * frequencies
                real = key[..., 0]*phase.cos()[:, None] - key[..., 1]*phase.sin()[:, None]
                imag = key[..., 0]*phase.sin()[:, None] + key[..., 1]*phase.cos()[:, None]
                rotated = torch.stack((real, imag), -1).flatten(-2)
                q = torch.stack([matrix(head, owner) @ normal(current) for head in module.query.heads])
                logits = (rotated*q).sum(-1).T / math.sqrt(32)
                terms = weight[None] * torch.exp(logits-logits.amax(-1, keepdim=True))
                probability = terms / terms.sum(-1, keepdim=True)
                read = torch.einsum("hk,khd->hd", probability, bv[batch, keep].reshape(-1, module.d//32, 32))
                amount = weight.sum()
                read = read.flatten() * torch.where(amount >= 1, 1, amount)
                x = current + matrix(module.output, owner) @ read
                z = normal(x)
                hidden = F.silu(matrix(module.ffn.gate, owner) @ z) * (matrix(module.ffn.up, owner) @ z)
                outputs.append(x + matrix(module.ffn.down, owner) @ hidden)
            result = u.new_zeros(u.shape[2:])
            if selected:
                result = result.index_put((torch.tensor(selected),), torch.stack(outputs))
            batch_full.append(result)
            positive = bw[batch] > 0
            last = bt[batch, positive].max() if positive.any() else bt.new_tensor(0)
            valid = positive & ((last-bt[batch]) < module.window)
            batch_caches.append({int(bt[batch, i]): tuple(field[batch, i] for field in bank[1:])
                                 for i in valid.nonzero().flatten().tolist()})
        full_nodes.append(torch.stack(batch_full))
        caches.append(batch_caches)
    return tuple(torch.stack([row[i] for row in descriptions]) for i in range(5)), torch.stack(full_nodes), caches


def actual(module, u, mass, times, owners, initial, control, queries=None):
    desc, context, state = module.prepare(u, mass, times, owners, initial)
    if queries is not None:
        desc = tuple(x[:, :, queries] for x in desc)
    return desc, module.finish(context, control), decoded(state)


def assert_result(got, want, dtype=torch.float64):
    atol, rtol = (3e-11, 3e-11) if dtype == torch.float64 else (8e-5, 8e-5)
    for a, b in zip((*got[0], got[1]), (*want[0], want[1])):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    for grow, wrow in zip(got[2], want[2]):
        for gc, wc in zip(grow, wrow):
            assert gc.keys() == wc.keys()
            for time in wc:
                for a, b in zip(gc[time], wc[time]):
                    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def loss(result):
    desc, full, caches = result
    tensors = (*desc, full, *(x for row in caches for cache in row for entry in cache.values() for x in entry))
    return sum(x.square().sum() for x in tensors)


def fixture(dtype=torch.float64, shared=True):
    torch.manual_seed(651)
    module = KVMemory(3, 64, [0, 1, 0] if shared else None, 4, 4, chunk=3).to(dtype)
    u = torch.randn(2, 2, 7, 64, dtype=dtype, requires_grad=True)
    weights = torch.full((2, 2, 7), .17, dtype=dtype, requires_grad=True)
    present = torch.tensor([[[1, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0]],
                            [[1, 1, 0, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0]]], dtype=torch.bool)
    mass = torch.where(present, weights, 0)
    times = torch.tensor([9, 13, 21, 25, 29, 37, 41]).expand(2, 2, 7)
    owners = torch.tensor([2, 0])
    control = torch.tensor([[[1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0]],
                            [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]], dtype=dtype)
    initial = KVState(torch.tensor([1, 5]).expand(2, 2, 2),
                      torch.full((2, 2, 2), .11, dtype=dtype, requires_grad=True),
                      torch.randn(2, 2, 2, 64, dtype=dtype, requires_grad=True),
                      torch.randn(2, 2, 2, 64, dtype=dtype, requires_grad=True),
                      torch.randn(2, 2, 2, 16, dtype=dtype, requires_grad=True))
    roots = [u, weights, initial.weight, initial.keys, initial.values, initial.features, *module.parameters()]
    return (module, u, mass, times, owners, initial, control), roots


@pytest.mark.parametrize("dtype,shared", [(torch.float64, False), (torch.float64, True), (torch.float32, True)])
def test_independent_read_plus_full_raw_state_and_all_gradients(dtype, shared):
    args, roots = fixture(dtype, shared)
    got, want = actual(*args), formula(*args)
    assert_result(got, want, dtype)
    ga = torch.autograd.grad(loss(got), roots, retain_graph=True)
    gb = torch.autograd.grad(loss(want), roots)
    atol, rtol = (3e-9, 3e-10) if dtype == torch.float64 else (8e-3, 2e-4)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def test_sparse_continuation_values_raw_state_and_gradients():
    args, roots = fixture()
    module, u, mass, times, owners, initial, control = args
    whole = actual(*args)
    left, context, saved = module.prepare(u[:, :, :3], mass[:, :, :3], times[:, :, :3], owners, initial)
    lg = module.finish(context, control[:, :, :3])
    right, context, terminal = module.prepare(u[:, :, 3:], mass[:, :, 3:], times[:, :, 3:], owners, saved)
    split = tuple(torch.cat((a, b), 2) for a, b in zip(left, right)), torch.cat((lg, module.finish(context, control[:, :, 3:])), 2), decoded(terminal)
    assert_result(split, whole)
    ga = torch.autograd.grad(loss(split), roots, retain_graph=True)
    gb = torch.autograd.grad(loss(whole), roots)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-10)


@pytest.mark.parametrize("settle,length", [(True, 149), (False, 535)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_long_original_time_window_against_selected_formula(settle, length, dtype):
    torch.manual_seed(405)
    module = KVMemory(1, 64, None, 4, 4, settle=settle, chunk=29).to(dtype)
    u = torch.randn(1, 1, length, 64, dtype=dtype)
    mass = torch.full((1, 1, length), .07, dtype=dtype)
    mass[:, :, 5::9] = 0
    step = 4 if settle else 1
    times = (1 + step*torch.arange(length)).reshape(1, 1, length)
    edge = 128 if settle else 512
    queries = [0, edge-1, edge, edge+1, length-2, length-1]
    control = torch.zeros_like(mass)
    control[:, :, queries] = 1
    owners = torch.tensor([0])
    with torch.no_grad():
        got = actual(module, u, mass, times, owners, None, control, queries)
        want = formula(module, u, mass, times, owners, None, control, queries)
        cut = edge+1
        left, context, saved = module.prepare(u[:, :, :cut], mass[:, :, :cut], times[:, :, :cut], owners)
        lg = module.finish(context, control[:, :, :cut])
        right, context, terminal = module.prepare(u[:, :, cut:], mass[:, :, cut:], times[:, :, cut:], owners, saved)
        split_desc = tuple(torch.cat((a, b), 2)[:, :, queries] for a, b in zip(left, right))
        split_full = torch.cat((lg, module.finish(context, control[:, :, cut:])), 2)
    assert_result(got, want, dtype)
    assert_result((split_desc, split_full, decoded(terminal)), got, dtype)
    assert times[0, 0, edge] - times[0, 0, 0] == module.window
    assert len(got[2][0][0]) <= module.slots


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_huge_common_int64_translation_preserves_reads_and_raw_entries(dtype):
    args, _ = fixture(dtype)
    module, u, mass, times, owners, initial, control = args
    offset = 2**55 + 19
    shifted = KVState(initial.times + offset, initial.weight, initial.keys, initial.values, initial.features)
    with torch.no_grad():
        base = actual(*args)
        translated = actual(module, u, mass, times+offset, owners, shifted, control)
    normalized_cache = [[{t-offset: entry for t, entry in cache.items()} for cache in row] for row in translated[2]]
    assert_result((translated[0], translated[1], normalized_cache), base, dtype)


def test_only_selected_full_and_idle_raw_state_retention(monkeypatch):
    import tide.kv as implementation
    args, _ = fixture()
    module, u, mass, times, owners, initial, control = args
    module.chunk = u.shape[2]  # The short path retains one selected pack.
    _, context, saved = module.prepare(u, mass, times, owners, initial)
    calls, original = [], implementation.select_queries

    def select(*args):
        calls.append("select")
        return original(*args)

    monkeypatch.setattr(implementation, "select_queries", select)
    handles = [layer.register_forward_pre_hook(lambda _, args: calls.append(tuple(args[0].shape)))
               for layer in (module.query, module.output, module.ffn)]
    full = module.finish(context, control)
    assert calls == ["select", (2, 2, 3, 64), (2, 2, 3, 64), (2, 2, 3, 64)]
    assert torch.count_nonzero(full[control <= 0]) == 0
    calls.clear()
    assert torch.count_nonzero(module.finish(context, torch.zeros_like(control))) == 0
    assert calls == ["select"]
    for handle in handles:
        handle.remove()
    description, _, idle = module.prepare(u, torch.zeros_like(mass), times+10000, owners, saved)
    for got, want in zip(decoded(idle), decoded(saved)):
        for a, b in zip(got, want):
            assert a.keys() == b.keys()
            for time in a:
                for x, y in zip(a[time], b[time]):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert all(torch.count_nonzero(x) == 0 for x in description[1:])


def test_zero_reception_no_write_boundary_has_same_continuation_vjp():
    torch.manual_seed(981)
    module = KVMemory(1, 64, None, 4, 4, chunk=3).double()
    u = torch.randn(1, 1, 4, 64, dtype=torch.float64)
    mass = torch.tensor([[[.2, 0, .3, .4]]], dtype=u.dtype, requires_grad=True)
    times, owners = torch.tensor([[[1, 5, 9, 13]]]), torch.tensor([0])
    control = torch.tensor([[[0, 0, 1, 1]]], dtype=u.dtype)
    whole = actual(module, u, mass, times, owners, None, control)
    _, _, saved = module.prepare(u[:, :, :2], mass[:, :, :2], times[:, :, :2], owners)
    desc, context, _ = module.prepare(u[:, :, 2:], mass[:, :, 2:], times[:, :, 2:], owners, saved)
    whole_loss = sum(x[:, :, 2:].square().sum() for x in whole[0]) + whole[1].square().sum()
    split_loss = sum(x.square().sum() for x in desc) + module.finish(context, control[:, :, 2:]).square().sum()
    ga = torch.autograd.grad(whole_loss, mass, retain_graph=True)[0]
    gb = torch.autograd.grad(split_loss, mass)[0]
    torch.testing.assert_close(ga, gb, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("settle", [True, False])
def test_long_sparse_full_uses_bounded_original_position_key_tiles(settle, monkeypatch):
    import tide.kv as implementation
    torch.manual_seed(291)
    module = KVMemory(2, 64, None, 4, 4, settle=settle, chunk=64).double()
    length = module.slots + 3*module.chunk + 5
    u = torch.randn(2, 2, length, 64, dtype=torch.float64)
    positions = torch.arange(length)
    mass = ((positions % 5) != 2).to(u.dtype).expand(2, 2, length).clone() * .13
    times = (2**55 + 1 + positions*(4 if settle else 1)).expand(2, 2, length)
    control = torch.zeros_like(mass)
    points = [0, 63, 64, module.slots-1, module.slots, module.slots+64, length-1]
    control[0, 0, points] = 1
    control[0, 1, 65] = 1
    control[1, 1, length-2] = 1
    mass = torch.where(control > 0, .13, mass)
    queries = control.any(dim=(0, 1)).nonzero().flatten().tolist()
    original, calls, projections = implementation.weighted_attention, [], []

    def attention(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, kwargs["mask"].shape))
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(implementation, "weighted_attention", attention)
    handles = [layer.register_forward_pre_hook(lambda _, args: projections.append(tuple(args[0].shape)))
               for layer in (module.query, module.output, module.ffn)]
    with torch.no_grad():
        got = actual(module, u, mass, times, torch.tensor([0, 1]), None, control, queries)
        want = formula(module, u, mass, times, torch.tensor([0, 1]), None, control, queries)
    assert_result(got, want)
    expected = []
    for start in range(0, length, module.chunk):
        end = min(length, start+module.chunk)
        active = control[:, :, start:end] > 0
        if active.any():
            expected.append((int(active.any(dim=(1, 2)).sum()), int(active.sum(-1).max()),
                             end-max(0, start-module.slots+1)))
    assert len(calls) == len(expected) and len(projections) == 3*len(expected)
    for i, ((q, k, mask), (nodes, query_count, keys)) in enumerate(zip(calls, expected)):
        assert q == (nodes, 2, 2, query_count, 32) and k == (nodes, 2, 2, keys, 32)
        assert mask == (nodes, 2, 1, query_count, keys) and keys <= module.slots+module.chunk-1
        assert projections[3*i:3*i+3] == [(nodes, 2, query_count, 64)]*3
    for handle in handles:
        handle.remove()


def test_long_tiled_full_independent_gradients_and_eviction_continuation():
    torch.manual_seed(317)
    module = KVMemory(1, 64, [0], 4, 4, chunk=17).double()
    length, cut, queries = 147, 129, [17, 127, 128, 146]
    u = torch.randn(1, 1, length, 64, dtype=torch.float64, requires_grad=True)
    positive = torch.full((1, 1, length), .13, dtype=u.dtype, requires_grad=True)
    mass = torch.where((torch.arange(length) % 9 != 3)[None, None], positive, 0)
    times, owners = (1+4*torch.arange(length))[None, None], torch.tensor([0])
    control = torch.zeros_like(mass)
    control[:, :, queries] = 1
    got = actual(module, u, mass, times, owners, None, control, queries)
    want = formula(module, u, mass, times, owners, None, control, queries)
    left, context, saved = module.prepare(u[:, :, :cut], mass[:, :, :cut], times[:, :, :cut], owners)
    lg = module.finish(context, control[:, :, :cut])
    right, context, terminal = module.prepare(u[:, :, cut:], mass[:, :, cut:], times[:, :, cut:], owners, saved)
    split = (tuple(torch.cat((a, b), 2)[:, :, queries] for a, b in zip(left, right)),
             torch.cat((lg, module.finish(context, control[:, :, cut:])), 2), decoded(terminal))
    assert_result(got, want)
    assert_result(split, want)
    roots = [u, positive, *module.parameters()]
    expected = torch.autograd.grad(loss(want), roots, retain_graph=True)
    for result in (got, split):
        for a, b in zip(torch.autograd.grad(loss(result), roots, retain_graph=True), expected):
            torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-10)
