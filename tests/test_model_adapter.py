"""A tiny deterministic decoder with a literal per-head/per-token formula anchor."""
from dataclasses import replace
import math
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.model_adapter import TinyDecoderAdapter


def data(dtype):
    x = (torch.sin(torch.arange(80, dtype=dtype)*.13).reshape(2, 5, 8)/3).requires_grad_()
    positions = torch.tensor([[3, 7, 9, 14, 20], [2, 4, -1, 12, -1]])
    occurrences = torch.tensor([[0, 1, 2, 3, 4], [0, 1, -1, 2, -1]])
    valid = occurrences >= 0
    return x, positions, occurrences, valid


def literal(model, x, positions, valid):
    w = model.weights.extra; head_width = model.width // model.query_heads
    def norm(v, scale):
        return v / (v.dot(v)/len(v) + model.epsilon).sqrt() * scale
    def rotate(v, position):
        result = []
        for i in range(0, len(v), 2):
            angle = position / model.rope_base**(i/len(v))
            result.extend((v[i]*math.cos(angle)-v[i+1]*math.sin(angle),
                           v[i]*math.sin(angle)+v[i+1]*math.cos(angle)))
        return torch.stack(result)
    rows, final = [], []
    for b in range(len(x)):
        keys, values, outputs = [], [], []
        for t in range(x.shape[1]):
            if not valid[b, t]: outputs.append(x[b, t]*0); continue
            h = x[b, t]; n = norm(h, model.norm_attention)
            q = (n @ w['attn_q']).split(head_width)
            k = (n @ w['attn_k']).split(head_width)
            keys.append(torch.stack([rotate(v, int(positions[b, t])) for v in k]))
            values.append((n @ w['attn_v']).reshape(model.kv_heads, head_width))
            if model.window: keys, values = keys[-model.window:], values[-model.window:]
            all_keys, all_values = torch.stack(keys), torch.stack(values)
            heads = []
            for i, query in enumerate(q):
                query = rotate(query, int(positions[b, t])); owner = i // (model.query_heads//model.kv_heads)
                scores = torch.stack([key[owner].dot(query) for key in keys]) / math.sqrt(head_width)
                heads.append(scores.softmax(0) @ all_values[:, owner])
            h = h + torch.cat(heads) @ w['attn_out']
            n = norm(h, model.norm_ffn); gate = n @ w['ffn_gate']
            h = h + ((gate * gate.sigmoid()) * (n @ w['ffn_up'])) @ w['ffn_down']
            outputs.append(h)
        rows.append(torch.stack(outputs)); final.append((all_keys, all_values))
    return torch.stack(rows), final


def gradients(loss, model, x):
    leaves = dict(model.named_parameters()) | {'input': x}
    return dict(zip(leaves, torch.autograd.grad(loss, tuple(leaves.values()), allow_unused=True, retain_graph=True)))


@pytest.mark.parametrize('window', [0, 3])
@pytest.mark.parametrize('layout', ['BTD', 'TBD'])
def test_model_composition_formula_cache_vjp_and_chunks(dtype, window, layout):
    m = TinyDecoderAdapter(dtype=dtype, window=window)
    x, p, ids, valid = data(dtype)
    expected, slots = literal(m, x, p, valid)
    actual, cache, stats = m(x if layout == 'BTD' else x.transpose(0, 1), p, ids, valid, layout=layout)
    if layout == 'TBD': actual = actual.transpose(0, 1)
    equivalent(expected, actual)
    equivalent([s[0] for s in slots], list(cache.keys)); equivalent([s[1] for s in slots], list(cache.values))
    assert stats['max_sequence'] == 5 and stats['valid_events'] == 8 and stats['padded_events'] == 2
    assert cache.occurrences == ((0, 1, 2, 3, 4), (0, 1, 2))
    for a, b in ((expected[0].square().sum(), actual[0].square().sum()),
                 (slots[0][0].sum(), cache.keys[0].sum()), (slots[1][1].sum()*0, cache.values[1].sum()*0)):
        equivalent(gradients(a, m, x), gradients(b, m, x))
    assert gradients(actual.sum(), m, x)['weights.read'] is None
    cut_cache = None; pieces = []
    for a, b in ((0, 2), (2, 2), (2, 3), (3, 5)):
        y, cut_cache, _ = m(x[:, a:b], p[:, a:b], ids[:, a:b], valid[:, a:b], cut_cache)
        pieces.append(y)
    equivalent(actual, torch.cat(pieces, 1)); equivalent(cache.keys, cut_cache.keys)
    equivalent(cache.values, cut_cache.values); equivalent(cache.positions, cut_cache.positions)
    equivalent(gradients(actual.square().sum(), m, x), gradients(torch.cat(pieces, 1).square().sum(), m, x))


def test_model_snapshot_detach_optimizer_and_explicit_ledger(dtype):
    m = TinyDecoderAdapter(dtype=dtype, window=3); x, p, ids, valid = data(dtype)
    _, old, _ = m(x[:, :2], p[:, :2], ids[:, :2], valid[:, :2])
    saved = tuple(v.detach().clone() for v in old.values)
    y, new, _ = m(x[:, 2:], p[:, 2:], ids[:, 2:], valid[:, 2:], old.detach())
    equivalent(saved, old.values)
    grad = gradients(y.sum(), m, x)['input']; assert torch.count_nonzero(grad[:, :2]) == 0
    with pytest.raises(ValueError, match='ledger'):
        m(x[:, 2:], p[:, 2:], ids[:, 2:]+1, valid[:, 2:], old)
    with pytest.raises(ValueError, match='increase'):
        m(x[:, 2:], p[:, 2:]*0, ids[:, 2:], valid[:, 2:], old)
    with pytest.raises(ValueError, match='cache'):
        m(x[:, :0], p[:, :0], ids[:, :0], valid[:, :0], replace(old, keys=old.keys[:1]))
    y.sum().backward(); torch.optim.AdamW(m.parameters(), lr=.001, eps=1e-5).step()
    with pytest.raises(ValueError, match='parameter update'):
        m(x[:, :0], p[:, :0], ids[:, :0], valid[:, :0], new)
    y, _, _ = m(x, p, ids, valid)  # explicit reset + prefill after update
    assert torch.isfinite(y).all()


def test_empty_samples_and_wrong_metadata(dtype):
    m = TinyDecoderAdapter(dtype=dtype); x, p, ids, valid = data(dtype)
    valid.zero_(); result, cache, stats = m(x, p, ids, valid)
    assert torch.count_nonzero(result) == 0 and stats['attention_sequence_calls'] == 0
    assert cache.occurrences == ((), ())
    for arg in ({'positions': p.float()}, {'valid': valid.long()}, {'layout': 'flat'}):
        kwargs = dict(x=x, positions=p, occurrences=ids, valid=valid); kwargs.update(arg)
        with pytest.raises(ValueError): m(**kwargs)
