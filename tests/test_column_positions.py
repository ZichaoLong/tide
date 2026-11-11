"""KV §2–4: relative position columns, original int64 clocks and scoped VJPs."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.compare import equivalent
from tidegraph.reference import run as reference
from tidegraph.weighted import SettleColumnPlan, KVConfig
from tidegraph.weighted.column_attention import logit_attention
from test_settle_columns import general, inputs, external, observations, objective
from isolated_cases import vjp


@pytest.mark.parametrize("read_mode", ["0", "-", "+"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_rotary_distance_expiry_use_original_clock(read_mode, implementation):
    memory = KVConfig(heads=2, memory_read="qry", window=17, position_origin=.75,
        position_scale=5, read_frequencies=(.17,), full_frequencies=((.11,), (.23,)), distance_slopes=(.04, .03))
    binding, spec = general(torch.float64, memory=memory, read_mode=read_mode)
    layout, ports = inputs(torch.float64)
    layout = replace(layout, times=layout.times+2**54)
    actual = SettleColumnPlan(binding, spec, implementation=implementation).run_ports(layout, ports)
    stop = (2**54+10)*spec.stride
    # The independent oracle scans time. Start at the empty cut just before
    # these inputs; scanning from zero would test 2**54 empty positions.
    left = Continuation(binding.graph.identity, 2, cut=2**54*spec.stride)
    expected = reference(binding.graph, binding.model, left,
                         external(spec, layout, ports), stop, sealed_until=stop)
    a, e = observations(actual, expected, spec)
    equivalent(a, e)
    roots = {n: p for n, p in binding.model.named_parameters() if p.requires_grad}
    roots.update({f"{p}.{name}": getattr(x, name) for p, x in ports.items() for name in ("strength", "content")})
    equivalent(vjp(objective(a), roots), vjp(objective(e), roots))
    equivalent(vjp(objective(a), roots, zero=True), vjp(objective(e), roots, zero=True))


def test_logit_bank_against_literal_softmax_and_directional_derivative():
    logits = torch.linspace(-.4, .5, 24, dtype=torch.float64).reshape(2, 2, 3, 2).requires_grad_()
    values = torch.linspace(-.2, .7, 24, dtype=torch.float64).reshape(2, 2, 2, 3).requires_grad_()
    weights = torch.tensor([[[.2, .5], [.3, .4], [0., .7]], [[.6, 0.], [0., 0.], [.3, .7]]], dtype=torch.float64, requires_grad=True)
    actual = logit_attention(logits, values, weights)
    rows = []
    for b in range(2):
        heads = []
        for h in range(2):
            queries = []
            for q in range(3):
                use = weights[b, q] > 0
                queries.append((torch.softmax(logits[b, h, q, use]+weights[b, q, use].log(), 0)[:, None]
                                *values[b, h, use]).sum(0) if use.any() else values.new_zeros(3))
            heads.append(torch.stack(queries))
        rows.append(torch.stack(heads))
    expected = torch.stack(rows)
    bar = torch.linspace(.1, .9, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    roots = [logits, values, weights]
    equivalent(actual, expected)
    a = torch.autograd.grad(actual, roots, bar, retain_graph=True)
    e = torch.autograd.grad(expected, roots, bar)
    equivalent(a, e)
    ds = [torch.full_like(x, .03) for x in roots];ds[-1] = ds[-1]*(weights.detach() > 0)
    h = 1e-5
    with torch.no_grad():
        finite = sum((logit_attention(*(x+h*d for x, d in zip(roots, ds)))*bar).flatten())
        finite -= sum((logit_attention(*(x-h*d for x, d in zip(roots, ds)))*bar).flatten())
    equivalent(sum((g*d).sum() for g, d in zip(a, ds)), finite/(2*h))


def test_nonempty_bound_read0_with_rotary_bank():
    from test_kv_columns import fixture
    from tidegraph.content import Content
    from tidegraph.readout import ReadInput
    from tidegraph.weighted.kv_columns_state import KVStateColumns, prepare_kv_columns
    from tidegraph.weighted.kv_columns import read_kv_columns
    node, old, mass, content, layout, roots = fixture(torch.float64)
    node.kernel.config = replace(node.kernel.config, read_frequencies=(.17,), position_scale=5)
    initial = KVStateColumns.from_states(old)
    # This API exposes column roots. Decode scalar observations from the same
    # columns; independently rooted empty per-sample tensors belong to the
    # separate record API and are not a promise of a concatenated bank.
    old = initial.states()
    roots = {n: v for n, v in roots.items() if not n[0].isdigit()}
    roots.update({"initial."+n: getattr(initial, n) for n in ("keys", "values", "summaries", "weights")})
    window = prepare_kv_columns(node.kernel, layout, mass, content, initial)
    actual = read_kv_columns(node, window, read_mode="0", initial=initial)
    rows = [node.read_program.step(node, ReadInput(old[b], int(layout.times[j]),
                Content(torch.cat((mass[j:j+1], content[j])))))
            for b, (a, z) in enumerate(layout.spans) for j in range(a, z)]
    expected = {n: torch.stack([row[n] for row in rows]) for n in rows[0]}
    equivalent(actual, expected)
    equivalent(vjp(objective(actual), roots), vjp(objective(expected), roots))
