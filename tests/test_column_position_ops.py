"""KV §2 rotary identity: literal relative formulas, roots and finite-domain guards."""
from dataclasses import replace
import math
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.weighted.column_layout import SequenceLayout
from tidegraph.weighted.column_position_ops import PositionQuery, position_logits_many, rebase_positions
from tidegraph.weighted.packing_observation import observe
from isolated_cases import vjp


def literal(r):
    """Scalar relative coordinate rotations, no shared origin or candidate helper."""
    outputs = []
    for b, ((qa, qz), (ka, kz)) in enumerate(zip(r.query_layout.spans, r.bank_layout.spans)):
        heads = []
        for h, frequencies in enumerate(r.frequencies):
            rows = []
            for qi in range(qz-qa):
                values = []
                for ki in range(kz-ka):
                    delta = r.query.new_tensor(int(r.bank_layout.times[ka+ki])-int(r.query_layout.times[qa+qi]))/r.scale
                    q, k = r.query[b, h, qi], r.key[b, h, ki]
                    first = q.numel()-2*len(frequencies)
                    value = (q[:first]*k[:first]).sum()
                    for j, frequency in enumerate(frequencies):
                        phase = delta*frequency
                        a, z = k[first+2*j], k[first+2*j+1]
                        value = value+q[first+2*j]*(a*phase.cos()-z*phase.sin())
                        value = value+q[first+2*j+1]*(a*phase.sin()+z*phase.cos())
                    value = value/math.sqrt(q.numel())
                    if r.slopes:
                        value = value+r.slopes[h]*delta
                    values.append(value)
                rows.append(torch.stack(values))
            heads.append(torch.stack(rows))
        outputs.append(torch.stack(heads))
    return torch.stack(outputs)


def requests(dtype, *, frequency=0.17, origin=2**60):
    ql = SequenceLayout(torch.tensor([0, 3, 6]), torch.tensor([origin+t for t in (11, 14, 23, 9, 15, 25)]))
    kl = SequenceLayout(torch.tensor([0, 4, 8]), torch.tensor([origin+t for t in (1, 7, 13, 21, 2, 8, 17, 24)]))
    qtimes, ktimes = ql.times.reshape(2, 3), kl.times.reshape(2, 4)
    generator = torch.Generator().manual_seed(851)
    result, roots = [], {}
    for i in range(3):
        q = torch.randn((2, 2, 3, 6), generator=generator, dtype=dtype).requires_grad_()
        k = torch.randn((2, 2, 4, 6), generator=generator, dtype=dtype).requires_grad_()
        if i == 1:
            q = result[0].query
        else:
            roots[f"q{i}"] = q
        roots[f"k{i}"] = k
        result.append(PositionQuery(q, k, qtimes[:, :, None]-ktimes[:, None], ql, kl,
                                   ((frequency,), (-frequency/2, .13)), 2.7, (.03, .04)))
    return result, roots


def test_rebased_nodes_match_literal_relative_fields_and_each_root(dtype):
    rows, roots = requests(dtype)
    shapes = []
    with observe() as work, torch.autograd.graph.saved_tensors_hooks(
            lambda t: (shapes.append(tuple(t.shape)), t)[1], lambda t: t):
        actual = position_logits_many(rows)
    expected = [literal(r) for r in rows]
    equivalent(actual, expected)
    assert work.metrics()["packing/bucket/kv_rope_rebased/calls"] == 3
    assert (2, 3, 4, 6) not in shapes  # No per-query rotated key bank.
    bar = torch.linspace(.2, .8, actual[0].numel(), dtype=dtype).reshape_as(actual[0])
    for selected in ([0], [0, 1]):
        for zero in (False, True):
            a = vjp(sum((actual[i]*bar).sum() for i in selected), roots, zero=zero)
            e = vjp(sum((expected[i]*bar).sum() for i in selected), roots, zero=zero)
            equivalent(a, e)
            assert a["q2"] is None and a["k2"] is None
    # saved_tensors_hooks intentionally replace PyTorch's normal saved-version
    # mechanism; exercise mutation guards in a fresh ordinary graph.
    actual = position_logits_many(rows)
    with torch.no_grad():
        roots["q0"].add_(.01)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        actual[0].sum().backward()


@pytest.mark.parametrize("reason", ["integer_span", "phase_span"])
def test_outside_finite_domain_keeps_literal_relative_path(dtype, reason):
    rows, roots = requests(dtype, origin=0, frequency=1.e6 if reason == "phase_span" else .17)
    if reason == "integer_span":
        limit = 2**(24 if dtype == torch.float32 else 53)
        for i, r in enumerate(rows):
            layout = replace(r.query_layout, times=r.query_layout.times+limit)
            rows[i] = replace(r, query_layout=layout, age=r.age+limit)
    with observe() as work:
        actual = position_logits_many(rows)
    expected = [literal(r) for r in rows]
    equivalent(actual, expected)
    equivalent(vjp(actual[0].sum(), roots), vjp(expected[0].sum(), roots))
    assert work.metrics()["packing/bucket/kv_rope_relative_"+reason+"/calls"] == 3
    assert "packing/bucket/kv_rope_rebased/calls" not in work.metrics()


def test_common_integer_origin_avoids_absolute_clock_aliasing_and_half_is_explicit():
    base = 2**60
    ql = SequenceLayout(torch.tensor([0, 1]), torch.tensor([base+1]))
    kl = SequenceLayout(torch.tensor([0, 1]), torch.tensor([base]))
    q = torch.tensor([[[[1., 0.]]]], dtype=torch.float32, requires_grad=True)
    r = PositionQuery(q, q, torch.ones((1, 1, 1), dtype=torch.int64), ql, kl, ((1.,),), 1.)
    value = position_logits_many([r])[0]
    equivalent(value, torch.full_like(value, math.cos(1)/math.sqrt(2)))
    assert float(ql.times.to(torch.float32)[0]) == float(kl.times.to(torch.float32)[0])
    assert abs(float(value.detach())-1/math.sqrt(2)) > .3
    _, reason = rebase_positions(ql, kl, ((1.,),), 1., torch.float16)
    assert reason == "half"


def test_distance_only_uses_bank_matmul_without_rotating_key_copies(dtype):
    rows, roots = requests(dtype)
    rows = [replace(r, frequencies=((), ())) for r in rows]
    with observe() as work:
        actual = position_logits_many(rows)
    expected = [literal(r) for r in rows]
    equivalent(actual, expected)
    equivalent(vjp(actual[0].sum(), roots), vjp(expected[0].sum(), roots))
    assert work.metrics()["packing/bucket/kv_rope_content_distance/calls"] == 3


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("phase", ["rebased", "relative_phase_span"])
def test_default_position_policy_against_independent_relative_schedule(dtype, implementation, phase):
    from tidegraph import Continuation
    from tidegraph.reference import run as reference
    from tidegraph.weighted import KVConfig, RegionalConfig, bind, fully_connected, SettleColumnPlan, layered_inputs
    from tidegraph.experiment.work_observation import observe_work
    from test_settle_columns import observations, objective
    spec = fully_connected(2, 3)
    frequency = .17 if phase == "rebased" else 1.e9
    memory = KVConfig(width=2, summary_width=2, heads=1, ffn_width=3, epsilon=.3,
                      memory_read="qry", read_frequencies=(frequency,), full_frequencies=((frequency,),),
                      distance_slopes=(.03,), position_scale=5.)
    binding = bind(spec.graph, memories=memory, regions=RegionalConfig(hidden_width=3), dtype=dtype)
    spec = replace(spec, graph=binding.graph)
    x = torch.linspace(.2, .7, 12, dtype=dtype).reshape(2, 3, 2).requires_grad_()
    base = 2**54
    with observe_work(implementation) as observed:
        actual = SettleColumnPlan(binding, spec, implementation=implementation).run(x, start_position=base)
    assert observed["metrics"]["packing/bucket/kv_rope_"+phase+"/calls"] > 0
    left = Continuation(binding.graph.identity, 2, cut=base*spec.stride)
    stop = (base+3)*spec.stride
    expected = reference(binding.graph, binding.model, left,
                         layered_inputs(spec, x, start_position=base, start_occurrence=0), stop, sealed_until=stop)
    a, e = observations(actual, expected, spec)
    equivalent(a, e)
    roots = {n: p for n, p in binding.model.named_parameters() if p.requires_grad} | {"input": x}
    equivalent(vjp(objective(a), roots), vjp(objective(e), roots))
