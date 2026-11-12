"""Literal incidence/value/transpose witnesses, independent of old tuple groups."""
import pytest
import torch
from tidegraph.weighted.regional_packing import RegionalPacking

BIG = 2**53+3


def fixture(dtype):
    domains = {90: torch.tensor([2, 8]), 3: torch.tensor([2, 5]),
               BIG: torch.tensor([5, 8]), 99: torch.tensor([11])}
    source = {node: dict(w=torch.tensor(values, dtype=dtype, requires_grad=True),
                        vector=torch.tensor([[x, x+1] for x in values], dtype=dtype, requires_grad=True))
              for node, values in ((3, [2., 3.]), (90, [0., 4.]), (BIG, [5., 6.]), (99, [7.]))}
    return domains, source


def test_members_vary_but_real_count_and_int64_ids_define_batches(dtype):
    domains, sources = fixture(dtype)
    packing = RegionalPacking(domains)
    assert len(packing.groups) == 2
    assert [g.candidates for g in packing.groups] == [1, 2]
    one, two = [g.pack(sources) for g in packing.groups]
    assert one.nodes.tolist() == [[99]]
    assert two.nodes.tolist() == [[3, 90], [3, BIG], [90, BIG]]
    assert two.nodes.dtype == torch.int64
    literal = torch.stack([torch.stack([sources[3]['w'][0], sources[90]['w'][0]]),
                           torch.stack([sources[3]['w'][1], sources[BIG]['w'][0]]),
                           torch.stack([sources[90]['w'][1], sources[BIG]['w'][1]])])
    torch.testing.assert_close(two.fields['w'], literal)
    # A present numeric zero remains in the actual candidate set.
    assert two.fields['w'][0, 1] == 0 and two.nodes[0, 1] == 90
    packed = [g.pack(sources) for g in packing.groups]
    result = packing.unpack([p.fields['w'].square()+p.fields['vector'].sum(-1) for p in packed])
    for node in domains:
        torch.testing.assert_close(result[node], sources[node]['w'].square()+sources[node]['vector'].sum(-1))
    roots = [tensor for fields in sources.values() for tensor in fields.values()]
    expected = sum((s['w'].square()+s['vector'].sum(-1)).square().sum() for s in sources.values())
    actual = sum(value.square().sum() for value in result.values())
    for a, e in zip(torch.autograd.grad(actual, roots), torch.autograd.grad(expected, roots)):
        torch.testing.assert_close(a, e)


def test_explicit_support_distinguishes_absence_from_connected_zero(dtype):
    domains, sources = fixture(dtype)
    packing = RegionalPacking(domains)
    packed = packing.groups[1].pack(sources)
    # Only row 2 is requested. BIG is absent there; node 90 is present with zero w.
    bars = packed.pullback({'w': torch.zeros_like(packed.fields['w'])}, torch.tensor([True, False, False]))
    assert bars[BIG]['w'] is None
    assert bars[3]['w'] is not None and torch.count_nonzero(bars[3]['w']) == 0
    assert bars[90]['w'] is not None and torch.count_nonzero(bars[90]['w']) == 0
    assert all(fields['vector'] is None for fields in bars.values())
    unused = packed.pullback({'w': torch.ones_like(packed.fields['w'])}, torch.zeros(3, dtype=torch.bool))
    assert all(value is None for fields in unused.values() for value in fields.values())
    controls, support = packing.controls_pullback({99: torch.zeros(1, dtype=dtype)}, sources[99]['w'])
    assert controls[1] is None and not support[1].any()
    assert controls[0] is not None and support[0].tolist() == [True]


def test_unpublished_group_keeps_none_and_used_zero_keeps_zero(dtype):
    domains, _ = fixture(dtype)
    packing = RegionalPacking(domains)
    values = [torch.ones((len(g.rows), g.candidates), dtype=dtype, requires_grad=True) for g in packing.groups]
    outputs = packing.unpack(values)
    grads = torch.autograd.grad(outputs[99].sum()*0, values, allow_unused=True)
    assert grads[1] is None
    assert grads[0] is not None and torch.count_nonzero(grads[0]) == 0


def test_repeated_tensor_identity_accumulates_each_node_use(dtype):
    root = torch.tensor([.2, .7], dtype=dtype, requires_grad=True)
    unused = torch.tensor([.4], dtype=dtype, requires_grad=True)
    domains = {2: torch.tensor([0, 2]), 7: torch.tensor([1, 2]), 8: torch.tensor([4])}
    sources = {2: {'w': root}, 7: {'w': root}, 8: {'w': unused}}
    packing = RegionalPacking(domains)
    fields = [g.pack(sources).fields['w'] for g in packing.groups]
    result = packing.unpack(fields)
    gradient, disconnected = torch.autograd.grad(result[2].sum()+2*result[7].sum(), (root, unused), allow_unused=True)
    torch.testing.assert_close(gradient, torch.tensor([3., 3.], dtype=dtype))
    # Whole packed-field ordinary autograd has whole-column root semantics. The
    # grouped regional adapter uses explicit support for stricter source roots.
    assert disconnected is not None and torch.count_nonzero(disconnected) == 0


def test_aligned_and_empty_domains(dtype):
    source = {3: {'w': torch.tensor([1., 2.], dtype=dtype, requires_grad=True)},
              9: {'w': torch.tensor([3., 4.], dtype=dtype, requires_grad=True)}}
    packing = RegionalPacking({9: torch.tensor([0, 1]), 3: torch.tensor([0, 1])})
    packed = packing.groups[0].pack(source)
    out = packing.unpack([packed.fields['w']])
    torch.testing.assert_close(out[3], source[3]['w'])
    assert torch.autograd.grad(out[3].sum(), (source[9]['w'],), allow_unused=True)[0] is not None
    empty = RegionalPacking({3: torch.empty(0, dtype=torch.int64)})
    assert empty.unpack([], like=source[3]['w'])[3].shape == (0,)
    assert RegionalPacking({}).unpack([]) == {}
    # An otherwise aligned group must not drop an empty node's public column.
    mixed = RegionalPacking({3: torch.tensor([0, 1]), 9: torch.tensor([0, 1]),
                             12: torch.empty(0, dtype=torch.int64)})
    mixed_out = mixed.unpack([mixed.groups[0].pack(source).fields['w']])
    assert set(mixed_out) == {3, 9, 12} and mixed_out[12].shape == (0,)
    assert not mixed_out[12].requires_grad


@pytest.mark.parametrize('rows', [torch.tensor([2, 2]), torch.tensor([3, 1]), torch.tensor([-1]), torch.tensor([1.])])
def test_invalid_metadata_is_rejected(rows):
    with pytest.raises(ValueError):
        RegionalPacking({3: rows})


def test_bad_fields_and_control_metadata_are_rejected(dtype):
    domains, source = fixture(dtype)
    group = RegionalPacking(domains).groups[1]
    source[90]['vector'] = source[90]['vector'][:, :1]
    with pytest.raises(ValueError, match='metadata'):
        group.pack(source)
    with pytest.raises(ValueError, match='count'):
        RegionalPacking(domains).unpack([])
