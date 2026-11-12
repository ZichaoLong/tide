"""Architecture §2: aligned/ragged physical atoms and their independent roots.

The literal per-fiber formula does not use column packing or the mean VJP.
Repeated atoms, zero fibers and independent equal-valued inputs are distinct.
"""
import pytest
import torch
from torch.nn import functional as F
from tidegraph import Graph, Node, Region
from tidegraph.source_class import SourceClasses
from tidegraph.weighted import KVConfig, bind
from tidegraph.weighted.message_columns import MessageColumns, aggregate_columns


@pytest.mark.parametrize("aligned", [True, False])
@pytest.mark.parametrize("zero", [True, False])
def test_physical_multiplicity_formula_and_disconnected_roots(dtype, aligned, zero):
    graph = Graph((Node(0),), (), (Region(1),), (0, 0), (0,))
    binding = bind(graph, memories=KVConfig(width=2, summary_width=2),
                   source_classes=SourceClasses((), (0, 1)), alpha_agg=.2, dtype=dtype)
    program = binding.model.nodes[0].aggregate_program
    rows = ([0, 2, 4], [0, 2, 4] if aligned else [1, 2, 4])
    payloads = [MessageColumns(torch.tensor(r),
        torch.tensor([0., -.1, 0.] if zero else [.2, .7, .5], dtype=dtype, requires_grad=True),
        torch.tensor([[.2, -.4], [.7, .1], [-.5, .6]], dtype=dtype, requires_grad=True)) for r in rows]
    # One actual source occurs twice; the other has independent roots even
    # when its values and row domain happen to be identical.
    sources = [(payloads[0], 0), (payloads[0], 0), (payloads[1], 1)]
    actual_rows, mass, value = aggregate_columns(program, sources)
    expected_rows = sorted(set(rows[0]+rows[1]))
    assert actual_rows.tolist() == expected_rows
    expected_mass, expected_value = [], []
    for row in expected_rows:
        weights, terms = [], []
        for payload, label in sources:
            positions = payload.rows.tolist()
            if row not in positions:
                continue
            i = positions.index(row)
            z, w = payload.content[i], payload.strength[i].clamp_min(0)
            transform = program.transforms[str(label)]
            a, b = transform.matrices()
            normalized = z/torch.sqrt(z.square().mean()+program.epsilon)
            term = z+program.alpha*(b@F.silu(a@normalized+transform.embedding))
            weights.append(w);terms.append(term)
        total = sum(weights)
        expected_mass.append(total)
        expected_value.append(sum(w*z for w, z in zip(weights, terms))/total
                              if total > 0 else value.new_zeros(2))
    expected_mass, expected_value = torch.stack(expected_mass), torch.stack(expected_value)
    torch.testing.assert_close(mass, expected_mass)
    torch.testing.assert_close(value, expected_value)
    roots = [x for p in payloads for x in (p.strength, p.content)]+list(program.parameters())
    bars = torch.linspace(-.3, .9, value.numel(), dtype=dtype).reshape(value.shape)
    actual_loss = (value*bars).sum()+.23*mass.sum()
    expected_loss = (expected_value*bars).sum()+.23*expected_mass.sum()
    actual = torch.autograd.grad(actual_loss, roots, allow_unused=True)
    expected = torch.autograd.grad(expected_loss, roots, allow_unused=True)
    for a, b in zip(actual, expected):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b)
    if zero:
        assert all(actual[i] is None for i in (1, 3, *range(4, len(roots))))
