"""0770593 architecture §2.2: direct-anchor sharing uses ordered factor roles.

Equal parameter values can hide a wrong sharing decision in forward values.
Literal source formulas and separately rooted output VJPs expose the difference.
"""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from lh_tensor_baseline import StaticKV
from tidegraph import Edge, Graph, Node, Region
from tidegraph.compare import equivalent
from tidegraph.settle import SettleGraph
from tidegraph.weighted import KVConfig, SourceBinding, bind
from tidegraph.weighted.source_binding import SourceTransform


def material(dtype, equal_values, same_slots):
    graph = Graph((Node(0), Node(1), Node(1)), (Edge(0, 1, 1), Edge(0, 2, 1)),
                  (Region(1), Region(2)), (0,), (1, 2))
    binding = bind(graph, memories=KVConfig(width=2, summary_width=2, epsilon=.2),
                   source_binding=SourceBinding("adapted", 2), source_hidden_width=2,
                   alpha_agg=.4, dtype=dtype)
    a = torch.nn.Parameter(torch.tensor([[.3, -.1], [.2, .4]], dtype=dtype))
    b = torch.nn.Parameter(a.detach().clone() if equal_values else
                           torch.tensor([[-.2, .4], [.1, .3]], dtype=dtype))
    embedding = torch.nn.Parameter(torch.tensor([.1, -.2], dtype=dtype))
    factors = ((a, a, b, b), (a, a, b, b) if same_slots else (b, a, a, b))
    transforms = [SourceTransform(a, b, embedding, row) for row in factors]
    for node, transform in zip(binding.model.nodes[1:], transforms):
        node.aggregate_program.transforms["0"] = transform
    # Identical deduplicated lists are precisely the former invalid cache key.
    assert tuple(map(id, transforms[0].parameters())) == tuple(map(id, transforms[1].parameters()))
    spec = SettleGraph(binding.graph, (1, 2), binding="ports", time_stride=3)
    return StaticKV(binding, spec), (a, b, embedding), factors


def literal(z, roots, factors):
    """Independent matrix formula; no candidate normalization/projection helper."""
    a, b, embedding = roots
    left_a, right_a, left_b, right_b = factors
    normalized = z/torch.sqrt(.2+(z*z).mean(-1, keepdim=True))
    hidden = normalized@(a+left_a@right_a).t()+embedding
    activated = hidden*torch.sigmoid(hidden)
    return z+.4*(activated@(b+left_b@right_b).t())


@pytest.mark.parametrize("equal_values,same_slots", [(True, False), (False, False), (False, True)])
def test_direct_source_slot_identity_values_and_independent_gradients(dtype, equal_values, same_slots):
    anchor, roots, factors = material(dtype, equal_values, same_slots)
    assert len(anchor.sources[1].transforms) == (1 if same_slots else 2)
    z = torch.tensor([[[.3, -.4], [.1, .2]]], dtype=dtype, requires_grad=True)
    masses = torch.ones((1, 1, 2), dtype=dtype)
    mass, value = anchor._aggregate(1, z, masses.bool(), {0: z.unsqueeze(0)}, {0: masses})
    expected = [literal(z, roots, row) for row in factors]
    equivalent(mass, torch.ones((2, 1, 2), dtype=dtype))
    equivalent(value, torch.stack(expected))
    if equal_values:
        equivalent(expected[0], expected[1], atol=0, rtol=0)
    elif not same_slots:
        assert not torch.allclose(expected[0], expected[1])
    bar = torch.tensor([[[.7, -.3], [0., .4]]], dtype=dtype)
    expected_gradients = []
    for index, target in enumerate(expected):
        actual = torch.autograd.grad((value[index]*bar).sum(), (z, *roots), retain_graph=True)
        wanted = torch.autograd.grad((target*bar).sum(), (z, *roots), retain_graph=True)
        equivalent(actual, wanted)
        expected_gradients.append(wanted)
    if equal_values:
        # The two output roots have equal values but different A/B derivatives.
        assert not torch.allclose(expected_gradients[0][1], expected_gradients[1][1])
