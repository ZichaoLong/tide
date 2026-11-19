"""Independent matrix formulas for optional fused projection candidates."""
import pytest
import torch

from benchmarks.projections import projection_candidate
from tide.parameters import FeedForward, HeadQuery, NodeOwners, norm


def explicit_linear(layer, x, ids):
    rows = []
    for row, node in enumerate(ids.tolist()):
        group = layer._host_groups[node]
        weight = layer.weight[group]
        if layer.shared:
            weight = weight + layer.left[node] @ layer.right[node]
        out = x[row] @ weight.T
        if layer.bias is not None:
            out = out + layer.bias[group]
        rows.append(out)
    return torch.stack(rows)


@pytest.mark.parametrize("groups", [None, [0, 0, 0], [0, 1, 0]])
@pytest.mark.parametrize("kind", ["head_query", "ffn"])
def test_projection_fusion_independent_formula_and_shared_vjp(kind, groups):
    torch.manual_seed(119)
    constructor = HeadQuery if kind == "head_query" else FeedForward
    module = constructor(3, 64, groups=groups, rank=3).double()
    ids = torch.tensor([2, 0, 2, 1])
    owners = NodeOwners(ids, (0, 1, 2))
    x = torch.randn(4, 2, 3, 64, dtype=torch.float64, requires_grad=True)
    if kind == "head_query":
        expected = torch.cat([explicit_linear(head, x, ids) for head in module.heads], -1)
    else:
        z = norm(x)
        gate, up = explicit_linear(module.gate, z, ids), explicit_linear(module.up, z, ids)
        expected = x + explicit_linear(module.down, torch.nn.functional.silu(gate) * up, ids)
    parameters = tuple(module.parameters())
    parameter_ids = tuple(id(p) for p in parameters)
    seed = torch.randn_like(expected)
    reference = torch.autograd.grad((expected * seed).sum(), (x, *parameters), retain_graph=True)
    for candidate in ("native", "both"):
        with projection_candidate(module, candidate):
            actual = module(x, owners)
            gradient = torch.autograd.grad((actual * seed).sum(), (x, *parameters), retain_graph=True)
            assert tuple(id(p) for p in module.parameters()) == parameter_ids
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-12)
        for a, b in zip(gradient, reference):
            torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-11)
    assert "forward" not in module.__dict__


def test_projection_candidate_restores_after_exception():
    module = HeadQuery(1, 64)
    with pytest.raises(RuntimeError, match="witness"):
        with projection_candidate(module):
            raise RuntimeError("witness")
    assert "forward" not in module.__dict__
