"""Nonempty independent LH banks and independently rooted public observations."""
import os
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from lh_tensor_check import evaluate, compare
from lh_tensor_observations import ROOT_FAMILIES, initial_banks, initial_continuation
from test_lh_tensor_baseline import fixture


@pytest.mark.parametrize("client", ["python", "native"])
@pytest.mark.parametrize("family", ROOT_FAMILIES)
def test_nonempty_left_independent_field_vjps(client, family):
    anchor = fixture(torch.float64, profile="textbook")
    x = torch.linspace(-.2, .4, 24, dtype=torch.float64).reshape(2, 3, 4)
    support = torch.tensor([[[True, False, True], [True, True, False]],
                            [[True, True, False], [False, True, False]]])
    options = dict(client=client, library=os.environ.get("TIDE_BUILD_DIR"), left_entries=3,
                   start=2**53//anchor.spec.stride+8, root_family=family, capture_bytes=2**26)
    expected, _ = evaluate(anchor.binding, anchor.spec, dict(positions=3), x, support, level="direct", **options)
    actual, info = evaluate(anchor.binding, anchor.spec, dict(positions=3), x, support, level="default", **options)
    report = compare(actual, expected)
    assert report["strict_numerical_projection_passed"], report
    assert info["root_family"] == family
    assert any(name.startswith("left/") for name in actual["gradients"])
    assert all(t.dtype == torch.int64 for name, t in actual["raw"].items()
               if name.endswith(("times", "last_time", "observations")))


def test_literal_banks_pack_without_running_the_model():
    anchor = fixture(torch.float64)
    x = torch.zeros(2, 2, 4, dtype=torch.float64)
    banks = initial_banks(anchor, x, 13, 3)
    q = initial_continuation(anchor, banks, 13, 2)
    bank = banks[0]
    mask = bank["valid"][0, 0]
    assert torch.equal(q.states[0, 0].slots["times"], bank["times"][mask])
    assert torch.equal(q.states[0, 0].slots["weights"], bank["weights"][0, 0, mask])
    g, = torch.autograd.grad(q.states[0, 0].slots["keys"].sum(), bank["keys"])
    expected = torch.zeros_like(g)
    expected[0, 0, mask] = 1
    assert torch.equal(g, expected)
    assert not q.pending and not q.ledger
    with pytest.raises(ValueError, match="start"):
        initial_banks(anchor, x, 3, 3)
