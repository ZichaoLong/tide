"""Bool metadata preserves support, malformed-control refusal and private roots."""
from types import SimpleNamespace

import pytest
import torch

from tidegraph import Continuation, Graph, History, Node, Region
from tidegraph.ops import Model
from tidegraph.region import RegionProgram, Selection, evaluate_batch, prepare_request, validate_selection
from tidegraph.region_metadata import _group_flags, finite, positive


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.float64])
def test_grouped_flags_literal_boundaries_do_not_change_original_roots(storage_dtype):
    values = [torch.tensor(value, dtype=storage_dtype, requires_grad=True)
              for value in (0., 1., -1., float("nan"), float("inf"), -float("inf"))]
    versions = [value._version for value in values]
    assert _group_flags(values, positive=True) == [False, True, False, False, True, False]
    assert _group_flags(values, positive=False) == [True, True, True, False, False, False]
    assert positive(values) == [False, True, False, False, True, False]
    assert finite(values) == [True, True, True, False, False, False]
    assert all(value.requires_grad and value.is_leaf for value in values)
    assert versions == [value._version for value in values]
    gradients = torch.autograd.grad(3*values[0]+0*values[1], values[:3], allow_unused=True)
    assert gradients[0].item() == 3 and gradients[1].item() == 0 and gradients[2] is None


def test_grouped_finite_controls_preserve_vector_empty_and_dtype_order():
    values = [torch.empty((0, 2)), torch.tensor([[1., 2.], [3., 4.]]).t(), torch.tensor([1., float("nan")])]
    assert _group_flags(values, positive=False) == [True, True, False]
    mixed = [torch.tensor(1., dtype=torch.float64), torch.tensor(float("inf")), torch.tensor(0., dtype=torch.float64)]
    assert finite(mixed) == [True, False, True]
    assert positive(mixed) == [True, True, False]


def test_grouped_cast_keeps_positive_underflow_refusal_and_control_roots():
    from tidegraph.weighted.placement_tensors import controls_batch
    request = SimpleNamespace(payload_device=torch.device("cpu"), payload_dtype=torch.float32)
    a = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(1e-100, dtype=torch.float64, requires_grad=True)
    results = [Selection({0}, {0: a}, History()), Selection({1}, {1: b}, History())]
    assert _group_flags([a.float(), b.float()], positive=True) == [True, False]
    with pytest.raises(ValueError, match="positive support"):
        controls_batch(results, [request, request])
    assert a.grad is None and b.grad is None


def test_bulk_finite_validation_keeps_vector_control_refusal_without_committing_history():
    class Bulk(RegionProgram):
        profile = "bulk-control-test-v1"
        joint_batch = batched_autograd = history_independent = True

        def batch(self, requests):
            return [Selection({0}, {0: torch.tensor([1., 2.]),
                1: torch.tensor([0., float("nan") if i else 1.])}, History(last_time=r.time))
                for i, r in enumerate(requests)]

        batch_grad = batch

    g = Graph((Node(0), Node(0)), (), (Region(2, selector=Bulk.profile),), (0, 1), ())
    m = Model(g, dtype=torch.float32, region_programs={0: Bulk()})
    q = Continuation(g.identity, 2)
    frames = [(b, 0, 2**54+7, [(v, torch.tensor(.5)) for v in range(2)]) for b in range(2)]
    requests = [prepare_request(g, m, q, *frame) for frame in frames]
    values = m.regions[0].batch(requests)
    controls = [c for value in values for c in value.controls.values()]
    flags = dict(zip(map(id, controls), _group_flags(controls, positive=False)))
    validate_selection(g, m, 0, requests[0], values[0], control_finite=flags)
    with pytest.raises(ValueError, match="nonfinite control"):
        validate_selection(g, m, 0, requests[1], values[1], control_finite=flags)
    with pytest.raises(ValueError, match="nonfinite control"):
        evaluate_batch(g, m, q, frames)
    assert not q.history and not q.states and not q.ledger
