"""Allocation §6/S.2: rejected actual rows survive grouping and serialization.

Bern has strictly positive support for finite scores. These extreme rows must
reject underflow, independently of the numerical root implementation.
"""
import json
from dataclasses import replace
import pytest
import torch
from tidegraph.weighted.allocation_failure import AllocationFailure
from tidegraph.weighted.allocation_rows import allocate_batch


@pytest.mark.parametrize("native", [False, True])
def test_failure_rows_snapshot_and_json(dtype, native):
    scores = [torch.zeros(n, dtype=dtype, requires_grad=True) for n in (3, 2, 1, 2)]
    with torch.no_grad():
        scores[3].copy_(scores[3].new_tensor([2000., -2000.]))
    caps = [torch.ones_like(s) for s in scores]
    bounds = [s.new_tensor(1.) for s in scores]
    bounds[3] = scores[3].new_tensor(.125)
    with pytest.raises(AllocationFailure, match="positive.*support underflowed") as caught:
        if native:
            import _tide_native as core
            core.weighted_allocation_batch([(s, c, b, b, s.new_tensor(1.), None) for s, c, b in zip(scores, caps, bounds)], "bern", None)
        else:
            allocate_batch(scores, caps, bounds, bounds, regularizer="bern")
    error = caught.value
    assert error.row == 3
    assert error.tensors["scores"].grad_fn is None and not error.tensors["scores"].requires_grad
    with torch.no_grad():
        scores[3].zero_()
    assert error.tensors["scores"].tolist() == [2000., -2000.]
    record = json.loads(json.dumps(error.record(), allow_nan=False))
    assert record["tensors"]["scores"]["values"] == [2000., -2000.]
    assert record["tensors"]["scores"]["dtype"] == str(dtype).removeprefix("torch.")
    assert record["details"]["regularizer"] == "bern" and record["details"]["branch"] == "equality"


@pytest.mark.parametrize("native", [False, True])
def test_failure_region_group_mapping_and_int64_time(dtype, native):
    from test_weighted_region_batch import fixture, native_region
    from tidegraph.weighted.region_batch import evaluate
    program, requests = fixture(dtype, 0)
    program.config = replace(program.config, region_score="ind", region_key=None, selection_width=None,
                             query_weight=None, alpha_loc=None, alpha_sel=None,
                             region_control_mode="fixed", region_evidence=None, total_fraction=.5, k=1)
    # Put a singleton before two equal-shaped rows. Failure belongs to row 2.
    requests = [requests[4], requests[0], requests[1]]
    descriptors = tuple((v, dict(d, score=d["score"].new_tensor(2000. if i == 0 else -2000.),
                                w=d["w"].new_tensor(.05), mu=d["mu"].new_tensor(1.)))
                        for i, (v, d) in enumerate(requests[2].candidates))
    requests[2] = replace(requests[2], candidates=descriptors, time=2**60+13)
    with pytest.raises(AllocationFailure) as caught:
        if native:
            native_region(program, requests, grad=False)
        else:
            evaluate(program, requests)
    error = caught.value
    assert error.row == 2 and error.time == 2**60+13
    assert error.nodes == [v for v, _ in descriptors]
    assert error.members == list(requests[2].layout.members)
    assert json.loads(json.dumps(error.record()))["time"] == 2**60+13
