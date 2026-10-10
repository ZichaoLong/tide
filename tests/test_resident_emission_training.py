"""Actual public slot-affine gradients/updates against independent Python autograd."""
import pytest
import torch
from resident_test_target import owner_devices
from tidegraph import ResidentPlacement, ResidentTrainingLimits
from test_resident_training import target, training_case
from resident_training_cases import runtime, inputs, roots


@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("schedule", ["streaming", "greedy"])
def test_projection_public_training(target, family, schedule, tmp_path):
    placement = None
    if schedule == "greedy":
        start = torch.device(target).index
        placement = ResidentPlacement(devices=owner_devices(target, 2))
    training_case(target, family, schedule, "adamw" if schedule == "greedy" else "sgd", tmp_path,
                  full="lh-silu-rms-v1", aggregation="all_softmax", emission="slot_affine",
                  placement=placement, model_device="cpu")


@pytest.mark.parametrize("mode", ["hst", "softp"])
def test_controlled_projection_is_explicitly_refused(target, mode):
    r = runtime("pdg", target, mode=mode, emission="slot_affine", model_device="cpu")
    with torch.no_grad(), pytest.raises((RuntimeError, ValueError), match="slot-affine|broadcast"):
        r.training_session(1)


def test_projection_backward_capacity_is_explicit(target):
    r = runtime("pdg", target, emission="slot_affine", model_device="cpu")
    with torch.no_grad(), r.training_session(1, limits=ResidentTrainingLimits(backward_bytes=1)) as s:
        args, kw = inputs(s, torch.ones(1, 2, 4)*.01, 0, 2)
        window = s.advance_device(args, **kw)
        with pytest.raises((RuntimeError, ValueError), match="budget"):
            s.backward([roots(s, window, "all")])


def test_projection_node_without_output_slots(target):
    import os
    from tidegraph import (Graph, Node, Region, GraphConfig, GraphRuntime,
                           ExecutionOptions, ExecutionPlacement)
    graph = Graph((Node(0, emission="slot_affine"),), (), (Region(1),), (0,), ())
    cfg = GraphConfig("pdg", graph, width=4)
    r = GraphRuntime(cfg, device=target, model_device="cpu",
        native_library=os.environ["TIDE_BUILD_DIR"], resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],
        options=ExecutionOptions(implementation="native", schedule="greedy", trace=True,
                                 placement=ExecutionPlacement(preset="resident")))
    cpu = GraphRuntime(cfg, device="cpu", options=ExecutionOptions(schedule="reference", packed=False, trace=True))
    x = torch.ones(1, 2, 4)*.01
    oracle = cpu.session(1)
    args, kw = inputs(oracle, x, 0, 2)
    expected = oracle.advance(args, **kw)
    from tidegraph.compare import equivalent
    with torch.no_grad(), r.training_session(1) as s:
        args, kw = inputs(s, x, 0, 2)
        window = s.advance_device(args, **kw)
        equivalent(expected, s.result())
        s.backward([s.cotangents(window, final=torch.ones_like(window.state_values))])
        assert s.step().applied
