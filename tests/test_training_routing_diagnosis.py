"""The observer must preserve real checkpointed training and its update."""
from dataclasses import replace

import torch

from tide import ModelConfig, TideModel, TopologyConfig, placement_adamw
from scripts.check_training_paths import build_candidate, observe, screen, training_data
from scripts.diagnose_routing_divergence import CaptureBudget
from scripts.diagnose_training_routing import capture_update


def test_capture_does_not_change_checkpoint_backward_or_optimizer_update():
    torch.manual_seed(43)
    config = ModelConfig(topology=TopologyConfig(kind="fc", n=2, layers=2), d=64,
                         memory="K", checkpoint_regions=True)
    model = TideModel(config)
    optimizer = placement_adamw(model, lr=1e-4, foreach=False)
    value = torch.randn(1, 3, 64)
    _, state, _ = observe(model, optimizer, None, value, "fp32", ["cpu"], 2, update=True)
    anchor = training_data(model, optimizer, state)
    observed = []
    for capture in (False, True):
        model, optimizer, state = build_candidate(config, "baseline", anchor,
            ["cpu"], "fp32", "auto", overrides={})
        if capture:
            result, _, _, trace = capture_update(model, optimizer, state, value, "fp32", "cpu", 2, CaptureBudget(1 << 20))
            assert len(trace.calls) == 2
            assert len(trace.entries) == 2*len(model.graph.regions)
        else:
            result, _, _ = observe(model, optimizer, state, value, "fp32", ["cpu"], 2, update=True)
        observed.append(screen(result)["sha256"])
    assert observed[0] == observed[1]
