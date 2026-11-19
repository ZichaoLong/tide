"""A normalized current instance can still reach a true positive-tail limit."""
import pytest
import torch

from tide import ModelConfig, TideModel, TopologyConfig


@pytest.mark.parametrize("kind", ["shannon", "pool"])
def test_normalized_model_keeps_true_positive_sender_range_guard(kind):
    # The independent score offsets follow the normalized features. This is
    # a valid current FC configuration and finite parameters, not a claim
    # that a downstream optimizer reaches these particular offsets.
    model = TideModel(ModelConfig(topology=TopologyConfig(n=2, layers=1),
                                d=64, allocation=kind))
    with torch.no_grad():
        model.describe.readout.value.zero_()
        model.describe.offset.value[:, 0] = torch.tensor([-110., 1.])
        for regional in model.regional:
            regional.readout.zero_()
    with pytest.raises(ArithmeticError, match="positive sender"):
        model(torch.ones(1, 1, 64))
