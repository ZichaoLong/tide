"""Static ownership domain, not a device/communication qualification."""
import pytest
import torch
from tidegraph.weighted.resident_placement import node_placement


def test_weighted_placement_is_per_node_and_preserves_requested_order():
    assert node_placement(None, "cuda:1", 3) == [torch.device("cuda:1")] * 3
    assert list(map(str, node_placement(["cuda:2", "cuda:0", "cuda:2"], "cuda:1", 3))) == ["cuda:2", "cuda:0", "cuda:2"]


@pytest.mark.parametrize("requested", [[], ["cuda:0"], "cuda:0", [0, 1], [True, False], ["cpu", "cuda:0"], ["cuda", "cuda:0"]])
def test_weighted_placement_rejects_missing_or_implicit_owners(requested):
    with pytest.raises(ValueError):
        node_placement(requested, "cuda:0", 2)
