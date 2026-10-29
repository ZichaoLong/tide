"""Decode actual slots; None, connected zero and never-updated owners differ."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from types import SimpleNamespace
import pytest
import torch
from weighted_training_observation import SGDLayout


def test_resident_sgd_decoder_alias_offsets_connectivity_and_slots():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.tensor([1., 2.]))
    model.alias = model.a
    model.alias2 = model.a
    model.b = torch.nn.Parameter(torch.tensor([3.]))
    model.c = torch.nn.Parameter(torch.tensor([4.]), requires_grad=False)
    layout = SGDLayout(model)
    raw = dict(values=torch.tensor([1., 2., 3.]), first=torch.tensor([.2, .4, 0.]),
               second=torch.zeros(3), maximum=torch.zeros(3), steps=torch.tensor([2, 0]), corrections=torch.zeros(2, 2))
    checkpoint = dict(optimizer="sgd", state=raw, offsets=[0, 2], trainable=["alias", "alias2", "b"],
                      groups=[dict(parameters=["alias", "b"], lr=.001, momentum=.3,
                                   weight_decay=.01, dampening=0., nesterov=False, maximize=False)],
                      parameters={"alias": model.a.detach(), "b": model.b.detach(),
                                  "c": model.c.detach(), "legacy_buffer": torch.ones(3)})
    parameters = layout.parameters(checkpoint)
    assert set(parameters) == {"a", "b", "c"}
    assert torch.equal(parameters["a"], model.a)
    g = SimpleNamespace(names=["alias", "b"], offsets=[0, 2], values=torch.zeros(3), connected=torch.tensor([True, False]))
    result = layout.decode(g, checkpoint)
    assert result["gradients"]["b"] is None and result["gradients"]["c"] is None
    assert torch.equal(result["gradients"]["a"], torch.zeros(2))
    assert result["update_counts"] == dict(a=2, b=0, c=0)
    assert result["optimizer"]["state"].keys() == {0}
    torch.testing.assert_close(result["optimizer"]["state"][0]["momentum_buffer"], torch.tensor([.2, .4]))
    raw["maximum"][0] = 1
    with pytest.raises(ValueError, match="unused SGD"):
        layout.decode(g, checkpoint)
    raw["maximum"][0] = 0
    checkpoint["groups"][0]["lr"] = .01
    with pytest.raises(ValueError, match="SGD option differs: lr"):
        layout.decode(g, checkpoint)
    checkpoint["groups"][0]["lr"] = .001
    g.offsets[-1] = 3
    with pytest.raises(ValueError, match="slice exceeds actual storage"):
        layout.decode(g, checkpoint)
