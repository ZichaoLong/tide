"""Adapter selection checks; these do not qualify an accelerator or C++ owner."""
from types import SimpleNamespace
import pytest
from tidegraph.weighted.resident import WeightedResident


class DeviceOwner:
    def __init__(self):
        self.calls = []

    def advance_device(self, inputs, stop):
        self.calls.append(("advance_device", inputs, stop))

    def outputs(self):
        self.calls.append(("outputs",))
        return [SimpleNamespace(batch=1, time=4, port=2, value="payload")]

    def metrics(self):
        self.calls.append(("metrics",))
        return dict(candidate_updates=3, outputs=1)

    def advance(self, inputs, stop):
        self.calls.append(("advance", inputs, stop))
        return "complete-result"


def adapter(owner):
    flow = object.__new__(WeightedResident)
    flow.owner, flow.cut = owner, 0
    flow._validate = lambda: None
    flow._inputs = lambda inputs, stop, sealed: list(inputs)
    flow._result = lambda actual: ("converted", actual)
    return flow


@pytest.mark.parametrize("mode", ["outputs", "metrics"])
def test_noncomplete_request_does_not_invoke_full_return_or_checkpoint(mode):
    owner = DeviceOwner()
    flow = adapter(owner)
    actual = flow.advance(["input"], stop=5, sealed_until=5, return_mode=mode)
    assert actual == dict(schema="tide-resident-return-v1", return_mode=mode, start=0, stop=5,
                         outputs=[(1, 4, 2, "payload")] if mode == "outputs" else None,
                         stats=dict(candidate_updates=3, outputs=1))
    assert flow.cut == 5
    assert owner.calls == [("advance_device", ["input"], 5), *(([("outputs",)]) if mode == "outputs" else []), ("metrics",)]


def test_default_complete_return_and_missing_build_capability_are_explicit():
    owner = DeviceOwner()
    flow = adapter(owner)
    assert flow.advance([], stop=2, sealed_until=2) == ("converted", "complete-result")
    assert owner.calls == [("advance", [], 2)]
    with pytest.raises(ValueError, match="return_mode"):
        flow.advance([], stop=3, sealed_until=3, return_mode="unknown")
    assert flow.cut == 2 and len(owner.calls) == 1
    old = adapter(SimpleNamespace(advance=owner.advance))
    with pytest.raises(ValueError, match="rebuild"):
        old.advance([], stop=3, sealed_until=3, return_mode="metrics")
    assert old.cut == 0 and len(owner.calls) == 1
