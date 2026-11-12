"""Experiment call/lifetime contract with a CPU owner double, not a device gate."""
from types import SimpleNamespace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.compare import equivalent
from tidegraph.reference import run
from tidegraph.experiment.block_resident import BlockResident
from tidegraph.experiment.block_inputs import BlockInputs
from tidegraph.experiment.tasks import WeightedOutputTask
from test_experiment_block_records import fixture, externals, actual_fields


class Owner:
    def __init__(self, spec, model, batch):
        self.spec, self.model = spec, model
        self.state = Continuation(spec.graph.identity, batch)
        self.cut, self.detached, self.calls = 0, True, []

    def advance(self, positions, payload, present):
        assert not torch.is_grad_enabled() and self.detached
        assert positions.ndim == 2 and payload.ndim == 4 and present.ndim == 3
        self.calls.append("advance-tensors")
        inputs = positions, payload, present
        self.cut = (int(positions[0, -1])+1)*self.spec.stride
        self.expected = run(self.spec.graph, self.model, self.state, externals(self.spec, inputs, self.state),
                            self.cut, sealed_until=self.cut)
        self.state = self.expected.continuation.detach()
        self.window = actual_fields(self.spec, inputs, self.expected)
        self.detached = False
        return self.window

    def cotangents(self, window, roots):
        assert window is self.window
        assert set(roots) == {"output.strength", "output.content"}
        assert all(not x.requires_grad for x in roots.values())
        self.calls.append("field-roots")
        return roots

    def backward(self, roots):
        assert len(roots) == 1 and not self.detached
        self.calls.append("backward")
        return SimpleNamespace(parameters=SimpleNamespace(names=[], values=torch.zeros(1)), inputs=[None])

    def _release(self):
        self.detached = True
        # Observation fields may cease to be borrowed after the consume call.
        # The adapter must already own requested CPU values at this boundary.
        for f in self.window.fields.values():
            f.values.fill_(float("nan"))

    def step(self):
        self.calls.append("step")
        self._release()
        return None  # Public v2 has no old `.applied` step record.

    def detach(self):
        self.calls.append("detach")
        self._release()

    def continuation(self):
        assert self.detached
        self.calls.append("continuation")
        return self.state


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("mode", ["complete", "outputs", "metrics"])
def test_adapter_uses_field_loss_and_detaches_before_complete_decode(training, mode):
    plan, spec, model, _ = fixture("staggered-v1")
    owner = BlockResident.__new__(BlockResident)
    owner.training, owner.spec, owner.devices = training, dict(return_mode=mode), (torch.device("cpu"),)
    owner.settle, owner.inputs = spec, BlockInputs(plan)
    owner.task = WeightedOutputTask(dict(objective=dict(scale=.03)), plan)
    owner.owner = Owner(spec, model, plan["samples"])
    actual, loss = owner.step(0)
    expected = owner.owner.expected
    assert loss == pytest.approx(float(owner.task.loss(expected, training=training)), rel=2e-6, abs=1e-7)
    if mode == "complete":
        equivalent(actual, expected, atol=0, rtol=0)
        assert owner.owner.calls[-1] == "continuation"
    elif mode == "outputs":
        equivalent(actual["outputs"], expected.outputs, atol=0, rtol=0)
    else:
        assert actual == dict(return_mode="metrics", cut=plan["span"])
    assert owner.owner.calls[:4] == (["advance-tensors", "field-roots", "backward", "step"] if training else
                                   ["advance-tensors", "detach"] + (["continuation"] if mode == "complete" else []))
    assert owner.last_work["candidate_updates"] == len(expected.trace)
