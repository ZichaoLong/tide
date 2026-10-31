import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.comparison import NonFiniteMismatch
from tidegraph.experiment import finite
from tidegraph.experiment.config import resolve
from tidegraph.experiment.eager import Eager


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_packed_finite_rejects_late_chunk_and_noncontiguous_values(monkeypatch, bad):
    monkeypatch.setattr(finite, "PACK_BYTES", 32)
    leaves = [torch.ones(4) for _ in range(8)]
    leaves[-1][2] = bad
    for packing in ("scalar", "packed"):
        with pytest.raises(NonFiniteMismatch):
            finite.finite_tensors(dict(values=leaves, disconnected=None), packing=packing)
        with pytest.raises(NonFiniteMismatch):
            finite.finite_tensors(torch.tensor([[0., bad], [1., 2.]]).T, packing=packing)
        with pytest.raises(NonFiniteMismatch):
            finite.finite_tensors([torch.ones(2, dtype=torch.complex64),
                                  torch.tensor([complex(0, bad)])], packing=packing)


def test_failure_localization_keeps_parameter_name_and_stage():
    for packing in ("scalar", "packed"):
        with pytest.raises(NonFiniteMismatch) as error:
            finite.finite_tensors({"shared.weight": torch.ones(4),
                                  "private.bias": torch.tensor([float("inf")])},
                                 packing=packing, path="runtime.gradients")
        assert error.value.path == "runtime.gradients[private.bias]"
        assert error.value.record()["kind"] == "nonfinite"


def test_finite_packing_preserves_values_aliases_and_independent_gradients(monkeypatch):
    monkeypatch.setattr(finite, "PACK_BYTES", 32)
    x = torch.arange(12., requires_grad=True)
    y = (x*3).reshape(3, 4)
    value = [x, x, x[:4], x[4:8], y, y.T, None, torch.empty(0), torch.arange(4, dtype=torch.int64),
             torch.tensor([1+2j, 3-4j]), torch.ones(2, dtype=torch.float64)]
    before = x.detach().clone()
    for packing in ("scalar", "packed"):
        finite.finite_tensors(value, packing=packing)
        assert torch.equal(x.detach(), before)
        assert torch.equal(torch.autograd.grad(y.sum(), x, retain_graph=True)[0], torch.full_like(x, 3))
    assert value[0] is value[1] and x.grad is None


def request(packing):
    return dict(schema="tide-experiment-v1", model=dict(recipe=dict(topology="fc", width=8,
        depth=2, region_width=2, history=4)), task=dict(samples=2, positions=3, input_pattern="staggered-v1"),
        execution=dict(device="cpu", client="pytorch", placement="cpu", packing="packed",
                       finite_check_packing=packing), measurement=dict(mode="training"))


def test_nonfinite_gradient_names_the_root_and_prevents_optimizer_publication():
    owner = Eager(resolve(request("packed")))
    name, parameter = next(owner.parameter_owner.named_parameters())
    before = owner.parameters()
    hook = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    try:
        with pytest.raises(NonFiniteMismatch) as error:
            owner.step(0)
        assert error.value.path == f"runtime.gradients[{name}]"
        equivalent(before, owner.parameters(), atol=0, rtol=0)
        assert not owner.optimizer.state
    finally:
        hook.remove()
        owner.close()


def test_policy_identity_budget_and_unsupported_client_are_explicit():
    scalar, packed = resolve(request("scalar")), resolve(request("packed"))
    assert scalar["model_id"] == packed["model_id"] and scalar["task_id"] == packed["task_id"]
    assert scalar["execution_id"] != packed["execution_id"]
    assert "finite_check_packing" not in scalar["execution"]["requested"]
    assert packed["capacity_budget"]["per_device"]["cpu"]["estimated_peak_bytes"] == (
        scalar["capacity_budget"]["per_device"]["cpu"]["estimated_peak_bytes"] + 32*1024**2)
    bad = request("packed"); bad["execution"]["client"] = "libtorch"
    with pytest.raises(ValueError, match="finite_check_packing"):
        resolve(bad)


def test_finite_policy_preserves_complete_training_shared_parameters_and_disk_resume(tmp_path):
    torch.set_num_threads(1)
    scalar, packed = Eager(resolve(request("scalar"))), Eager(resolve(request("packed")))
    compare = lambda a, b: equivalent(a, b, atol=0, rtol=0)
    try:
        for index in (0, 1):
            compare(scalar.step(index), packed.step(index))
            compare(scalar.training_state(), packed.training_state())
            compare(scalar.parameters(), packed.parameters())
        packed.save(tmp_path)
        restored = Eager(resolve(request("packed")), checkpoint=tmp_path)
        try:
            compare(packed.decoded_state(), restored.decoded_state())
            expected = scalar.step(2)
            compare(expected, packed.step(2)); compare(expected, restored.step(2))
            compare(scalar.training_state(), restored.training_state())
            compare(scalar.parameters(), restored.parameters())
        finally:
            restored.close()
    finally:
        scalar.close(); packed.close()
