"""Independent checks for the optional standalone regional subpath reference.

The separate scripts/check_native_region.py exercises actual C++ binaries; an
ordinary Python test pass does not certify C++ build, devices or concurrency.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from tide.region import RegionNetwork

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_region_reference", ROOT / "native/regional/reference.py")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


@pytest.mark.parametrize("bf16", [False, True])
def test_reference_matches_active_region_values_and_shared_vjp(tmp_path, bf16):
    tensors = reference.fixture(tmp_path / "fixture")
    candidate = reference.region_type()(tensors, "cpu")
    production = RegionNetwork(64, backend="math")
    names = {name: name.replace("_", ".") if name != "readout" else name for name in reference.PARAMETERS}
    production.load_state_dict({names[key]: tensors[key] for key in reference.PARAMETERS})
    outputs = []
    all_inputs = []
    for model in (candidate, production):
        inputs = {name: tensors[name].clone().requires_grad_() for name in ("z", "w", "score")}
        values = []
        for task in range(4):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
                if model is candidate:
                    value = model(inputs["z"][task], inputs["score"][task], inputs["w"][task], bf16=bf16)
                else:
                    value = model(inputs["z"][task], inputs["score"][task], inputs["w"][task])
            values.append(value)
        value = torch.stack(values)
        (value * tensors["target"]).mean().backward()
        outputs.append(value.detach()); all_inputs.append(inputs)
    tolerance = dict(rtol=2e-4, atol=2e-5) if bf16 else dict(rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(*outputs, **tolerance)
    for name in ("z", "w", "score"):
        torch.testing.assert_close(all_inputs[0][name].grad, all_inputs[1][name].grad, **tolerance)
    for key, parameter in candidate.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(production.named_parameters())[names[key]].grad, **tolerance)
    assert all_inputs[0]["w"].grad[1, 0].abs().max() > 0
    assert torch.equal(all_inputs[0]["w"].grad[2], torch.zeros_like(all_inputs[0]["w"].grad[2]))


def test_literal_quotient_fp64_vjp_and_zero_prior_right_difference(tmp_path):
    tensors = reference.fixture(tmp_path / "fixture", tasks=2)
    tensors = {name: value.double() for name, value in tensors.items()}
    models, values, gradients = [], [], []
    for literal in (False, True):
        model = reference.region_type()(tensors, "cpu")
        z, score, w = [tensors[name][1].clone().requires_grad_() for name in ("z", "score", "w")]
        output = model(z, score, w, literal=literal)
        # Query zero remains zero by specification; other positive queries
        # witness the right derivative of the zero-prior key.
        loss = (output[1:] * tensors["target"][1, 1:]).sum()
        gradients.append(torch.autograd.grad(loss, (z, score, w, *model.parameters())))
        models.append(model); values.append(output)
    torch.testing.assert_close(*values, rtol=1e-12, atol=1e-12)
    for left, right in zip(*gradients):
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-11)
    z, score, w = [tensors[name][1].clone() for name in ("z", "score", "w")]
    epsilon = 1e-7
    perturbed = w.clone(); perturbed[0, 0, 0] += epsilon
    def objective(weights):
        return (models[0](z, score, weights)[1:] * tensors["target"][1, 1:]).sum()
    finite = (objective(perturbed) - objective(w)) / epsilon
    torch.testing.assert_close(finite, gradients[0][2][0, 0, 0], rtol=2e-5, atol=2e-7)


def test_shared_owner_gradient_is_sum_of_task_contributions(tmp_path):
    tensors = reference.fixture(tmp_path / "fixture")
    combined = reference.region_type()(tensors, "cpu")
    local_gradients = []
    for task in range(4):
        inputs = [tensors[name][task] for name in ("z", "score", "w")]
        target = tensors["target"][task]
        (combined(*inputs) * target).mean().backward()
        local = reference.region_type()(tensors, "cpu")
        (local(*inputs) * target).mean().backward()
        local_gradients.append({name: value.grad for name, value in local.named_parameters()})
    for name, value in combined.named_parameters():
        torch.testing.assert_close(value.grad, sum(row[name] for row in local_gradients))


def test_portable_bundle_integrity_rejects_corruption(tmp_path):
    tensors = reference.fixture(tmp_path / "fixture")
    loaded = reference.read_bundle(tmp_path / "fixture")
    for key in tensors:
        torch.testing.assert_close(tensors[key], loaded[key], rtol=0, atol=0)
    path = tmp_path / "fixture/z.f32"
    raw = bytearray(path.read_bytes()); raw[0] ^= 1; path.write_bytes(raw)
    with pytest.raises(ValueError, match="integrity"):
        reference.read_bundle(tmp_path / "fixture")
