"""Reception execution choices through real private state and optimizer steps."""
from dataclasses import replace

import pytest
import torch

from tide import (ModelConfig, TideModel, TopologyConfig, canonical_state_dict,
                  load_checkpoint, place_model, save_checkpoint)
from benchmarks.continuous import nested_comparison
from benchmarks.statistics import audit_model
from test_training_optimizations import update


PROFILES = ("incoming_light_64", "incoming_light_128", "incoming_wide")


def config(profile, timing="settle", **options):
    return ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4", n=2,
                       hubs=1, partition="mixed", timing=timing), d=64, memory="X",
                       sharing="layer_shared", attention_backend="math", aggregation=profile,
                       region_batch_size=4, **options)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("timing", ["settle", "timed"])
def test_fp64_fused_masked_reception_updates_continuation_and_restore(tmp_path, profile, timing):
    torch.manual_seed(519)
    cfg = config(profile, timing)
    baseline = TideModel(cfg).double()
    candidate = TideModel(replace(cfg, reception_qkv_backend="fused", reception_mask_padding=True,
                                 checkpoint_reception=True)).double()
    candidate.load_state_dict(baseline.state_dict())
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-4) for model in (baseline, candidate)]
    states = [None, None]
    # Mixed regions contain D/H nodes with different static in-degrees. This
    # actually exercises padding, as well as genuine runtime zero strengths.
    masks = [candidate._reception_mask((r,), "cpu") for r in range(len(candidate.graph.regions))]
    assert any(mask is not None and (~mask).any() for mask in masks)
    for _ in range(3):
        inputs = torch.randn(1, 5, 64, dtype=torch.float64)
        observations = []
        for i, (model, optimizer) in enumerate(zip((baseline, candidate), optimizers)):
            states[i], grads, results = update(model, optimizer, states[i], inputs)
            observations.append(dict(state=states[i], gradients=grads, results=results,
                                     parameters=model.state_dict(), optimizer=optimizer.state_dict()))
        report = nested_comparison(*observations, atol=2e-8, rtol=2e-7)
        assert report["within_threshold"] and not report["support_mismatches"], report
    checkpoint = tmp_path / "reception.pt"
    save_checkpoint(checkpoint, candidate, state=states[1], optimizer=optimizers[1])
    restored, rs, slots = load_checkpoint(checkpoint)
    assert restored.config == candidate.config
    ropt = torch.optim.AdamW(restored.parameters()); ropt.load_state_dict(slots)
    assert nested_comparison(dict(parameters=restored.state_dict(), state=rs, optimizer=slots),
                             dict(parameters=candidate.state_dict(), state=states[1],
                                  optimizer=optimizers[1].state_dict()), atol=0, rtol=0)["exact"]
    inputs = torch.randn(1, 5, 64, dtype=torch.float64)
    observations = []
    for model, optimizer, state in ((candidate, optimizers[1], states[1]), (restored, ropt, rs)):
        following, grads, results = update(model, optimizer, state, inputs)
        observations.append(dict(state=following, gradients=grads, results=results,
                                 parameters=model.state_dict(), optimizer=optimizer.state_dict()))
    report = nested_comparison(*observations, atol=0, rtol=0)
    assert report["exact"], report


@pytest.mark.parametrize("profile,precision,timing", [
    ("incoming_light_64", "fp32", "settle"),
    ("incoming_light_128", "fp32", "timed"),
    ("incoming_wide", "fp32", "settle"),
    ("incoming_light_128", "bf16", "settle"),
])
def test_fused_masked_reception_restores_actual_update(tmp_path, profile, precision, timing):
    torch.manual_seed(520)
    cfg = config(profile, timing, reception_qkv_backend="fused", reception_mask_padding=True)
    model = TideModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    state = None
    for _ in range(3):
        state, _, _ = update(model, optimizer, state, torch.randn(1, 4, 64), precision)
    checkpoint = tmp_path / "reception.pt"
    save_checkpoint(checkpoint, model, state=state, optimizer=optimizer)
    restored, rs, slots = load_checkpoint(checkpoint)
    ropt = torch.optim.AdamW(restored.parameters()); ropt.load_state_dict(slots)
    inputs = torch.randn(1, 4, 64)
    observations = []
    for owner, opt, old in ((model, optimizer, state), (restored, ropt, rs)):
        following, grads, results = update(owner, opt, old, inputs, precision)
        observations.append(dict(state=following, gradients=grads, results=results,
                                 parameters=owner.state_dict(), optimizer=opt.state_dict()))
    report = nested_comparison(*observations, atol=0, rtol=0)
    assert report["exact"], report


def test_fused_reception_keeps_work_and_parameter_inventory():
    torch.manual_seed(521)
    reference = TideModel(config("incoming_light_64")).double()
    candidate = TideModel(replace(reference.config, reception_qkv_backend="fused",
                                  reception_mask_padding=True)).double()
    candidate.load_state_dict(reference.state_dict())
    inputs = torch.randn(1, 3, 64, dtype=torch.float64)
    with torch.no_grad():
        _, _, ordinary = audit_model(reference, inputs)
        _, _, fused = audit_model(candidate, inputs)
    for key in ("parameters", "effective_touched", "work", "node_matrices"):
        assert ordinary[key] == fused[key], key
    assert all(f"receive.{name}" in fused["node_matrices"] for name in ("q", "k", "v"))


def test_placed_reception_explicit_fallback_keeps_canonical_gradients():
    torch.manual_seed(522)
    cfg = replace(config("incoming_light_128", reception_mask_padding=True), region_batch_size=1)
    ordinary = TideModel(cfg).double()
    candidate = TideModel(replace(cfg, reception_qkv_backend="fused")).double()
    candidate.load_state_dict(ordinary.state_dict())
    placed = place_model(candidate, ["cpu", "cpu"])
    assert placed.receive.effective_qkv_backend == "separate_owner_wrapper"
    assert canonical_state_dict(placed).keys() == ordinary.state_dict().keys()
    inputs = torch.randn(1, 3, 64, dtype=torch.float64)
    observations = []
    for model in (ordinary, placed):
        x = inputs.clone().requires_grad_()
        first = model(x[:, :1])
        result = model(x[:, 1:], first.state)
        loss = first.readout().square().mean() + result.readout().square().mean()
        loss.backward()
        gradients = (dict(canonical_state_dict(model, gradients=True)) if model is placed else
                     {name: p.grad for name, p in model.named_parameters()})
        observations.append(dict(results=(first, result), input_gradient=x.grad, gradients=gradients))
    report = nested_comparison(*observations, atol=2e-9, rtol=2e-8)
    assert report["within_threshold"] and not report["support_mismatches"], report


def test_reception_execution_configuration_and_older_checkpoint_defaults():
    old = ModelConfig().to_dict()
    del old["reception_qkv_backend"], old["reception_mask_padding"]
    assert ModelConfig.from_dict(old) == ModelConfig()
    for options in (dict(reception_qkv_backend="implicit"), dict(reception_mask_padding=1)):
        with pytest.raises(ValueError, match="reception_"):
            ModelConfig(**options)
