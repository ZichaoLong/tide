"""Separate region budgets/state and shared gradients survive tensor batching."""
from dataclasses import replace

import pytest
import torch

from tide import ModelConfig, TopologyConfig, TideModel, detach_state, save_checkpoint, load_checkpoint
from tide.precision import precision_context
from benchmarks.statistics import audit_model


@pytest.mark.parametrize("sharing,memory,timing,partition", [
    ("independent", "K", "settle", "group"),
    ("region_shared", "X", "settle", "group"),
    ("layer_shared", "X", "timed", "group"),
    ("region_shared", "X", "timed", "mixed"),
])
def test_region_batch_values_vjp_and_private_continuation(sharing, memory, timing, partition):
    torch.manual_seed(328)
    cfg = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4",
                     n=2, hubs=1, timing=timing, partition=partition), d=64,
                     memory=memory, sharing=sharing, attention_backend="math")
    serial, batched = TideModel(cfg).double(), TideModel(replace(cfg, region_batch_size=4)).double()
    batched.load_state_dict(serial.state_dict())
    x = torch.randn(2, 4, 64, dtype=torch.float64)
    results, gradients = [], []
    for model in (serial, batched):
        inp = x.clone().requires_grad_()
        a = model(inp[:, :2], record_work=True)
        b = model(inp[:, 2:], a.state, record_work=True)
        results.append((a, b))
        loss = a.readout().square().mean() + b.readout().square().mean()
        if timing == "timed":
            loss = loss + model.flush(b.state).readout().square().mean()
        loss.backward()
        gradients.append([inp.grad] + [p.grad for p in model.parameters()])
    for ra, rb in zip(*results):
        torch.testing.assert_close(ra.values, rb.values, rtol=2e-8, atol=2e-8)
        torch.testing.assert_close(ra.strengths, rb.strengths, rtol=2e-8, atol=2e-8)
        assert ra.work == rb.work
        for ma, mb in zip(ra.state.memories, rb.state.memories):
            for kind in ma:
                assert torch.equal(ma[kind].times, mb[kind].times)
    for ga, gb in zip(*gradients):
        assert (ga is None) == (gb is None)
        if ga is not None:
            torch.testing.assert_close(ga, gb, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_region_batch_recompute_restore_update(precision, tmp_path):
    torch.manual_seed(721)
    cfg = ModelConfig(topology=TopologyConfig(kind="lattice", n=2, depth=1), d=64,
                     sharing="region_shared", region_batch_size=4, checkpoint_regions=True,
                     local_attention_backend="math")
    model = TideModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    state = None
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with precision_context("cpu", precision):
            result = model(torch.randn(1, 3, 64), state)
            loss = result.readout().square().mean()
        loss.backward(); optimizer.step()
        state = detach_state(result.state)
    checkpoint = tmp_path / "batch.pt"
    save_checkpoint(checkpoint, model, optimizer=optimizer, state=state)
    restored, rs, slots = load_checkpoint(checkpoint)
    ropt = torch.optim.AdamW(restored.parameters(), lr=1e-4); ropt.load_state_dict(slots)
    x = torch.randn(1, 3, 64)
    for m, opt, s in ((model, optimizer, state), (restored, ropt, rs)):
        opt.zero_grad(set_to_none=True)
        with precision_context("cpu", precision):
            loss = m(x, s).readout().square().mean()
        loss.backward(); opt.step()
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_region_batch_work_audit_counts_each_region_and_shared_parameter():
    torch.manual_seed(483)
    cfg = ModelConfig(topology=TopologyConfig(kind="lattice", n=2, depth=1), d=64,
                     sharing="layer_shared", attention_backend="math")
    serial, batch = TideModel(cfg).double(), TideModel(replace(cfg, region_batch_size=4)).double()
    batch.load_state_dict(serial.state_dict())
    x = torch.randn(1, 4, 64, dtype=torch.float64)
    with torch.no_grad():
        _, _, a = audit_model(serial, x)
        _, _, b = audit_model(batch, x)
    assert a["effective_touched"] == b["effective_touched"]
    for key in ("candidate_updates", "full_actions", "positive_messages", "regional_attention_positive_pairs"):
        assert a["work"][key] == b["work"][key]
