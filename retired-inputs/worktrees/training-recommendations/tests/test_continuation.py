"""Continuation across windows, pending flush, and a real resumed update."""
from dataclasses import fields, is_dataclass

import pytest
import torch

from benchmarks.baselines import TransformerBaseline
from benchmarks.continuous import checkpoint_resume, nested_comparison, run_stream, state_memory
from tide import ModelConfig, TideModel, TopologyConfig, detach_state


def _loss(value):
    return (value*torch.sin(torch.arange(value.numel(), device=value.device,
                                       dtype=value.dtype)).reshape_as(value)).mean()


@pytest.mark.parametrize("memory", ["K", "S"])
def test_tide_cross_128_window_full_backward_and_state(memory):
    torch.manual_seed(903)
    model = TideModel(ModelConfig(topology=TopologyConfig(n=1, layers=1), d=64,
                                  memory=memory, attention_chunk=32)).double()
    x = torch.randn(1, 131, 64, dtype=torch.float64, requires_grad=True)
    whole = model(x)
    a = model(x[:, :65]); b = model(x[:, 65:129], a.state); c = model(x[:, 129:], b.state)
    continued = torch.cat((a.readout(), b.readout(), c.readout()), 1)
    torch.testing.assert_close(continued, whole.readout(), atol=2e-10, rtol=2e-10)
    comparison = nested_comparison(c.state, whole.state, atol=2e-10, rtol=2e-10)
    assert comparison["within_threshold"] and not comparison["support_mismatches"]
    inputs = (x, *model.parameters())
    first = torch.autograd.grad(_loss(whole.readout()), inputs, retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(_loss(continued), inputs, allow_unused=True)
    for a, b in zip(first, second):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=3e-10, rtol=3e-9)
    if memory == "K":
        cache = c.state.memories[0]["K"]
        assert cache.times.dtype == torch.int64 and cache.times.shape[-1] == 128
        assert int(cache.times.min()) == 3*model.graph.period+1
        assert int(cache.times.max()) == 130*model.graph.period+1


def test_timed_pending_flush_chunked_full_gradients_and_int64_times():
    torch.manual_seed(42)
    model = TideModel(ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1,
                                  profile="compact-5-dd4", timing="timed"),
                                  d=64, memory="X", sharing="region_shared")).double()
    x = torch.randn(1, 5, 64, dtype=torch.float64, requires_grad=True)
    whole = model(x); whole_flush = model.flush(whole.state)
    a = model(x[:, :2]); b = model(x[:, 2:], a.state); flushed = model.flush(b.state)
    expected = torch.cat((whole.readout(), whole_flush.readout()), 1)
    continued = torch.cat((a.readout(), b.readout(), flushed.readout()), 1)
    assert any(bool((layer.strengths > 0).any()) for layer in a.state.pending)
    assert all(not bool((layer.strengths > 0).any()) for layer in flushed.state.pending)
    assert torch.equal(torch.cat((a.times, b.times, flushed.times)), torch.cat((whole.times, whole_flush.times)))
    torch.testing.assert_close(continued, expected, atol=1e-9, rtol=1e-8)
    comparison = nested_comparison(flushed.state, whole_flush.state, atol=1e-9, rtol=1e-8)
    assert comparison["within_threshold"] and not comparison["support_mismatches"]
    inputs = (x, *model.parameters())
    first = torch.autograd.grad(_loss(expected), inputs, retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(_loss(continued), inputs, allow_unused=True)
    for a, b in zip(first, second):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-7)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("family", ["tide", "transformer"])
def test_continuous_harness_full_training_boundary(precision, family):
    torch.manual_seed(42)
    model = (TideModel(ModelConfig(topology=TopologyConfig(n=2, layers=2), d=64)) if family == "tide"
             else TransformerBaseline(64, 2, window=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(2, 7, 64)
    y, state, measured = run_stream(model, x, 3, 2, mode="train", precision=precision, optimizer=optimizer)
    assert y.shape == x.shape and state.position == 7
    assert measured["input_positions"] == 14 and measured["output_slots"] == 14
    assert measured["first_backbone_output_seconds"] is not None
    assert len(measured["decode_seconds"]) == 4
    assert measured["backward_seconds"] > 0 and measured["optimizer_seconds"] > 0
    assert measured["complete_seconds"] >= measured["input_plus_flush_seconds"]
    assert measured["finite_output"] and measured["finite_parameters"] and measured["finite_gradients"]
    assert all(not tensor.requires_grad for tensor in _tensor_leaves(state))
    assert state_memory(state)["retained_storage_bytes"] >= state_memory(state)["payload_bytes"]


def test_training_state_checks_nonempty_slots_and_saturated_history():
    from tide import ClockConfig
    from benchmarks.continuous import training_state_observation
    model = TideModel(ModelConfig(topology=TopologyConfig(n=1, layers=1), d=64,
                                  clock=ClockConfig(kv_window=9)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(1, 5, 64)
    _, state, _ = run_stream(model, x, 5, 5, mode="train", optimizer=optimizer)
    _, state, _ = run_stream(model, x, 5, 5, mode="train", optimizer=optimizer, state=state)
    record = training_state_observation(model, optimizer, state)
    assert record["optimizer_step_min"] == record["optimizer_step_max"] == 2
    assert record["kv_retained_slot_min"] == record["kv_retained_slot_max"] == 3
    assert record["private_state_detached"] and record["private_state_finite"] and record["optimizer_finite"]
    next(iter(optimizer.state.values()))["exp_avg"].flatten()[0] = torch.nan
    assert not training_state_observation(model, optimizer, state)["optimizer_finite"]


def _tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _tensor_leaves(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_checkpoint_actual_update_reports_original_thresholds(tmp_path, precision):
    torch.manual_seed(12)
    model = TideModel(ModelConfig(topology=TopologyConfig(n=2, layers=2), d=64))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)
    x = torch.randn(2, 5, 64)
    state = None
    for _ in range(3):
        _, state, record = run_stream(model, x, 3, 2, state=state, mode="train", precision=precision,
                                     optimizer=optimizer)
        assert record["finite_parameters"]
    updated = {name: value.detach().clone() for name, value in model.named_parameters()}
    result = checkpoint_resume(model, optimizer, state, x, precision, tmp_path/"continued.pt")
    assert result["saved_data"]["exact"]
    assert result["next_forward"]["exact"] and result["next_state"]["exact"]
    assert result["state"] in {"within_threshold", "finite_deviation"}
    assert result["updated_parameters"]["atol"] == 1e-6
    assert result["updated_parameters"]["rtol"] == 2e-3
    assert any(not torch.equal(updated[name], value) for name, value in model.named_parameters())
    assert all(check["finite"] for check in result.values() if isinstance(check, dict) and "finite" in check)
    with pytest.raises(FileExistsError):
        checkpoint_resume(model, optimizer, state, x, precision, tmp_path/"continued.pt")


def test_numerical_comparison_does_not_treat_parameter_sign_as_message_support():
    from tide.kv import KVState
    a = {"weight": torch.tensor([-1e-9, 0.0, 1.0])}
    b = {"weight": torch.tensor([1e-9, 0.0, 1.0])}
    numeric = nested_comparison(a, b)
    assert numeric["within_threshold"] and not numeric["support_mismatches"]
    times = torch.arange(3, dtype=torch.int64).reshape(1, 1, 3)
    keys = torch.zeros(1, 1, 3, 1)
    absent = KVState(times, torch.tensor([[[0.0, 0.0, 1.0]]]), keys, keys, keys)
    positive = KVState(times, torch.tensor([[[1e-9, 0.0, 1.0]]]), keys, keys, keys)
    semantics = nested_comparison(absent, positive)
    assert semantics["within_threshold"] and semantics["support_mismatches"] == ["root.weight"]


@pytest.mark.parametrize("joint_class", ["A", "B", "C", "D"])
def test_continuous_joint_match_audits_exact_chunk_decode_schedule(joint_class, monkeypatch):
    from benchmarks.continuous import continuous_joint_match
    from benchmarks.matching import transformer_parameters
    calls = []
    original = TideModel._run
    def observed(model, x, state=None, **kwargs):
        calls.append((x.shape[0], x.shape[1], 0 if state is None else state.position, torch.is_grad_enabled()))
        return original(model, x, state, **kwargs)
    monkeypatch.setattr(TideModel, "_run", observed)
    config = ModelConfig(topology=TopologyConfig(n=2, layers=2), d=64)
    constructor, report = continuous_joint_match(config, joint_class, device="cpu", precision="fp32",
        mode="train", batch=2, prefill=3, prefill_chunk=2, decode=2, seed=82)
    assert calls == [(2, 2, 0, True), (2, 1, 2, True), (2, 1, 3, True), (2, 1, 4, True)]
    assert report["initial_audit"]["input_positions"] == 10
    assert report["initial_audit"]["sample_time_slots"] == 10
    assert report["audit_effective_mode"] == "grad_forward"
    assert report["audit_boundary"]["optimizer_updates"] == 0
    assert constructor["layers"] == (4 if joint_class in {"A", "B"} else 2)
    target = (report["initial_audit"]["parameters"]["unique_parameters"] if joint_class in {"A", "C"}
              else report["initial_audit"]["effective_touched"]["mean"])
    assert report["match"]["targets"]["parameters"] == target
    assert report["match"]["actual"]["parameters"] == transformer_parameters(
        constructor["d"], constructor["layers"], constructor["ffn_width"])


def test_continuous_joint_match_timed_flush_has_no_fictional_input_positions():
    from benchmarks.continuous import continuous_joint_match
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1,
                         profile="compact-5-dd4", timing="timed"), d=64, memory="X")
    _, report = continuous_joint_match(config, "D", device="cpu", precision="bf16", mode="inference",
                                       batch=1, prefill=3, prefill_chunk=2, decode=1, seed=83)
    assert report["audit_boundary"]["timed_flush"]
    assert report["initial_audit"]["input_positions"] == 4
    assert report["initial_audit"]["sample_time_slots"] > 4*TideModel(config).graph.period
    assert report["initial_audit"]["logical_axis"] == "tick"


def test_continuous_joint_cli_records_paired_prefix_and_actual_match(tmp_path, monkeypatch):
    import json
    import sys
    from benchmarks import continuous
    from benchmarks.workloads import paired_input, tensor_hash
    config = ModelConfig(topology=TopologyConfig(n=2, layers=2), d=64)
    path, output = tmp_path/"config.json", tmp_path/"matched.json"
    path.write_text(json.dumps(config.to_dict()))
    monkeypatch.setattr(sys, "argv", ["continuous", "--config", str(path), "--subject", "transformer",
        "--joint-match", "B", "--batch", "1", "--prefill", "2", "--prefill-chunk", "1", "--decode", "1",
        "--updates", "1", "--warmup", "0", "--seed", "84", "--transformer-chunk", "256", "--output", str(output)])
    assert continuous.main() == 0
    record = json.loads(output.read_text())
    assert record["comparison"]["kind"] == "joint" and record["comparison"]["joint_class"] == "B"
    prefix = record["comparison"]["shared_input_prefix_width"]
    expected = paired_input(1, 3, 64, 85)
    assert record["comparison"]["tide_input_sha256"] == tensor_hash(expected)
    assert record["comparison"]["shared_input_prefix_sha256"] == tensor_hash(expected[..., :prefix])
    assert record["parameters"] == record["comparison"]["match"]["actual"]["parameters"]
    assert record["spec"]["chunk"] == 256
    assert record["observations"][0]["input_positions"] == 3



@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_timed_restored_update_includes_flush_in_loss_and_backward(tmp_path, monkeypatch, precision):
    torch.manual_seed(39)
    model = TideModel(ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1,
                    profile="compact-5-dd4", timing="timed"), d=64, memory="X", sharing="region_shared"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(1, 3, 64)
    _, state, _ = run_stream(model, x, 2, 1, mode="train", precision=precision, optimizer=optimizer)
    gradient_norms = []
    original = TideModel.flush
    def observed(current, pending, **kwargs):
        result = original(current, pending, **kwargs)
        if torch.is_grad_enabled():
            result.values.register_hook(lambda gradient: gradient_norms.append(float(gradient.abs().sum())))
        return result
    monkeypatch.setattr(TideModel, "flush", observed)
    record = checkpoint_resume(model, optimizer, state, x, precision, tmp_path/"timed.pt")
    assert record["next_forward"]["exact"] and record["next_state"]["exact"]
    assert record["next_flush_result"]["exact"]
    assert len(gradient_norms) == 2 and all(value > 0 for value in gradient_norms)
    assert all(window["flush_output_slots"] > 0 and
               window["final_state_position"] > state.position+x.shape[1]
               for window in record["update_windows"])
    assert record["state"] in {"within_threshold", "finite_deviation"}
