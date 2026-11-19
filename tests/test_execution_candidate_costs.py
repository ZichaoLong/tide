"""Finite ordinary timings preserve candidate meaning and separate observers."""
import json
from dataclasses import replace

import pytest
import torch

import benchmarks.execution_candidates as costs
from benchmarks.execution_candidates import main, resolved_config, variant_options
from tide import ModelConfig, TopologyConfig


def config():
    return ModelConfig(topology=TopologyConfig(n=2, layers=1), d=64, memory="K",
        kv_width=32, ffn_width=192, region_batch_size=4, checkpoint_reception=True,
        parameter_indexing="static_view", attention_backend="math")


def test_execution_variants_only_override_the_two_declared_options():
    base = replace(config(), ffn_packing="power2", ffn_weight_access="grouped")
    expected = {"baseline": ("gather", "rectangular"), "weight_access": ("grouped", "rectangular"),
                "packing": ("gather", "power2"), "combined": ("grouped", "power2")}
    for name, (weight, packing) in expected.items():
        actual = resolved_config(base, name)
        assert actual == replace(base, ffn_weight_access=weight, ffn_packing=packing)
        assert variant_options(name) == dict(ffn_weight_access=weight, ffn_packing=packing)
        assert ModelConfig.from_dict(actual.to_dict()) == actual
    with pytest.raises(ValueError, match="unknown execution"):
        resolved_config(base, "unknown")


def test_ordinary_training_and_separate_work_and_storage_witness(tmp_path):
    path, output = tmp_path/"config.json", tmp_path/"costs.json"
    path.write_text(json.dumps(config().to_dict()))
    assert main(["--config", str(path), "--output", str(output), "--variant", "packing",
        "--batch", "1", "--length", "4", "--chunk", "2", "--warmup", "1", "--samples", "2",
        "--threads", "1", "--transformer-layers", "2", "--observe-work", "--memory-ledger"]) == 0
    record = json.loads(output.read_text())
    assert record["state"] == "completed_finite"
    assert [row["state_after"] for row in record["normal"]] == [8, 12]
    assert [row["continuation"]["optimizer_step_max"] for row in record["normal"]] == [2, 3]
    assert record["anchor"]["private_state_position"] == 4
    assert record["normal"][0]["phase_times"]["complete_seconds"] > 0
    assert record["normal_end_rss"]["process_lifetime_peak_rss_kib"] > 0
    assert record["diagnostics"]["work"]["state_position"] == 8
    assert record["diagnostics"]["work"]["actual_work"]["input_positions"] == 4
    ledger = record["diagnostics"]["memory_ledger"]
    assert ledger["state_position"] == 8 and ledger["finite"]
    assert ledger["phases"]["forward"]["unique_storage_union_bytes"] > 0
    assert set(ledger["phases"]["forward"]["categories"]) >= {
        "parameters", "optimizer", "gradients", "private_state", "saved_for_backward"}
    assert record["input_sha256"] and record["initial_parameters"]["sha256"]
    with pytest.raises(SystemExit):
        main(["--config", str(path), "--output", str(output)])


@pytest.mark.parametrize("subject", ["tide", "transformer"])
def test_first_backbone_and_independent_sequence_throughput_have_untimed_support_witness(tmp_path, subject):
    path, output = tmp_path/"config.json", tmp_path/"costs.json"
    path.write_text(json.dumps(config().to_dict()))
    assert main(["--config", str(path), "--output", str(output), "--subject", subject, "--mode", "inference",
        "--batch", "1", "--length", "4", "--chunk", "2", "--decode", "2", "--warmup", "1",
        "--samples", "1", "--threads", "1", "--transformer-layers", "1", "--transformer-d", "64",
        "--transformer-ffn-width", "128", "--transformer-head-dim", "16"]) == 0
    record = json.loads(output.read_text())
    assert record["normal_summary"]["first_backbone_output_seconds"] > 0
    assert record["normal_summary"]["stable_decode_positions_per_second"] > 0
    assert record["normal_summary"]["independent_complete_sequences_per_second"] > 0
    assert not record["normal"][0]["diagnostic_checks_in_timing"]
    assert record["diagnostics"]["availability"]["finite_outputs"]
    assert set(record["diagnostics"]) == {"availability"}
    assert record["normal_final_parameters"] == record["initial_parameters"]
    if subject == "transformer":
        assert record["spec"]["d"] == 64 and record["spec"]["ffn_width"] == 128
        with pytest.raises(SystemExit):
            main(["--config", str(path), "--output", str(tmp_path/"bad.json"),
                  "--subject", "transformer", "--variant", "combined"])


def test_explicit_large_transformer_training_refuses_before_real_weights(tmp_path, monkeypatch):
    path, output = tmp_path/"config.json", tmp_path/"refused.json"
    path.write_text(json.dumps(config().to_dict()))
    original, devices = costs.TransformerBaseline, []
    def checked_constructor(*args, **kwargs):
        device = torch.empty(()).device.type
        devices.append(device)
        assert device == "meta", "oversized Transformer must never allocate real weights"
        return original(*args, **kwargs)
    monkeypatch.setattr(costs, "TransformerBaseline", checked_constructor)
    with pytest.raises(SystemExit) as refusal:
        main(["--config", str(path), "--output", str(output), "--subject", "transformer", "--mode", "train",
              "--transformer-layers", "32", "--transformer-d", "4352", "--transformer-ffn-width", "15208",
              "--transformer-head-dim", "64", "--threads", "1"])
    assert refusal.value.code == 2 and devices == ["meta"]
    assert not output.exists()
