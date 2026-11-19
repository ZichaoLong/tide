"""Actual history replay and sampled resource refusal for decode diagnosis."""
import json

import torch

from benchmarks.inference import InferenceSchedule, OutputDiagnostic, run_request
from benchmarks.memory_decode_costs import main, state_metadata
from benchmarks.workloads import paired_input
from tide import ClockConfig, ModelConfig, TideModel, TopologyConfig


def test_cli_replays_real_eviction_and_keeps_prefill_checks_outside_capture(tmp_path):
    config = ModelConfig(topology=TopologyConfig(kind="fc", n=2, layers=2), d=64,
                         kv_width=32, clock=ClockConfig(kv_window=8))
    cfg, output = tmp_path/"config.json", tmp_path/"decode.json"
    cfg.write_text(json.dumps(config.to_dict()))
    assert main(["--config", str(cfg), "--output", str(output), "--batch", "2",
                 "--prefill", "4", "--prefill-chunk", "2", "--decode", "3", "--threads", "1"]) == 0
    record = json.loads(output.read_text())
    assert not record["normal_timing"] and not record["throughput_reported"]
    torch.manual_seed(751)
    model = TideModel(config).eval()
    inputs = paired_input(2, 7, 64, 752)
    reference_output = OutputDiagnostic()
    state, _ = run_request(model, inputs, InferenceSchedule(4, 2, 3),
                           observer=reference_output, timing=False)
    expected = state_metadata(state)
    stages, runtime = (record["diagnostics"][key] for key in ("stages", "runtime"))
    assert stages["before_decode"] == runtime["before_decode"]
    assert stages["after_decode"] == runtime["after_decode"] == expected
    assert stages["initial_parameters"] == runtime["initial_parameters"] == record["final_parameters"]
    assert stages["input_sha256"] == runtime["input_sha256"] == record["input_sha256"]
    for observed in (stages, runtime):
        assert observed["capture_start_position"] == 4
        assert [row["state_after"] for row in observed["calls"]] == [5, 6, 7]
        assert observed["outputs"]["finite_outputs"] and observed["after_decode"]["times_int64"]
        assert observed["after_decode"]["kv_history"][0]["first_positive_times"][0] > observed["before_decode"]["kv_history"][0]["first_positive_times"][0]
    assert next(row for row in stages["report"]["rows"] if row["path"] == "decode")["calls"] == 3
    assert all(row["path"].startswith("decode") for row in runtime["report"]["operators"])
    assert any(row["operator"] == "aten.cat.default" and "memory.K.prepare" in row["path"]
               for row in runtime["report"]["materializations"])
    assert all("prefill" not in row["path"] for row in runtime["report"]["matrix_groups"])


def test_cli_preserves_sampled_reserved_refusal_before_decode(tmp_path, monkeypatch):
    import benchmarks.memory_decode_costs as diagnosis
    config = ModelConfig(topology=TopologyConfig(kind="fc", n=1, layers=1), d=64)
    cfg, output = tmp_path/"config.json", tmp_path/"refusal.json"
    cfg.write_text(json.dumps(config.to_dict()))
    monkeypatch.setattr(diagnosis, "allocator_stats", lambda *args, **kwargs:
        {"cpu": dict(allocated_bytes=0, reserved_bytes=2**30, peak_allocated_bytes=0, peak_reserved_bytes=2**30)})
    assert main(["--config", str(cfg), "--output", str(output), "--reserved-budget-gib", "0.5",
                 "--batch", "1", "--prefill", "2", "--decode", "1", "--threads", "1"]) == 1
    record = json.loads(output.read_text())
    assert record["state"] == "failed" and record["failure_kind"] == "sampled_budget_refusal"
    assert "reserved" in record["error"] and not record["diagnostics"]
