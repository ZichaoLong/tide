"""Full-request diagnostic replay, timing boundaries and sampled refusal."""
import hashlib
import json
import statistics

import pytest
import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode

from benchmarks import inference_breakdown as breakdown
from benchmarks.baselines import TransformerBaseline
from benchmarks.inference import InferenceSchedule, OutputDiagnostic, run_request
from benchmarks.workloads import paired_input, tensor_hash
from tide import ClockConfig, ModelConfig, TideModel, TopologyConfig


def _arguments(tmp_path, subject, output_name="breakdown.json"):
    config = ModelConfig(topology=TopologyConfig(kind="fc", n=2, layers=2), d=64,
                         kv_width=32, clock=ClockConfig(kv_window=8))
    args = ["--subject", subject, "--output", str(tmp_path/output_name),
            "--batch", "2", "--prefill", "4", "--prefill-chunk", "2",
            "--decode", "3", "--threads", "1", "--warmup", "1", "--samples", "2"]
    if subject == "tide":
        cfg = tmp_path/"config.json"
        cfg.write_text(json.dumps(config.to_dict()))
        args += ["--config", str(cfg)]
    else:
        args += ["--transformer-d", "64", "--transformer-ffn-width", "128",
                 "--transformer-layers", "2", "--transformer-head-dim", "32",
                 "--transformer-window", "4", "--transformer-chunk", "2"]
    return args, config


@pytest.mark.parametrize("subject", ["tide", "transformer"])
def test_full_prefill_decode_replays_and_checks_are_outside_capture(tmp_path, monkeypatch, subject):
    observers, normal_calls, checked = [], [], []

    def factory(cls):
        def create(device):
            observer = cls(device)
            observers.append(observer)
            return observer
        return create

    monkeypatch.setattr(breakdown, "DetailedStageTimes", factory(breakdown.DetailedStageTimes))
    monkeypatch.setattr(breakdown, "OperatorTimes", factory(breakdown.OperatorTimes))

    def outside_capture():
        assert not any(observer.stack for observer in observers)
        assert _get_current_dispatch_mode() is None

    original_state = breakdown.private_state_metadata
    original_output = breakdown.output_metadata
    original_request = breakdown.run_request

    def state_check(*args):
        outside_capture()
        checked.append("state")
        return original_state(*args)

    def output_check(*args):
        outside_capture()
        checked.append("output")
        return original_output(*args)

    def normal_request(*args, **kwargs):
        outside_capture()
        assert kwargs.get("observer") is None and kwargs.get("timing", True)
        normal_calls.append(1)
        return original_request(*args, **kwargs)

    monkeypatch.setattr(breakdown, "private_state_metadata", state_check)
    monkeypatch.setattr(breakdown, "output_metadata", output_check)
    monkeypatch.setattr(breakdown, "run_request", normal_request)
    args, config = _arguments(tmp_path, subject)
    assert breakdown.main([*args, "--diagnostics", "stages,operators"]) == 0
    record = json.loads((tmp_path/"breakdown.json").read_text())
    assert record["state"] == "completed_finite"
    assert len(normal_calls) == 3  # One warmup and two ordinary samples only.
    assert checked.count("output") == 10 and checked.count("state") == 12
    assert record["initial_parameters"] == record["final_parameters"]
    assert record["replay_checks"]["independent_diagnostic_metadata_identical"]
    assert record["replay_checks"]["output_identity"]["exact_hashes_identical"]
    stages, operators = (record["diagnostics"][kind] for kind in ("stages", "operators"))
    for diagnostic in (stages, operators):
        assert diagnostic["capture_start_position"] == 0
        assert not diagnostic["normal_timing"] and not diagnostic["throughput_reported"]
        assert diagnostic["initial_parameters"] == record["initial_parameters"]
        assert diagnostic["input_sha256"] == record["input_sha256"]
        assert [row["phase"] for row in diagnostic["calls"]] == ["prefill"]*2+["decode"]*3
        assert [row["state_before"] for row in diagnostic["calls"]] == [0, 2, 4, 5, 6]
        assert [row["state_after"] for row in diagnostic["calls"]] == [2, 4, 5, 6, 7]
        assert all(row["output"]["finite_through_call"] and row["private_state"]["finite"]
                   and row["private_state"]["detached"] for row in diagnostic["calls"])
        assert all(len(row["output"]["identities"]["readout"]["sha256"]) == 64
                   for row in diagnostic["calls"])
    phase_rows = {row["path"]: row for row in stages["report"]["rows"]}
    assert phase_rows["prefill"]["calls"] == 2
    assert phase_rows["decode"]["calls"] == 3
    assert any(row["path"].startswith("prefill/call_") for row in operators["report"]["rows"])
    assert any(row["path"].startswith("decode/call_") for row in operators["report"]["rows"])
    assert all(row["path"].startswith(("prefill/", "decode/")) for row in operators["report"]["rows"])
    assert operators["report"]["dispatched_calls"] > 0

    torch.manual_seed(751)
    reference = (TideModel(config) if subject == "tide" else TransformerBaseline(
        d=64, layers=2, ffn_width=128, head_dim=32, window=4, chunk=2)).eval()
    output = OutputDiagnostic()
    state, _ = run_request(reference, paired_input(2, 7, 64, 752), InferenceSchedule(4, 2, 3),
                           observer=output, timing=False)
    assert stages["outputs"] == operators["outputs"] == output.report(state)
    assert stages["final_state"] == operators["final_state"] == original_state(reference, state)
    if subject == "tide":
        assert stages["final_state"]["times_int64"]
        assert all(call["output"]["time"]["dtype"] == "torch.int64" for call in stages["calls"])
    else:
        assert stages["final_state"]["times_int64"] is None
        assert stages["final_state"]["cache_position_bounds"] == [
            dict(layer=0, start=4, stop=7), dict(layer=1, start=4, stop=7)]
        assert all(call["output"]["support_sha256"] is None for call in stages["calls"])
    normal = record["normal"]
    assert len(normal["observations"]) == 2
    assert all(row["final_state_position"] == 7 and row["input_positions"] == 14
               and not row["diagnostic_checks_in_timing"] for row in normal["observations"])
    expected = statistics.median(2/row["complete_seconds"] for row in normal["observations"])
    assert normal["summary"]["independent_complete_sequences_per_second"] == expected


def test_complete_output_identity_handles_noncontiguous_and_empty_payloads():
    value = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5).transpose(1, 3)
    for tensor in (value, value[..., :0], value.contiguous()):
        identity = breakdown.output_identity(tensor, chunk_bytes=16)
        expected = tensor_hash(tensor) if tensor.numel() else hashlib.sha256(b"").hexdigest()
        assert identity["sha256"] == expected
        assert identity["logical_bytes"] == tensor.numel()*tensor.element_size()


def test_finite_output_difference_is_retained_without_equivalence_claim(tmp_path, monkeypatch):
    args, _ = _arguments(tmp_path, "transformer", "finite-difference.json")
    original = breakdown.forward_state

    def changed_operator_readout(*args, **kwargs):
        value, state, raw = original(*args, **kwargs)
        if isinstance(_get_current_dispatch_mode(), breakdown.OperatorTimes):
            value = value + 0.125
        return value, state, raw

    monkeypatch.setattr(breakdown, "forward_state", changed_operator_readout)
    assert breakdown.main(args) == 0
    record = json.loads((tmp_path/"finite-difference.json").read_text())
    checks = record["replay_checks"]
    assert checks["independent_diagnostic_metadata_identical"]
    assert not checks["output_identity"]["exact_hashes_identical"]
    assert len(checks["output_identity"]["differing_calls"]) == 5
    assert all(row["outputs"]["finite_outputs"] for row in record["diagnostics"].values())
    assert "not tolerance/VJP qualification" in checks["output_identity"]["scope"]


def test_single_stage_pass_and_existing_output_are_kept_separate(tmp_path, monkeypatch):
    args, _ = _arguments(tmp_path, "transformer")

    def no_operator_pass(*args, **kwargs):
        pytest.fail("unrequested operator pass was constructed")

    monkeypatch.setattr(breakdown, "OperatorTimes", no_operator_pass)
    assert breakdown.main([*args, "--diagnostics", "stages"]) == 0
    path = tmp_path/"breakdown.json"
    original = path.read_bytes()
    record = json.loads(original)
    assert list(record["diagnostics"]) == ["stages"]
    assert record["replay_checks"] == {}
    with pytest.raises(SystemExit):
        breakdown.main([*args, "--diagnostics", "stages"])
    assert path.read_bytes() == original


def test_reserved_budget_refusal_is_durable_before_normal_or_diagnostics(tmp_path, monkeypatch):
    args, _ = _arguments(tmp_path, "tide", "refusal.json")
    monkeypatch.setattr(breakdown, "allocator_stats", lambda *args, **kwargs:
        {"cpu": dict(allocated_bytes=0, reserved_bytes=2**30,
                     peak_allocated_bytes=0, peak_reserved_bytes=2**30)})
    assert breakdown.main([*args, "--reserved-budget-gib", "0.5"]) == 1
    record = json.loads((tmp_path/"refusal.json").read_text())
    assert record["state"] == "failed" and record["failure_kind"] == "sampled_budget_refusal"
    assert "reserved" in record["error"]
    assert not record["diagnostics"] and not record["normal"]["observations"]
