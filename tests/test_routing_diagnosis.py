"""Scalar routing capture, exact metadata and independent boundary witnesses."""
from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from tide import ModelConfig, TideModel, TopologyConfig
from benchmarks.workloads import parameter_hash
from scripts.diagnose_routing_divergence import (CaptureBudget, CaptureLimit, RoutingTrace,
    allocation_payload, compare_traces, difference, main, replay_allocation)
from scripts.qualify_precision import THRESHOLDS


def small_config(**options):
    return ModelConfig(topology=TopologyConfig(kind="lattice", n=2, depth=1),
                       d=64, attention_backend="math", **options)


def synthetic_trace(control, time):
    row = dict(nodes=(17, 21), group=(2,), layer=3, time=torch.tensor([time], dtype=torch.int64),
               fixed=False, received=torch.ones(2, 1, 1), raw_score=torch.tensor([1., 0.]).reshape(2, 1, 1),
               score=torch.tensor([1., 0.]).reshape(2, 1, 1), control=torch.tensor(control).reshape(2, 1, 1))
    return SimpleNamespace(entries={(0, 2): row}, calls=[dict(index=0, phase="forward")])


def test_exact_support_is_separate_from_value_tolerance_and_keeps_large_int64():
    time = 2**55 + 17
    left, right = synthetic_trace([1., 0.], time), synthetic_trace([1., 1e-8], time)
    result = compare_traces(left, right, THRESHOLDS["fp32"])
    control = next(row for row in result["stages"] if row["stage"] == "control")
    assert control["difference"]["threshold_failures"] == 0
    assert result["support_differences"] == 1
    assert result["witnesses"] == [dict(call=0, region=2, layer=3, node=21, local_node=1,
                                       sample=0, time_index=0, original_time=time)]
    right.entries[0, 2]["time"] += 1
    result = compare_traces(left, right, THRESHOLDS["fp32"])
    metadata = result["metadata_failures"][0]
    assert metadata["field"] == "original_int64_time"
    assert metadata["difference"]["different"] == 1
    assert "max_abs" not in metadata["difference"]  # no lossy FP64 clock conversion


def test_decimal_allocation_witness_exposes_wrong_support_below_tolerance():
    payload = dict(group=(2,), selected_index=0, fixed=False,
        score=torch.tensor([1., 0.]).reshape(1, 2, 1, 1),
        received=torch.ones(1, 2, 1, 1),
        control=torch.tensor([1., 1e-8]).reshape(1, 2, 1, 1))
    result = replay_allocation(payload, [False, False], 1, "quad", "group", sample=0, position=0)
    assert result["same_shape_replay"]["threshold_failures"] == 0
    assert not result["same_shape_replay"]["exact"]
    assert not result["support_matches_decimal"]
    assert result["cpu_fp64_support_matches_decimal"]
    assert result["independent_decimal"]["control"] == [1., 0.]
    assert result["independent_decimal"]["margin"] == [1., 0.]


def test_grouped_replay_keeps_region_axis_and_zero_reception_constraints():
    # Region 0's high-scoring node has no reception and cannot consume mass.
    # Region 1 needs an independent +1/4 shift on its two remaining scores.
    scores = torch.tensor([[100., .6, .4], [-1., .2, .3]])[:, :, None, None].expand(2, 3, 2, 2).clone()
    received = torch.ones_like(scores); received[0, 0] = 0
    controls = torch.tensor([[0., .6, .4], [0., .45, .55]])[:, :, None, None].expand_as(scores).clone()
    payload = dict(group=(4, 9), selected_index=0, fixed=False, score=scores, received=received, control=controls)
    first = replay_allocation(payload, [False]*3, 1, "quad", "group", sample=1, position=1)
    assert first["original_submission_shape"] == [3, 4, 2]
    assert first["single_region_shape"] == [3, 2, 2]
    assert first["independent_decimal"]["control"][0] == 0
    assert first["support_matches_decimal"]
    assert first["same_shape_replay"]["threshold_failures"] == 0
    assert first["grouped_vs_single_support_differences"] == 0
    payload["selected_index"] = 1
    second = replay_allocation(payload, [False]*3, 1, "quad", "group", sample=0, position=0)
    torch.testing.assert_close(torch.tensor(second["independent_decimal"]["control"]),
                               torch.tensor([0., .45, .55]), atol=1e-7, rtol=0)
    assert second["support_matches_decimal"]


@pytest.mark.parametrize("checkpoint", [False, True])
def test_capture_is_transparent_to_continued_forward_and_parameter_input_vjp(checkpoint):
    torch.manual_seed(915)
    model = TideModel(small_config(region_batch_size=4, checkpoint_regions=checkpoint,
                                  memory_sharing="region_shared"))
    before = parameter_hash(model)
    value = torch.randn(2, 3, 64)

    def execute(capture):
        model.zero_grad(set_to_none=True)
        x = value.clone().requires_grad_()
        trace = RoutingTrace(model)
        if capture:
            with trace.install():
                a = model(x[:, :2])
                b = model(x[:, 2:], a.state)
            trace.validate()
        else:
            a = model(x[:, :2]); b = model(x[:, 2:], a.state)
        loss = a.readout().square().sum() + b.readout().square().sum()
        # Backward intentionally occurs after hooks are removed. Checkpoint
        # closures retained during capture must also be transparent now.
        loss.backward()
        gradients = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
        return (a, b), x.grad, gradients, trace

    baseline, dx0, dp0, _ = execute(False)
    captured, dx1, dp1, trace = execute(True)
    assert parameter_hash(model) == before
    assert torch.equal(dx0, dx1)
    assert dp0.keys() == dp1.keys()
    for name, value0 in dp0.items():
        assert dp1[name] is None if value0 is None else torch.equal(value0, dp1[name]), name
    assert any(len(row["group"]) > 1 for row in trace.entries.values())
    for a, b in zip(baseline, captured):
        assert torch.equal(a.values, b.values)
        assert torch.equal(a.strengths, b.strengths)
        assert torch.equal(a.times, b.times)
    for (call, region), row in trace.entries.items():
        assert row["time"].dtype == torch.int64
        assert not row["control"].requires_grad
        assert row["control"].shape == row["received"].shape
        if region in model.layer_regions[-1]:
            output_nodes = list(model.layer_nodes[-1])
            indices = [output_nodes.index(node) for node in row["nodes"]]
            assert torch.equal(row["control"], captured[call].strengths[indices])
    assert trace.calls[1]["state_position"] == 2


def test_serial_batched_scalar_layout_and_payload_keep_original_region_nodes():
    torch.manual_seed(916)
    model = TideModel(small_config())
    value = torch.randn(2, 3, 64)
    traces = []
    budget = CaptureBudget(1 << 20)
    for size in (1, 4):
        model.config = replace(model.config, region_batch_size=size)
        trace = RoutingTrace(model, budget)
        with torch.no_grad(), trace.install():
            model(value)
        trace.validate(); traces.append(trace)
    compared = compare_traces(*traces, THRESHOLDS["fp32"])
    assert not compared["metadata_failures"]
    assert traces[0].entries.keys() == traces[1].entries.keys()
    key = next(key for key, row in traces[1].entries.items() if len(row["group"]) > 1 and not row["fixed"])
    payload = allocation_payload(traces[1], key)
    assert payload["selected_index"] == payload["group"].index(key[1])
    assert torch.equal(payload["score"][payload["selected_index"]], traces[1].entries[key]["score"])
    assert budget.used <= budget.maximum


def test_capture_bound_restores_hooks_and_is_charged_before_host_copy():
    model = TideModel(small_config(region_batch_size=4))
    methods = {name: getattr(model, name) for name in ("_run", "_receive", "_evaluate_region", "_score_regions")}
    batch_module = importlib.import_module("tide.region_batch")
    original_batch = batch_module.evaluate_regions
    budget = CaptureBudget(1)
    trace = RoutingTrace(model, budget)
    value = torch.randn(1, 1, 64)
    with torch.no_grad(), pytest.raises(CaptureLimit), trace.install():
        model(value)
    assert budget.used == 0
    assert trace.current == ()
    assert all(getattr(model, name) == function for name, function in methods.items())
    assert batch_module.evaluate_regions is original_batch
    with torch.no_grad():
        assert torch.isfinite(model(value).values).all()


def test_fixed_region_zero_boundary_and_float_nonfinite_are_not_hidden():
    payload = dict(group=(0,), selected_index=0, fixed=True,
        received=torch.tensor([0., .125, .5]).reshape(1, 3, 1, 1),
        control=torch.tensor([0., .25, 1.]).reshape(1, 3, 1, 1))
    result = replay_allocation(payload, [False]*3, 1, "quad", "group", sample=0, position=0)
    assert result["independent"] == [0., .25, 1.]
    assert result["support_matches_independent"]
    numerical = difference(torch.tensor([0.]), torch.tensor([float("nan")]), THRESHOLDS["fp32"])
    assert not numerical["finite"] and numerical["threshold_failures"] == 1


def test_small_cli_has_two_forward_paths_and_bounded_new_outputs(tmp_path):
    config = tmp_path/"config.json"; config.write_text(json.dumps(small_config().to_dict()))
    output = tmp_path/"routing.json"
    record = main(["--config", str(config), "--device", "cpu", "--batch", "1", "--prefill", "2",
                   "--prefill-chunk", "1", "--decode", "1", "--threads", "1", "--output", str(output)])
    assert record["state"] == "completed"
    assert record["historical_witness_reproduced"] is None
    assert [row["completed_calls"] for row in record["paths"]] == [3, 3]
    assert record["captured_bytes"] < 1 << 20
    assert record["thresholds"] == THRESHOLDS["fp32"]
    assert all(row["output"]["final_state_position"] == 3 for row in record["paths"])
    assert not record["comparison"]["metadata_failures"]
    assert json.loads(output.read_text())["state"] == "completed"
    with pytest.raises(SystemExit):
        main(["--config", str(config), "--output", str(output)])


def test_historical_sha_rejection_precedes_any_model_request(tmp_path, monkeypatch):
    config = small_config()
    config_path = tmp_path/"config.json"; config_path.write_text(json.dumps(config.to_dict()))
    histories = []
    for size in (1, 4):
        path = tmp_path/f"old-{size}.json"
        path.write_text(json.dumps(dict(source={"source_commit": "old"},
            schedule=dict(prefill=1, prefill_chunk=1, decode_steps=0, stream_positions=0, stream_chunk=128),
            matching_reference={"seed": 751}, precision={"name": "fp32"}, subjects={"tide": dict(
                spec=replace(config, region_batch_size=size).to_dict(), input_shape=[1, 1, 64],
                initial_parameters={"sha256": "wrong-parameters"}, input_sha256="wrong-input",
                diagnostics=dict(output_support_sha256="old-support", original_output_times_sha256="old-time"))})))
        histories.append(str(path))
    def forbidden(*args, **kwargs):
        raise AssertionError("identity mismatch must prevent model execution")
    monkeypatch.setattr(TideModel, "_run", forbidden)
    output = tmp_path/"reject.json"
    with pytest.raises(ValueError, match="historical parameter/input SHA mismatch"):
        main(["--config", str(config_path), "--batch", "1", "--prefill", "1", "--prefill-chunk", "1",
              "--decode", "0", "--threads", "1", "--historical-results", *histories,
              "--output", str(output)])
    record = json.loads(output.read_text())
    assert record["state"] == "execution_failed" and not record["paths"]
