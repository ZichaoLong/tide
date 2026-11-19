"""Whole-request boundaries, bounded real state and actual joint-match audits."""
import copy
import gc
import json
import sys
import weakref

import pytest
import torch

from benchmarks import inference
from benchmarks.baselines import TransformerBaseline, TransformerState, parameter_count
from benchmarks.continuous import nested_comparison
from benchmarks.inference import InferenceSchedule, OutputDiagnostic, diagnose_request, run_request
from benchmarks.statistics import WorkAudit
from benchmarks.streaming_audit import FrozenParameterInventory, StreamingWorkAudit
from benchmarks.transformer_placement import LayerPlacedTransformer
from benchmarks.workloads import paired_input, parameter_hash
from tide import ModelConfig, TideModel, TopologyConfig, place_model
from tide.precision import precision_context


def tiny(timing="settle"):
    topology = (TopologyConfig(n=2, layers=2) if timing == "settle" else
                TopologyConfig(kind="multiscale", n=2, hubs=1, profile="compact-5-dd4", timing="timed"))
    return ModelConfig(topology=topology, d=64, memory="K" if timing == "settle" else "X")


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_layer_placed_transformer_uses_real_bounded_cache_and_original_positions(precision):
    torch.manual_seed(14)
    source = TransformerBaseline(32, 3, window=4, chunk=2)
    placed = LayerPlacedTransformer(copy.deepcopy(source), ["cpu:0", "cpu:1"], partitions=[0, 0, 1])
    x = torch.randn(2, 9, 32)
    with torch.no_grad(), precision_context("cpu", precision):
        whole, target = source(x, return_state=True)
        first, a = placed(x[:, :3], return_state=True)
        second, b = placed(x[:, 3:6], a, return_state=True)
        third, c = placed(x[:, 6:], b, return_state=True)
        ordinary, exact = source(x[:, :3], return_state=True)
        _, exact = source(x[:, 3:6], exact, return_state=True)
        _, exact = source(x[:, 6:], exact, return_state=True)
    torch.testing.assert_close(first, ordinary, atol=0, rtol=0)
    assert nested_comparison(c, exact, atol=0, rtol=0)["exact"]
    torch.testing.assert_close(torch.cat((first, second, third), 1), whole,
                               atol=2e-2 if precision == "bf16" else 1e-6, rtol=2e-2 if precision == "bf16" else 1e-5)
    assert c.position == 9 and a.position == 3
    assert all(t.shape == (2, 1, 3, 32) for t in (*c.keys, *c.values))
    assert all(t.untyped_storage().nbytes() == t.numel()*t.element_size() for t in (*c.keys, *c.values))
    assert placed.parameter_count == source.parameter_count
    assert len(placed._layouts) == 1  # Logical CPU groups have one real CPU device.
    with precision_context("cpu", precision):
        output, empty = placed(x[:, :0], return_state=True)
        assert output.shape == (2, 0, 32) and empty.position == 0
        assert len(empty.keys) == 3 and all(t.shape[-2] == 0 for t in empty.keys)
        _, same = placed(x[:, :0], c, return_state=True)
        assert same is c


def test_layer_placed_transformer_preserves_parameter_and_input_derivatives():
    torch.manual_seed(10)
    source = TransformerBaseline(32, 2, window=4, chunk=2).double()
    placed = LayerPlacedTransformer(copy.deepcopy(source), ["cpu", "cpu"])
    x = torch.randn(1, 5, 32, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    reference = source(x)
    a, state = placed(y[:, :3], return_state=True)
    b = placed(y[:, 3:], state)
    actual = torch.cat((a, b), 1)
    weights = torch.sin(torch.arange(actual.numel(), dtype=torch.float64)).reshape_as(actual)
    left = torch.autograd.grad((reference*weights).sum(), (x, *source.parameters()))
    right = torch.autograd.grad((actual*weights).sum(), (y, *placed.parameters()))
    for expected, observed in zip(left, right):
        torch.testing.assert_close(observed, expected, atol=1e-12, rtol=2e-11)
    with pytest.raises(ValueError, match="nonempty"):
        LayerPlacedTransformer(TransformerBaseline(32, 2), ["cpu"]*3)
    with pytest.raises(ValueError, match="contiguous"):
        LayerPlacedTransformer(TransformerBaseline(32, 3), ["cpu"]*2, partitions=[0, 1, 0])


def test_timed_request_preserves_pending_until_the_only_final_drain(monkeypatch):
    torch.manual_seed(5)
    model = TideModel(tiny("timed")).double()
    schedule = InferenceSchedule(3, 2, 3, 4, 3)
    x = torch.randn(1, schedule.positions, 64, dtype=torch.float64)
    expected = model(x)
    tail = model.flush(expected.state)
    expected_values = torch.cat((expected.readout(), tail.readout()), 1)
    calls, outputs, times = [], [], []
    original = model._run
    def recorded(value, state=None, **kwargs):
        calls.append((value.shape[1], 0 if state is None else state.position, kwargs.get("inject", True)))
        return original(value, state, **kwargs)
    monkeypatch.setattr(model, "_run", recorded)
    def consume(value, raw, phase, state):
        outputs.append(value.clone()); times.append(raw.times.clone())
    actual_state, measured = run_request(model, x, schedule, observer=consume, timing=False)
    assert calls[:-1] == [(2, 0, True), (1, 2, True), (1, 3, True), (1, 4, True), (1, 5, True), (3, 6, True), (1, 9, True)]
    assert calls[-1][1] == 10 and calls[-1][2] is False
    assert sum(not inject for _, _, inject in calls) == 1
    torch.testing.assert_close(torch.cat(outputs, 1), expected_values, atol=1e-9, rtol=1e-8)
    assert torch.equal(torch.cat(times), torch.cat((expected.times, tail.times)))
    assert nested_comparison(actual_state, tail.state, atol=1e-9, rtol=1e-8)["within_threshold"]
    assert measured["input_positions"] == 10
    assert measured["stable_decode"]["calls"] == 2
    assert measured["chunked_stream"]["input_positions"] == 4


def test_timing_syncs_all_owners_releases_outputs_and_counts_real_input(monkeypatch):
    model = torch.nn.Identity()
    model._placement_devices = (torch.device("cpu:0"), torch.device("cpu:1"))
    clock, refs, synced, starts = [0.0], [], [], []
    def forward(model, x, state):
        assert not refs or refs[-1]() is None
        starts.append(0 if state is None else state.position)
        value = x*2
        refs.append(weakref.ref(value))
        return value, TransformerState(starts[-1]+x.shape[1], (), ()), None
    def sync(device):
        synced.append(str(device)); clock[0] += .1
    monkeypatch.setattr(inference, "forward_state", forward)
    monkeypatch.setattr(inference, "synchronize", sync)
    monkeypatch.setattr(inference.time, "perf_counter", lambda: clock[0])
    schedule = InferenceSchedule(3, 2, 3, 5, 3)
    state, measured = run_request(model, torch.ones(2, 11, 4), schedule)
    assert all(reference() is None for reference in refs)
    assert state.position == 11 and starts == [0, 2, 3, 4, 5, 6, 9]
    assert measured["input_positions"] == 22 and measured["output_slots"] == 22
    assert measured["complete_seconds"] == pytest.approx(8*.3)
    assert measured["prompt_output_ready_seconds"] == pytest.approx(2*.3)
    assert measured["first_incremental_call_seconds"] == pytest.approx(.3)
    assert measured["stable_decode"]["input_positions"] == 4
    assert measured["synchronized_devices"] == ["cpu:0", "cpu:1", "cpu"]
    assert len(synced) == 3*(7+2)
    with pytest.raises(ValueError, match="outside normal timing"):
        run_request(model, torch.ones(2, 11, 4), schedule, observer=lambda *args: None)


@pytest.mark.parametrize("timing", ["settle", "timed"])
def test_streaming_audit_exactly_matches_full_workaudit_but_releases_events(timing):
    torch.manual_seed(26)
    model = TideModel(tiny(timing))
    x = paired_input(2, 7, 64, 27)
    schedule = InferenceSchedule(2, 1, 2, 3, 2)
    with WorkAudit(model) as detailed:
        run_request(model, x, schedule, timing=False)
    expected = detailed.report()
    with StreamingWorkAudit(model) as compact:
        run_request(model, x, schedule, timing=False)
    actual = compact.report()
    for key in ("parameters", "input_positions", "sample_time_slots", "logical_axis", "work", "work_per_input_position", "query_packing", "node_matrices"):
        assert actual[key] == expected[key]
    for key in ("effective_touched", "stages"):
        if key == "effective_touched":
            assert actual[key] == {k: v for k, v in expected[key].items() if k != "by_sample_time"}
        else:
            assert actual[key] == {stage: {k: v for k, v in row.items() if k != "by_sample_time"} for stage, row in expected[key].items()}
    assert not compact.events and not compact.domain and not compact.work_events
    assert actual["retention"]["peak_sample_time_slots"] < actual["sample_time_slots"]
    assert not actual["retention"]["per_time_rows_retained"]


@pytest.mark.parametrize("policy,batch_size", [("serial", 1), ("per_device", 1), ("per_device", 2)])
def test_frozen_inventory_survives_actual_placement_without_dense_parameter_owners(policy, batch_size):
    torch.manual_seed(91)
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-9-two-inputs", n=2),
                         d=64, memory="K", sharing="region_shared", aggregation="incoming_light_64",
                         region_batch_size=batch_size, reception_mask_padding=True, reception_qkv_backend="fused")
    model = TideModel(config)
    x, schedule = paired_input(1, 4, 64, 92), InferenceSchedule(2, 1, 2)
    expected, audit = diagnose_request(model, x, schedule)
    inventory = FrozenParameterInventory(model)
    old = weakref.ref(next(model.receive.parameters()))
    partitions = [(model.graph.nodes[nodes[0]].position+model.graph.nodes[nodes[0]].layer) % 2 for nodes in model.graph.regions]
    model = place_model(model, ["cpu:0", "cpu:1"], region_partitions=partitions, region_concurrency=policy)
    gc.collect()
    assert old() is None
    assert not hasattr(inventory, "model") and not hasattr(inventory, "parameters")
    actual, placed_audit = diagnose_request(model, x, schedule, inventory=inventory)
    for key in ("parameters", "effective_touched", "stages", "input_positions", "work"):
        assert placed_audit[key] == audit[key]
    # Ownership may split formerly compatible batches. Physical packing call
    # counts are intentionally separate from selected/executed rows and atoms.
    for kind, row in placed_audit["query_packing"].items():
        for field in ("selected_rows", "executed_rows"):
            assert row[field] == audit["query_packing"][kind][field]
        if batch_size == 1:
            assert row["calls"] == audit["query_packing"][kind]["calls"]
    assert actual["output_support_sha256"] == expected["output_support_sha256"]
    assert actual["execution"]["effective_region_concurrency"] == policy
    assert actual["finite_outputs"] and actual["finite_private_state"]


def test_fused_reception_audit_counts_unchanged_matrices_and_effective_atoms():
    torch.manual_seed(73)
    base = dict(topology=TopologyConfig(n=3, layers=2), d=64, aggregation="incoming_light_64")
    separate = TideModel(ModelConfig(**base))
    fused = TideModel(ModelConfig(**base, reception_qkv_backend="fused", reception_mask_padding=True))
    fused.load_state_dict(separate.state_dict())
    x, schedule = paired_input(2, 4, 64, 74), InferenceSchedule(2, 1, 2)
    _, first = diagnose_request(separate, x, schedule)
    _, second = diagnose_request(fused, x, schedule)
    assert first["effective_touched"] == second["effective_touched"]
    for name in ("receive.q", "receive.k", "receive.v"):
        assert first["node_matrices"][name] == second["node_matrices"][name]


def test_bounded_hash_matches_existing_parameter_identity():
    model = TideModel(tiny())
    record = inference.bounded_parameter_identity(model, chunk_bytes=400)
    assert record["sha256"] == parameter_hash(model) and record["finite"]
    with torch.no_grad():
        next(model.parameters()).flatten()[0] = float("nan")
    assert not inference.bounded_parameter_identity(model, chunk_bytes=400)["finite"]


def test_timed_prompt_readiness_uses_original_time_not_first_positive_output():
    torch.manual_seed(17)
    model = TideModel(tiny("timed"))
    schedule = InferenceSchedule(2, 1, 2)
    x = paired_input(1, 4, 64, 18)
    diagnostic, _ = diagnose_request(model, x, schedule)
    _, measured = run_request(model, x, schedule)
    inference.add_availability(measured, diagnostic)
    prompt_tick = model.graph.input_time(schedule.prefill-1)+len(model.layer_nodes)-1
    ready = next(row for row, observed in zip(measured["calls"], diagnostic["original_time_bounds"])
                 if observed["last_original_time"] >= prompt_tick)
    assert measured["prompt_output_ready_seconds"] == ready["ready_seconds"]
    assert measured["prompt_output_ready_seconds"] > measured["prompt_ingested_seconds"]
    assert measured["first_any_backbone_output_seconds"] <= measured["prompt_output_ready_seconds"]
    assert measured["drain_seconds"] > 0


def test_cli_freezes_all_joint_matches_and_reuses_reference_without_constructing_tide(tmp_path, monkeypatch):
    config = tiny()
    path, output, reference = tmp_path/"config.json", tmp_path/"tide.json", tmp_path/"reference.json"
    path.write_text(json.dumps(config.to_dict()))
    base = ["inference", "--config", str(path), "--prefill", "2", "--prefill-chunk", "1", "--decode", "2",
            "--stream-positions", "2", "--stream-chunk", "1", "--samples", "1", "--warmup", "1", "--threads", "2",
            "--config-overrides", '{"attention_chunk": 32}']
    starts = []
    original = TideModel._run
    def capture(model, x, state=None, **kwargs):
        starts.append(0 if state is None else state.position)
        return original(model, x, state, **kwargs)
    monkeypatch.setattr(TideModel, "_run", capture)
    monkeypatch.setattr(sys, "argv", base+["--output", str(output), "--reference-output", str(reference)])
    assert inference.main() == 0
    record = json.loads(output.read_text())
    subject = record["subjects"]["tide"]
    assert starts == [0, 1, 2, 3, 4, 5]*4  # cold, diagnostic, warmup, measured
    assert subject["cold"]["calls"][0]["state_before"] == 0
    assert subject["observations"][0]["calls"][0]["state_before"] == 0
    assert set(record["matching_reference"]["matches"]) == {"A", "B", "C", "D"}
    assert record["matching_reference"]["audit"]["input_positions"] == 6
    assert record["config"]["attention_chunk"] == record["matching_reference"]["config"]["attention_chunk"] == 32
    def forbidden(*args, **kwargs):
        raise AssertionError("frozen reference must not allocate a Tide model")
    monkeypatch.setattr(TideModel, "__init__", forbidden)
    compared = tmp_path/"matched.json"
    monkeypatch.setattr(sys, "argv", base+["--subjects", "A", "B", "C", "D", "--reference", str(reference),
                "--devices", "cpu:0,cpu:1", "--transformer-chunk", "256", "--output", str(compared)])
    assert inference.main() == 0
    result = json.loads(compared.read_text())
    for name, comparison in result["subjects"].items():
        assert comparison["state"] == "completed"
        assert comparison["parameters"] == record["matching_reference"]["matches"][name]["actual"]["parameters"]
        assert comparison["spec"]["chunk"] == 256
        assert comparison["diagnostics"]["finite_outputs"]
        assert comparison["observations"][0]["input_positions"] == 6
    corrupted = json.loads(reference.read_text())
    corrupted["batch"] = 42
    with pytest.raises(ValueError, match="digest mismatch"):
        inference.validate_reference(corrupted, config, InferenceSchedule(2, 1, 2, 2, 1), batch=1, seed=751,
                                     precision="fp32", identity=inference.source_identity())


def test_cli_retains_failed_diagnostic_and_completed_cold_request(tmp_path, monkeypatch):
    path, output = tmp_path/"config.json", tmp_path/"failed.json"
    path.write_text(json.dumps(tiny().to_dict()))
    original = TideModel.forward
    def nonfinite(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        result.values = result.values*float("nan")
        return result
    monkeypatch.setattr(TideModel, "forward", nonfinite)
    monkeypatch.setattr(sys, "argv", ["inference", "--config", str(path), "--prefill", "1", "--decode", "0",
                        "--audit", "none", "--samples", "1", "--warmup", "0", "--output", str(output)])
    with pytest.raises(ArithmeticError, match="nonfinite inference diagnostic"):
        inference.main()
    record = json.loads(output.read_text())
    assert record["state"] == "failed" and record["last_stage"] == "tide:diagnostic"
    assert record["subjects"]["tide"]["cold"]["complete_seconds"] > 0
    assert not record["subjects"]["tide"]["diagnostics"]["finite_outputs"]
