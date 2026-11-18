"""Independent bad-cache fixtures and real cross-window state transitions."""
import json
from types import SimpleNamespace
import sys

import pytest
import torch

from benchmarks.cache_metadata import (CacheMetadataDiagnostic, WriteClockLedger,
                                       inspect_kv_bank, inspect_ssd_bank)
from benchmarks.inference import InferenceSchedule, run_request
from tide import ClockConfig, ModelConfig, TideModel, TopologyConfig, place_model
from tide.kv import KVState
from tide.parameters import NodeOwners
from tide.ssd import ConvState


def kv(times, weight=None, dtype=torch.int64):
    times = torch.tensor(times, dtype=dtype)[None, None]
    shape = (*times.shape, 4)
    weight = torch.ones_like(times, dtype=torch.float32) if weight is None else torch.tensor(weight, dtype=torch.float32)[None, None]
    return KVState(times, weight, torch.zeros(shape), torch.zeros(shape), torch.zeros((*times.shape, 1)))


def inspect(bank, expected, *, window=16, processed=100, capacity=4):
    return inspect_kv_bank(bank, owners=(42,), batch=1, capacity=capacity, window=window,
                           processed_time=processed, expected={(42, 0): expected})


def test_kv_sparse_inactive_and_empty_owners_use_last_write_clock():
    result = inspect(kv([-100, 1, 7, 11], [0, 1, 1, 1]), (1, 7, 11))
    assert not result["failures"] and not result["rows"][0]["failures"]
    row = result["rows"][0]
    assert row["positive_entries"] == 3 and not row["full"]
    assert row["last_write_time"] == 11 and row["processed_through_time"] == 100
    assert row["maximum_age_from_last_write"] == 10  # Not global clock 100-1.
    empty = inspect(kv([2, 91], [0, 0]), ())
    assert empty["rows"][0]["empty"] and not empty["rows"][0]["failures"]


@pytest.mark.parametrize("times,expected,reason", [
    ([1, 7, 101], (1, 7, 11), "timestamp_outside_processed_clock"),
    ([5, 7, 21], (7, 21), "outside_last_write_window"),
    ([7, 1, 11], (1, 7, 11), "positive_times_not_strictly_increasing"),
    ([1, 7, 7], (1, 7, 11), "positive_times_not_strictly_increasing"),
    ([1, 7], (1, 7, 11), "retained_times_differ_from_positive_writes"),
])
def test_kv_inspector_detects_future_expired_unordered_duplicate_and_lost_writes(times, expected, reason):
    result = inspect(kv(times), expected)
    assert reason in result["rows"][0]["failures"]


def test_kv_dtype_capacity_and_int64_above_float_exact_range():
    assert "times_not_int64" in inspect(kv([1, 2], dtype=torch.float64), (1, 2))["failures"]
    assert "capacity_exceeded" in inspect(kv([1, 2, 3]), (1, 2, 3), capacity=2)["failures"]
    base = 2**53+301
    result = inspect(kv([base, base+1, base+7]), (base, base+1, base+7), processed=base+9)
    assert not result["failures"] and not result["rows"][0]["failures"]
    assert result["rows"][0]["first_positive_time"] == base
    assert result["rows"][0]["maximum_age_from_last_write"] == 7


@pytest.mark.parametrize("raw,times,reason", [
    (8, [-1, 4, 6, 8], None), (7, [-1, 4, 6, 8], "raw_time_differs_from_last_positive_reception"),
    (8, [-1, 1, 6, 8], "convolution_time_outside_last_write_horizon"),
    (8, [4, -1, 6, 8], "convolution_padding_or_time_order"),
])
def test_ssd_raw_clock_and_convolution_original_time_contract(raw, times, reason):
    bank = SimpleNamespace(times=torch.tensor([[raw]], dtype=torch.int64),
        conv=ConvState(torch.tensor(times, dtype=torch.int64)[None, None], torch.zeros(1, 1, len(times), 6)))
    result = inspect_ssd_bank(bank, owners=(12,), batch=1, stride=2, processed_time=80,
                              expected={(12, 0): (4, 6, 8)}, last_writes={(12, 0): 8})
    assert not result["failures"]
    if reason is None:
        assert not result["rows"][0]["failures"]
    else:
        assert reason in result["rows"][0]["failures"]


def small_config(kind):
    if kind == "fc":
        return ModelConfig(topology=TopologyConfig(n=1, layers=2), d=64, clock=ClockConfig(kv_window=16))
    timed = kind != "settle"
    return ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1, profile="compact-5-dd4",
                    timing="timed" if timed else "settle", period=1 if kind == "period1" else None),
                    d=64, memory="X", sharing="region_shared", region_batch_size=2,
                    clock=ClockConfig(kv_window=4 if kind == "period1" else 8 if timed else 16))


@pytest.mark.parametrize("kind", ["fc", "settle", "timed", "period1"])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_real_bounded_ledger_tracks_fill_eviction_raw_times_and_final_drain(kind, precision):
    torch.manual_seed(112)
    model = TideModel(small_config(kind))
    if kind == "period1":
        partitions = [(model.graph.nodes[nodes[0]].position+model.graph.nodes[nodes[0]].layer) % 2 for nodes in model.graph.regions]
        model = place_model(model, ["cpu:0", "cpu:1"], region_partitions=partitions, region_concurrency="per_device")
    schedule = InferenceSchedule(4, 2, 6)
    observer = CacheMetadataDiagnostic(model, schedule, 1, sample_every=2)
    with observer.ledger:
        state, boundary = run_request(model, torch.randn(1, 10, 64), schedule, precision=precision, observer=observer, timing=False)
    report = observer.report(state, require_kv_eviction=True)
    assert report["passed"], report
    assert report["kv_owner_samples_with_verified_eviction"] > 0
    assert report["output_time_checks"] == len(boundary["calls"])
    assert report["output_time_failure_count"] == 0
    assert all("prepare" not in vars(memory) for memory in model.memories.values())
    if kind == "fc":
        assert report["kv_owner_samples_observed"] == report["kv_owner_samples_ever_full"] == 2
        assert report["maximum_retained_oracle_timestamps"] == 8
    if not model.settle:
        final = report["snapshots"][-1]
        assert final["phase"] == "drain"
        assert all(not bank["outgoing_positive_pending"] for bank in final["pending"]["banks"])
    for snapshot in report["snapshots"]:
        for witness in snapshot["evictions"]:
            assert witness["current_first_time"] > witness["previous_first_time"]
            assert witness["old_first_age_now"] >= witness["window"]
            assert witness["retained_write_set_verified"]


def test_prepare_write_ledger_rejects_nonmonotonic_original_inputs():
    model = TideModel(small_config("fc"))
    ledger = WriteClockLedger(model, 1)
    ledger.observe("K", torch.ones(1, 1, 2), torch.tensor([[[3, 2]]]), NodeOwners.from_static((0,), device="cpu"))
    assert ledger.failure_count == 1


def test_cli_persists_actual_eviction_and_clock_checks(tmp_path, monkeypatch):
    from scripts import check_inference_cache
    config, output = tmp_path/"model.json", tmp_path/"cache.json"
    config.write_text(json.dumps(small_config("fc").to_dict()))
    monkeypatch.setattr(sys, "argv", ["check_inference_cache", "--config", str(config), "--prefill", "4",
        "--prefill-chunk", "2", "--decode", "6", "--sample-every", "2", "--threads", "2",
        "--require-kv-eviction", "--output", str(output)])
    assert check_inference_cache.main() == 0
    record = json.loads(output.read_text())
    assert record["state"] == "passed" and record["diagnosis"]["kv_owner_samples_with_verified_eviction"] == 2
    assert record["diagnosis"]["positive_writes"] == 20
    assert sum(call["input_positions"] for call in record["actual_calls"]) == 10
    assert not any("seconds" in call for call in record["actual_calls"])


def test_cli_keeps_a_bad_timestamp_as_a_failed_check(tmp_path, monkeypatch):
    from scripts import check_inference_cache
    config, output = tmp_path/"model.json", tmp_path/"bad-cache.json"
    config.write_text(json.dumps(small_config("fc").to_dict()))
    original = TideModel.forward
    def corrupt(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        result.state.memories[0]["K"].times.add_(1000)
        return result
    monkeypatch.setattr(TideModel, "forward", corrupt)
    monkeypatch.setattr(sys, "argv", ["check_inference_cache", "--config", str(config), "--prefill", "1",
        "--prefill-chunk", "1", "--decode", "0", "--threads", "2", "--output", str(output)])
    assert check_inference_cache.main() == 1
    record = json.loads(output.read_text())
    assert record["state"] == "check_failed"
    failures = record["diagnosis"]["structural_failures"]
    assert any("timestamp_outside_processed_clock" in row.get("reasons", []) for row in failures)
