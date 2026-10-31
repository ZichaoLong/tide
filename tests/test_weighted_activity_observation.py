"""Actual complete-return activity and optional observer lifecycle."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from tidegraph.records import Atom, Continuation, Result, State
from test_packing_observation import DispatchCounts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from weighted_benchmark_activity import activity
from weighted_benchmark_work import observe_work


def test_boundary_activity_is_hand_counted_without_tensor_operations():
    value = torch.zeros(3)
    first = Atom(0, 1, 2, 1, 0, 1, value)
    second = Atom(1, 1, 4, 1, 1, 2, value)
    state = lambda length: State(value, slots={"times": torch.arange(length, dtype=torch.int64)})
    continuation = Continuation("fixture", 3, cut=5,
        states={(0, 0): state(2), (0, 1): state(0), (1, 0): state(1),
                (1, 1): State(value, slots={"H": torch.zeros(1)})}, pending=[second])
    result = Result(continuation,
        [dict(node=0, batch=0, active=True, fiber=[first]), dict(node=1, batch=0, active=False, fiber=[first, first]),
         dict(node=0, batch=1, active=True, fiber=[second])], [(0, 2, 0, value)], [first, first, second], {})
    with DispatchCounts() as operations:
        observed = activity(result)
    assert not operations.counts
    rows = observed["samples"]
    assert [(r["candidates"], r["full"], r["messages"], r["received_atoms"]) for r in rows] == [(2, 1, 2, 3), (1, 1, 1, 1), (0, 0, 0, 0)]
    assert rows[0]["raw_kv_cache_lengths"] == {"0": 1, "2": 1}
    assert rows[1]["pending"] == 1 and rows[2]["stored_kv_owners"] == 0
    assert observed["raw_kv_cache_lengths"] == {"0": 1, "1": 1, "2": 1}
    assert observed["nodes"] == {"0": dict(candidates=2, full=2, candidate_samples=2, active_samples=2),
                                 "1": dict(candidates=1, full=0, candidate_samples=1, active_samples=0)}
    continuation.states[(0, 0)].slots["times"] = torch.empty(2, dtype=torch.int64, device="meta")
    with pytest.raises(ValueError, match="decoded CPU"):
        activity(result)


def test_native_observer_requires_marker_and_resets_after_failure(monkeypatch):
    calls = []
    fake = SimpleNamespace(work_metrics=lambda: {}, reset_work=calls.append)
    monkeypatch.setitem(sys.modules, "_tide_native", fake)
    with pytest.raises(RuntimeError, match="matching rebuilt core"):
        with observe_work("native"):
            pytest.fail("old native observation must reject before execution")
    assert calls == []
    fake.work_metrics = lambda: {"packing/schema_version": 1, "packing/projection/private_bmm/rows": 4, "op/other": 7}
    with pytest.raises(RuntimeError, match="candidate"):
        with observe_work("native") as observed:
            raise RuntimeError("candidate")
    assert calls == [True, False]
    assert not observed["completed"] and not observed["throughput_sample"]
    assert observed["metrics"]["packing/projection/private_bmm/rows"] == 4
    assert "op/other" not in observed["metrics"]


def test_python_profile_publishes_buckets_outside_throughput(tmp_path, monkeypatch):
    import benchmark_weighted
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.chdir(ROOT)
    out = tmp_path/"profile"
    monkeypatch.setattr(sys, "argv", ["benchmark_weighted.py", "--device", "cpu",
        "--implementation", "python", "--mode", "inference", "--topology", "dag",
        "--dtype", "float64", "--steps", "1", "--profile", "--allow-dirty", "--output-dir", str(out)])
    benchmark_weighted.main()
    result = json.loads((out/"result.json").read_text())
    assert result["state"] == "passed" and result["kind"] == "profile"
    assert result["metrics"] == []
    assert result["packing_work"]["completed"] and not result["packing_work"]["throughput_sample"]
    assert result["packing_work"]["metrics"]["packing/projection/weighted_mm/calls"] > 0
    assert result["profile_work"]["boundary_activity"]["samples"][0]["candidates"] > 0


def test_python_work_observation_has_no_profiler_and_checks_resume(tmp_path, monkeypatch):
    import benchmark_weighted
    from core5_study_records import accept_result, timing_sample
    from tidegraph.weighted.packing_observation import enabled
    from weighted_benchmark_runners import Eager
    calls, original = [], Eager.step
    def step(self, index):
        calls.append((index, enabled()))
        return original(self, index)
    monkeypatch.setattr(Eager, "step", step)
    monkeypatch.setattr(torch.profiler, "profile", lambda **_: pytest.fail("work observation entered profiler"))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.chdir(ROOT)
    out = tmp_path/"observation"
    monkeypatch.setattr(sys, "argv", ["benchmark_weighted.py", "--device", "cpu",
        "--implementation", "python", "--mode", "inference", "--topology", "dag",
        "--dtype", "float64", "--observe-work", "--checkpoint-check", "--allow-dirty", "--output-dir", str(out)])
    benchmark_weighted.main()
    result = json.loads((out/"result.json").read_text())
    assert result["state"] == "passed" and result["post_measurement_complete_records"]
    assert result["kind"] == "work-observation" and not result["throughput_sample"]
    assert result["measured_windows"] == 1 and result["metrics"] == []
    assert result["exact_resume"]["exact"] and result["exact_resume"]["next_window"] == 2
    assert calls == [(0, False), (1, True), (2, False), (2, False), (0, False), (1, False)]
    assert result["packing_work"]["completed"] and not result["packing_work"]["throughput_sample"]
    assert result["packing_work"]["metrics"]["packing/projection/weighted_mm/calls"] > 0
    assert result["observed_work"]["boundary_activity"]["samples"][0]["candidates"] > 0
    assert result["observation_seconds"] > 0 and "timing_summary" not in result
    assert not (out/"trace.json").exists() and not timing_sample(result)
    with pytest.raises(ValueError, match="not a throughput"):
        accept_result({}, result)


@pytest.mark.parametrize("extra", [["--profile"], ["--steps", "2"], ["--implementation", "resident"]])
def test_work_observation_rejects_conflicting_scope(tmp_path, monkeypatch, extra):
    import benchmark_weighted
    out = tmp_path/"rejected"
    monkeypatch.setattr(sys, "argv", ["benchmark_weighted.py", "--device", "cpu",
        "--implementation", "python", "--observe-work", "--output-dir", str(out), *extra])
    with pytest.raises(SystemExit) as error:
        benchmark_weighted.main()
    assert error.value.code == 2 and not out.exists()
