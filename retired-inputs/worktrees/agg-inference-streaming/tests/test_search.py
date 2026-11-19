"""Capacity search bounds, real child termination and frozen-model contracts."""
from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tide import ModelConfig
from benchmarks.programme import run_child
from benchmarks.references import load_reference
from benchmarks.search import ThroughputStop, axis_values, best_observation, common_comparisons
from benchmarks.search_report import verify_metric_files, render, write_text_atomic


def test_single_axis_is_bounded_even_when_the_requested_maximum_is_huge():
    assert axis_values(2, 10**12, 8) == [2, 4, 8, 16, 32, 64, 128, 256]
    assert axis_values(128, 512, 12) == [128, 256, 512]
    with pytest.raises(ValueError):
        axis_values(1, 100, 13)


def test_noisy_samples_do_not_prove_a_throughput_plateau_and_best_requires_stability():
    stop = ThroughputStop(.03, 2)
    assert not stop.observe(100)
    assert not stop.observe(101)
    assert not stop.observe(80, stable=False)
    assert not stop.observe(102)
    assert stop.observe(99)
    assert best_observation([dict(admitted=True, stable=False, input_positions_per_second=1000),
                             dict(admitted=True, stable=True, input_positions_per_second=100)])['input_positions_per_second'] == 100
    assert best_observation([dict(admitted=False, stable=False)]) is None


def test_reference_rejects_changed_initial_parameters_and_distinguishes_shape(tmp_path):
    cfg = ModelConfig(d=64)
    arguments = dict(precision="bf16", seed=19, activity="natural", no_flush=False, batch=1, length=8)
    record = dict(state="audit_complete", config=cfg.to_dict(), arguments=arguments,
                  joint_matches_by_mode={"train": {}}, initial_audits={"train": {}})
    path = tmp_path/"reference.json"
    path.write_text(json.dumps(record))
    selected = Namespace(**dict(arguments, batch=4), modes=["train"])
    load_reference(path, cfg, selected, same_shape=False)
    with pytest.raises(ValueError, match="batch"):
        load_reference(path, cfg, selected, same_shape=True)
    selected.seed += 1
    with pytest.raises(ValueError, match="seed"):
        load_reference(path, cfg, selected, same_shape=False)


def test_rss_budget_terminates_owned_child_and_retains_failed_identity(tmp_path):
    row = dict(identity="limited", command=[sys.executable, "-c", "import time; x=bytearray(32*2**20); time.sleep(30)"],
               output=str(tmp_path/"result.json"), log=str(tmp_path/"child.log"), timeout_seconds=10,
               max_rss_bytes=8*2**20)
    run_child(row, lambda: None)
    assert row["state"] == "rss_budget_exceeded"
    assert row["descendants_gone"]
    assert row["sampled_process_tree_peak_rss_bytes"] > row["max_rss_bytes"]


def test_common_table_rejects_different_loads():
    base = dict(admitted=True, stable=True, batch=1, length=128, median_seconds=1, timed_peak_device_bytes=None)
    with pytest.raises(ValueError, match="B/L"):
        common_comparisons([dict(base, subject="tide"), dict(base, subject="transformer_A", batch=2)])


def test_plan_only_freezes_resolved_execution_overrides_without_a_model(tmp_path):
    output = tmp_path/"plan"
    command = [sys.executable, "-m", "benchmarks.search", "--config", "configs/fc-k.json", "--table", "common",
               "--max-axis-value", "2", "--rss-gib", "2", "--plan-only", "--output-dir", str(output),
               "--config-overrides", '{"checkpoint_regions":true,"local_attention_backend":"math"}']
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    resolved = json.loads((output/"config.json").read_text())
    assert resolved["checkpoint_regions"] is True
    assert resolved["local_attention_backend"] == "math"
    assert json.loads((output/"search.json").read_text())["children"] == []


@pytest.mark.parametrize("table,precision,mode,baseline", [("common", "fp32", "inference", "transformer_A"),
                                                          ("capacity", "bf16", "train", "transformer_D")])
def test_cpu_search_executes_independent_children_and_keeps_calibration_fixed(tmp_path, table, precision, mode, baseline):
    config = tmp_path/"small.json"
    config.write_text(json.dumps(dict(topology=dict(kind="fc", n=2, layers=2), d=64,
                                      memory="K", sharing="region_shared", allocation="quad")))
    output = tmp_path/"search"
    command = [sys.executable, "-m", "benchmarks.search", "--config", str(config), "--table", table,
               "--batch", "1", "--length", "2", "--max-axis-value", "2", "--max-points", "2",
               "--subjects", "tide", baseline, "--precision", precision, "--mode", mode, "--threads", "4",
               "--warmup", "1", "--steps", "2", "--maximum-noise", "100", "--rss-gib", "4",
               "--seconds", "90", "--child-timeout", "30", "--output-dir", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=100)
    assert completed.returncode == 0, completed.stderr+completed.stdout
    report = json.loads((output/"search.json").read_text())
    assert report["state"] == "completed"
    assert len(report["children"]) == 6
    assert all(row["state"] == "completed" for row in report["children"])
    assert len({row["pid"] for row in report["children"]}) == 6
    metrics = [r for line in report["lines"] for r in line["observations"]]
    assert all(r["admitted"] and len(r["seconds"]) == 2 for r in metrics)
    assert all(r["activity"]["initial_packing"] is not None for r in metrics)
    model_hashes = {r["initial_parameter_sha256"] for r in metrics if r["subject"] == "tide"}
    assert len(model_hashes) == 1
    if table == "common":
        assert len(report["common_shape_table"]) == 2
        assert all(len(r["comparisons"]) == 1 for r in report["common_shape_table"])
    else:
        baselines = [r for r in metrics if r["subject"] == baseline]
        assert baselines[0]["joint_match"] == baselines[1]["joint_match"]
        assert baselines[0]["initial_parameter_sha256"] == baselines[1]["initial_parameter_sha256"]
        assert all(r["best"] is not None for r in report["capacity_table"])
        assert all(r["activity"]["initial_minus_calibration"] is not None for r in metrics)
    assert verify_metric_files(report) == 7
    text = render([(dict(path=str(output/"search.json"), sha256="test"), report)])
    rendered = tmp_path/"report.md"
    write_text_atomic(rendered, text)
    assert rendered.read_text() == text
    assert "common-shape" in text and "capacity" in text
    first = Path(metrics[0]["result_file"]["path"])
    first.write_text(first.read_text()+" ")
    with pytest.raises(ValueError, match="identity changed"):
        verify_metric_files(report)
