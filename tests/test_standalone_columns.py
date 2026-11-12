"""Actual standalone default, independent record loss and process continuation.

The canonical graph-checkpoint byte comparison covers parameters/aliases, raw
states/history/ledger and SGD slots, independently of JSON cost observations.
Run with a matched TIDE_WEIGHTED_BENCHMARK and TIDE_BUILD_DIR from frozen source.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from tidegraph.planning.topology import topology
from tidegraph.planning.workload import make_plan

ROOT = Path(__file__).resolve().parents[1]


def execute(*args):
    result = subprocess.run(list(map(str,args)), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    return result


def checkpoint(run):
    root = run/"checkpoint"
    generation = json.loads((root/"latest.json").read_text())["generation"]
    return (root/generation/"state.pt").read_bytes()


@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd"])
def test_actual_default_columns_three_sgd_and_new_process_resume(tmp_path, family):
    executable = os.environ.get("TIDE_WEIGHTED_BENCHMARK")
    if not executable:
        pytest.skip("standalone qualification supplies TIDE_WEIGHTED_BENCHMARK")
    from tidegraph.experiment.records import write_json
    plan, _ = make_plan(topology("fc",depth=2,region_width=2),family=family,
                        width=4,samples=2,positions=3)
    write_json(tmp_path/"plan.json",plan)
    execute(sys.executable,ROOT/"scripts/export_weighted_benchmark.py","--native-library",os.environ["TIDE_BUILD_DIR"],
            "--workload",tmp_path/"plan.json","--binary","--output",tmp_path/"fixture.json")
    common=[executable,"--fixture",tmp_path/"fixture.json","--device","cpu","--dtype","float64",
            "--mode","training","--warmup","0","--threads","1","--return-mode","outputs","--validation","none"]
    def run(name,*args):
        execute(*common,"--output-dir",tmp_path/name,*args)
        return json.loads((tmp_path/name/"result.json").read_text())
    whole=run("whole","--steps","3","--checkpoint-check","true")
    assert whole["effective_execution"]["interface"] == "tensor-window-v1"
    assert whole["effective_execution"]["schedule"] == "settle-columns-v1"
    assert whole["effective_execution"]["fallback_reason"] is None
    assert whole["exact_resume"]["state"] == "passed"
    prefix=run("prefix","--steps","1")
    resumed=run("resumed","--steps","2","--resume",tmp_path/"prefix/checkpoint","--checkpoint-check","true")
    assert prefix["checkpoint"]["next_window"] == 1 and resumed["start_window"] == 1
    assert resumed["exact_resume"]["state"] == "passed"
    assert checkpoint(tmp_path/"whole") == checkpoint(tmp_path/"resumed")
    # Scalar native Streaming is an independent schedule. This is a task loss
    # projection comparison; all-field/None/VJP is checked by the C++ owner test.
    reference=run("reference","--steps","3","--input-interface","records","--schedule","streaming","--packing","scalar")
    assert reference["effective_execution"]["interface"] == "records"
    for actual,expected in zip(whole["metrics"],reference["metrics"],strict=True):
        assert actual["loss"] == pytest.approx(expected["loss"],rel=1e-8,abs=1e-10)
        for key in ("candidate_updates","full_calls","messages","outputs","pending"):
            assert actual[key] == expected[key]
    # Full record/root diagnostics retain their own interface despite auto.
    if family != "kv":
        return
    complete=run("complete","--steps","1","--return-mode","complete","--validation","strict")
    assert complete["effective_execution"]["interface"] == "records"
    assert complete["post_measurement_complete_records"]
    mismatch=subprocess.run(list(map(str,[*common,"--output-dir",tmp_path/"mismatch","--steps","1",
        "--input-interface","records","--resume",tmp_path/"prefix/checkpoint"])),capture_output=True,text=True,timeout=120)
    assert mismatch.returncode and "identity" in mismatch.stderr
    # A handcrafted unsupported policy must fail, never become strict native
    # math by dropping an unrecognised JSON field.
    fixture=json.loads((tmp_path/"fixture.json").read_text())
    fixture["regions"][0]["support_policy"]="positive-floor-v1"
    write_json(tmp_path/"floor.json",fixture)
    refused=subprocess.run(list(map(str,[executable,"--fixture",tmp_path/"floor.json","--device","cpu",
        "--output-dir",tmp_path/"floor"])),capture_output=True,text=True,timeout=120)
    assert refused.returncode and "strict support_policy" in refused.stderr
