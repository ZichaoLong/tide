"""Finite catalog validity, exact launch policies and conservative evidence gates."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from tidegraph.planning.study import expand, case, command
from tidegraph.planning.recipes import prepare
from tidegraph.planning import ExecutionSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from core5_study_records import accept_result, summarize, query_records


def catalog():
    return json.loads((ROOT/"benchmarks/core5-study-v1.json").read_text())


def test_catalog_has_valid_candidates_and_natural_large_scales_without_execution():
    study = catalog()
    for name, item in expand(study).items():
        plan, inv = prepare(item["recipe"], **item["window"])
        assert plan["inventory"]["totals"] == inv["totals"]
        assert len(plan["model_id"]) == len(plan["sha256"]) == 64
        assert all(c["ffn_width"] == 4*c["width"] for c in plan["memories"])
        if "target_parameters" in item["recipe"]:
            scale = plan["recipe"]["scale"]
            assert scale["actual_parameters"] == inv["totals"]["trainable"]
            assert scale["relative_difference"] == (scale["actual_parameters"]-scale["target_parameters"])/scale["target_parameters"]
    r = subprocess.run([sys.executable, "-S", str(ROOT/"scripts/run_core5_study.py"), "--list"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert set(json.loads(r.stdout)["cases"]) == set(expand(study))


def test_each_launch_carries_requested_precision_placement_workers_and_peers():
    s = catalog()
    for name in s["executions"]:
        r = case(s, "check-fc-kv", name)
        args = command(r, python="PYTHON", root=ROOT, output="OUT", workload="PLAN", native_library="CORE",
                       resident_library="RESIDENT", standalone_binary="CPP", fixture="MATERIAL")
        e = r["execution"]["requested"]
        for key in ("device", "dtype", "scoring_dtype", "workers", "intraop_threads", "interop_threads", "packing", "schedule"):
            assert args[args.index("--"+key.replace("_", "-"))+1] == str(e[key])
        assert args[args.index("--placement")+1] == ("native" if e["placement"] == "resident" else e["placement"])
        if e["peer_devices"]:
            assert all(peer in args for peer in e["peer_devices"])
        assert args[0] == ("CPP" if e["client"] == "libtorch" else "PYTHON")


def test_no_silent_inapplicable_runtime_options():
    for kw in (dict(device="npu:0", client="libtorch", placement="resident", scoring_dtype="float64"),
               dict(device="npu:0", placement="mixed-b", scoring_dtype="float64"),
               dict(client="pytorch", workers=2)):
        with pytest.raises(ValueError):ExecutionSpec(**kw)
    assert ExecutionSpec(device="npu:0", placement="mixed-a", scoring_dtype="float64")
    bad = catalog();bad["measurement"]["warmup"] = 2
    with pytest.raises(ValueError):expand(bad)
    r = case(catalog(), "check-fc-kv", "python-cpu")
    with pytest.raises(ValueError, match="resident limits"):
        command(r, python="python", root=ROOT, output="out", workload="p", resident_limits="limits")


def test_native_checkpoint_budget_is_explicit_and_cannot_be_silently_ignored():
    common = dict(python="PYTHON", root=ROOT, output="OUT", workload="PLAN", standalone_binary="CPP", fixture="MATERIAL")
    r = case(catalog(), "check-fc-kv", "cpp-cpu")
    argv = command(r, **common, checkpoint_max_bytes=2**34)
    assert argv[argv.index("--checkpoint-max-bytes")+1] == str(2**34)
    for invalid in (0, -1, True, 4.0, 2**63):
        with pytest.raises((ValueError, TypeError)):
            command(r, **common, checkpoint_max_bytes=invalid)
    r["measurement"]["checkpoint_check"] = False
    with pytest.raises(ValueError, match="checkpoint check"):
        command(r, **common, checkpoint_max_bytes=2**34)
    r = case(catalog(), "check-fc-kv", "python-cpu")
    with pytest.raises(ValueError, match="LibTorch"):
        command(r, **common, checkpoint_max_bytes=2**34)


def test_record_gate_requires_gradients_optimizer_identity_and_actual_resume(tmp_path):
    r = case(catalog(), "check-fc-kv", "python-cpu")
    good = dict(state="passed", post_measurement_complete_records=True, requested_dtype="float32",
                workload_id=r["plan"]["sha256"], parameter_values_checked=True,
                final_parameter_gradients_checked=True, optimizer_values_checked=True,
                exact_resume=dict(state="passed", exact=True))
    accept_result(r, good)
    for key in good:
        bad = deepcopy(good);bad.pop(key)
        with pytest.raises(ValueError):accept_result(r, bad)
    parent = tmp_path/"run.json"
    record = {k:v for k,v in r.items() if k not in {"plan", "inventory"}}
    record.update(schema="tide-core5-run-v1", source="commit", state="failed", model_id=r["plan"]["model_id"],
                  workload_id=r["plan"]["sha256"], parameters=r["inventory"]["totals"],
                  recipe=r["plan"]["recipe"], graph_counts=r["plan"]["graph"]["counts"])
    parent.write_text(json.dumps(record))
    (tmp_path/"measurement").mkdir()
    (tmp_path/"measurement/result.json").write_text(json.dumps(dict(good, metrics=[dict(seconds=3., full_calls=2)])))
    result = summarize(parent)
    assert result["measured_seconds"] == 3 and result["work"]["full_calls"] == 2
    assert result["state"] == "failed" and not result["throughput_qualified"]
    assert query_records(parent) == [result]
    # A reviewed record is queryable after its raw run directory is gone.
    result["run"] = "/retired/not-a-live-dependency/run.json"
    result["checks"] = good
    result["throughput_qualified"] = True  # A metric cannot override failure.
    evidence = tmp_path/"reviewed.json"
    evidence.write_text(json.dumps(dict(schema="tide-core5-finite-performance-evidence-v1",
        models={r["case_id"]:dict(parameters=record["parameters"], counts=record["graph_counts"], recipe=record["recipe"])},
        runs=[result])))
    assert query_records(evidence)[0]["throughput_qualified"] is False
    child = subprocess.run([sys.executable, "-S", str(ROOT/"scripts/index_core5_study.py"), str(evidence),
                            "--state", "failed", "--format", "json"], capture_output=True, text=True, check=True)
    assert json.loads(child.stdout)[0]["parameters"] == record["parameters"]


@pytest.mark.parametrize("kind", ["profile", "work-observation"])
def test_observation_costs_never_gain_throughput_qualification(tmp_path, kind):
    r = case(catalog(), "check-fc-kv", "python-cpu")
    r["measurement"].update(mode="inference", checkpoint_check=False)
    good = dict(state="passed", kind=kind, post_measurement_complete_records=True,
                requested_dtype="float32", workload_id=r["plan"]["sha256"], metrics=[dict(seconds=1.)])
    with pytest.raises(ValueError, match="observation"):
        accept_result(r, good)
    record = {k: v for k, v in r.items() if k not in {"plan", "inventory"}}
    record.update(schema="tide-core5-run-v1", source="commit", state="passed", model_id=r["plan"]["model_id"],
                  workload_id=r["plan"]["sha256"], parameters=r["inventory"]["totals"],
                  recipe=r["plan"]["recipe"], graph_counts=r["plan"]["graph"]["counts"])
    path = tmp_path/"run.json"
    path.write_text(json.dumps(record))
    (tmp_path/"measurement").mkdir()
    (tmp_path/"measurement/result.json").write_text(json.dumps(good))
    summary = summarize(path)
    assert summary["correctness_qualified"] and not summary["throughput_qualified"]
    assert summary["measured_seconds"] == 1. and summary["kind"] == "observation"
    reviewed = tmp_path/"reviewed.json"
    reviewed.write_text(json.dumps(dict(schema="tide-core5-finite-performance-evidence-v1",
        models={r["case_id"]: dict(parameters=record["parameters"], counts=record["graph_counts"], recipe=record["recipe"])},
        runs=[summary])))
    assert not query_records(reviewed)[0]["throughput_qualified"]


def test_inference_review_honors_declared_resume_without_rewriting_old_scope(tmp_path):
    from core5_study_records import complete_checks
    checks = {"post_measurement_complete_records": True}
    assert complete_checks("passed", "inference", checks, None)
    assert not complete_checks("passed", "inference", checks, None, checkpoint_required=True)
    assert complete_checks("passed", "inference", checks, {"state": "passed", "exact": True}, checkpoint_required=True)
    row = dict(case="case", state="passed", mode="inference", checks=checks, checkpoint_check=True)
    evidence = dict(schema="tide-core5-finite-performance-evidence-v1",
                    models={"case": dict(parameters={}, counts={}, recipe={})}, runs=[row])
    path = tmp_path/"reviewed.json"
    path.write_text(json.dumps(evidence))
    assert not query_records(path)[0]["correctness_qualified"]
    row.pop("checkpoint_check")
    path.write_text(json.dumps(evidence))
    old = query_records(path)[0]
    assert old["correctness_qualified"] and "exact_resume" not in old
