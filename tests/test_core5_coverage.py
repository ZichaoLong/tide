"""Actual configuration summaries and conservative, source-scoped evidence views."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import pytest
from tidegraph.planning.recipes import prepare
from tidegraph.planning.facts import describe, from_reviewed_model
from tidegraph.planning.execution import ExecutionSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from core5_coverage import table, models, view
from core5_study_records import query_records, complete_checks
from record_core5_checks import extract


def test_all_reviewed_models_reproduce_their_actual_windows_and_classification():
    evidence = json.loads((ROOT/"docs/evidence/core5-natural-study-20261029.json").read_text())
    expected = {"fc-medium-few-kv": [2048], "fc-medium-many-kv": [2048],
                "hb-medium-kv": [0, 2560], "fc-1b-kv": [0, 5120],
                "hb-1b-kv": [0, 4608], "lh-1b-kv": [0, 4608]}
    for name, model in evidence["models"].items():
        facts = from_reviewed_model(model)
        assert facts["kv_windows"] == expected[name]
        assert facts["source_classification"] == "source-node; external=0"
        assert facts["source_binding"]["mode"] == "region"
        assert sum(len(g["owners"]) for g in facts["memories"]) == facts["nodes"]
        assert sum(len(g["owners"]) for g in facts["regional"]) == facts["regions"]
    bad = deepcopy(evidence["models"]["fc-1b-kv"])
    bad["recipe"]["requested"]["history"] = 511
    with pytest.raises(ValueError, match="identity differs"):
        from_reviewed_model(bad)


def test_configuration_table_in_report_is_generated_from_full_model_facts():
    records = query_records(ROOT/"docs/evidence/core5-natural-study-20261029.json")
    headers, rows = table(models(records), "models")
    generated = "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join("---" for _ in headers)+" |",
                            *["| "+" | ".join(row)+" |" for row in rows]])
    report = (ROOT/"docs/evidence/core5-natural-study-20261029.md").read_text()
    block = report.split("<!-- core5-models:start -->\n")[1].split("\n<!-- core5-models:end -->")[0]
    assert block == generated


def test_runtime_and_reviewed_flags_do_not_promote_partial_or_failed_training():
    checks = dict(post_measurement_complete_records=True, parameter_values_checked=True,
                  final_parameter_gradients_checked=True, optimizer_values_checked=True)
    resume = dict(state="passed", exact=True)
    assert complete_checks("passed", "training", checks, resume)
    for key in checks:
        bad = dict(checks);bad.pop(key)
        assert not complete_checks("passed", "training", bad, resume)
    for state in ("failed", "running", "skipped"):
        assert not complete_checks(state, "training", checks, resume)
    assert not complete_checks("passed", "training", checks, None)


def test_embedded_reviewed_configuration_facts_keep_workload_identity():
    # This published report carries complete static facts inside each model,
    # without the legacy samples/positions reconstruction fields at its top.
    report = ROOT/"docs/evidence/core5-cuda-finite-20261031.json"
    records = query_records(report)
    row = next(r for r in records if r["case"] == "fc-small-kv")
    assert "samples" not in row["reviewed_model"]
    result = view(row)
    assert (result["topology"], result["samples"], result["positions"], result["widths"]) == ("fc", 1, 2, [8])
    assert len(models(records)) == 6
    for key, wrong in (("model_id", "0"*64), ("workload_id", "0"*64),
                       ("parameters", dict(row["parameters"], trainable=0))):
        bad = deepcopy(row)
        bad["reviewed_model"]["configuration_facts"][key] = wrong
        with pytest.raises(ValueError, match="identity mismatch"):
            view(bad)


def junit_case(parent, name, payload, checks=None, state="passed"):
    case = ET.SubElement(parent, "testcase", name=name, classname="test_core5_recipe_training")
    properties = ET.SubElement(case, "properties")
    ET.SubElement(properties, "property", name="core5_case", value=json.dumps(payload))
    if checks is not None:
        ET.SubElement(properties, "property", name="core5_checks", value=json.dumps(checks))
    if state != "passed": ET.SubElement(case, "failure" if state == "failed" else "skipped")


def test_junit_and_index_keep_failure_missing_checks_and_no_performance_distinct(tmp_path):
    plan, _ = prepare(dict(topology="fc", family="mixed", width=8, region_width=2, depth=8), samples=2, positions=2)
    facts = describe(plan)
    assert set(facts["families"]) == {"kv", "gated_delta", "ssd"} and facts["edges"] == 28
    payload = dict(case="fc-mixed", facts=facts, execution=ExecutionSpec(dtype="float64", scoring_dtype="float64").record(), mode="training", updates=2)
    checks = dict(post_measurement_complete_records=True, parameter_values_checked=True,
                  final_parameter_gradients_checked=True, optimizer_values_checked=True, exact_resume=dict(state="passed", exact=True))
    suite = ET.Element("testsuite")
    junit_case(suite, "complete", payload, checks)
    junit_case(suite, "failed", payload, checks, "failed")
    junit_case(suite, "missing-checks", payload)
    junit_case(suite, "skipped", payload, state="skipped")
    xml = tmp_path/"tests.xml";ET.ElementTree(suite).write(xml)
    report = extract(xml, "a"*40)
    assert len(report["models"]) == 1
    evidence = tmp_path/"checks.json";evidence.write_text(json.dumps(report))
    index = tmp_path/"index.json"
    index.write_text(json.dumps(dict(schema="tide-core5-evidence-index-v1", reports=[evidence.name])))
    rows = query_records(index)
    assert [r["correctness_qualified"] for r in rows] == [True, False, False, False]
    assert not any(r["throughput_qualified"] for r in rows)
    assert all(r["measured_seconds"] is None for r in rows)
    assert view(rows[0])["families"] == ["gated_delta", "kv", "ssd"]
    command = [sys.executable, "-S", str(ROOT/"scripts/index_core5_study.py"), str(index),
               "--view", "coverage", "--family", "gated_delta", "--client", "pytorch", "--format", "json"]
    child = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=True)
    assert len(json.loads(child.stdout)) == 4  # No historical run directory or Torch needed.
    index.write_text(json.dumps(dict(schema="tide-core5-evidence-index-v1", reports=[index.name])))
    with pytest.raises(ValueError, match="cyclic"):
        query_records(index)
