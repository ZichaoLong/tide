from copy import deepcopy
import csv
import io
import json
from tidegraph.experiment.config import resolve
from tidegraph.experiment.report import load, render, select, summarize


def result():
    config = resolve(dict(schema="tide-experiment-v1", model=dict(recipe=dict(topology="hb", depth=1,
        region_width=1, width=8, offsets=[0])), execution=dict(device="cpu")))
    return dict(state="passed", execution_status="completed", config_id=config["config_id"], resolved_config=config,
        validation=dict(mode="strict", numeric="passed", discrete="passed", complete="checked", strict_passed=True),
        checkpoint=dict(state="passed", exact_resume=True),
        metrics=[dict(phase="warmup", seconds=9), dict(seconds=.2), dict(seconds=.4)], timing_contract="full-return")


def test_reports_separate_execution_exploration_and_strict_qualification(tmp_path):
    success = result()
    row = summarize(success, tmp_path/"result.json")
    assert row["qualification"] == "strict-checks-passed"
    assert abs(row["median_seconds"]-.3) < 1e-9
    failed = deepcopy(success)
    failed["validation"].update(mode="explore", numeric="failed", strict_passed=False)
    row = summarize(failed, tmp_path/"result.json")
    assert row["execution_status"] == "completed"
    assert row["qualification"] == "strict-failure"
    assert row["exploratory"] is True
    failed["validation"] = dict(mode="none")
    row = summarize(failed, tmp_path/"result.json")
    assert row["qualification"] == "not-established"
    assert row["numerical"] == "unchecked"


def test_checkpoint_failure_and_unchecked_do_not_become_qualifications(tmp_path):
    data = result()
    for state in ({}, dict(state="failed", exact_resume=False)):
        data["checkpoint"] = state
        assert summarize(data, tmp_path/"result.json")["qualification"] != "strict-checks-passed"


def test_load_filter_csv_and_html_escape_without_importing_runtime(tmp_path):
    data = result()
    data["timing_contract"] = '<script>alert("unsafe")</script>'
    directory = tmp_path/"run"
    directory.mkdir()
    (directory/"result.json").write_text(json.dumps(data))
    rows = load([tmp_path, directory])
    assert len(rows) == 1
    assert len(select(rows, platform="cpu", topology="hb", params_min=1, status="completed")) == 1
    assert not select(rows, platform="cuda")
    html = render(rows, format="html")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert list(csv.DictReader(io.StringIO(render(rows, format="csv"))))[0]["qualification"] == "strict-checks-passed"
    assert "unchecked" not in render(rows, format="table")
