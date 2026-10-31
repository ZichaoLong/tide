from copy import deepcopy
import csv
import io
import json
import pytest
from tidegraph.experiment.config import resolve
from tidegraph.experiment.report import load, render, select, summarize, save
from tidegraph.experiment.records import fingerprint


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


def test_archived_results_remain_queryable_without_original_run_directory(tmp_path):
    source = tmp_path/"result.json"
    source.write_text(json.dumps(result()))
    rows = load([source])
    assert len(rows[0]["result_sha256"]) == 64
    archived = tmp_path/"reviewed.json"
    archived.write_text(render(rows, format="archive"))
    source.unlink()
    assert load([archived]) == rows
    assert len(select(load([archived]), topology="hb", platform="cpu")) == 1
    record = json.loads(archived.read_text())
    record["results"][0]["qualification"] = "changed"
    archived.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="content digest"):
        load([archived])


def test_archive_preserves_provenance_window_costs_and_strict_diagnostics(tmp_path):
    data = result()
    data.update(construction_seconds=1.25, warmup=[dict(seconds=9)],
        identity=dict(package=dict(installed_files_sha256="package-hash", source_provenance="fixed-source",
                                   files_sha256={"a.py": "hash-a"})))
    data["validation"].update(reports=[dict(differences=[dict(path="state.keys", max_abs=2e-6)])])
    row = summarize(data, tmp_path/"result.json")
    archive = tmp_path/"archive.json"
    archive.write_text(render([row], format="archive"))
    detail = load([archive])[0]["details"]
    assert detail["resolved_config"] == json.loads(json.dumps(data["resolved_config"]))
    assert detail["metrics"] == data["metrics"]
    assert detail["construction_seconds"] == 1.25 and detail["warmup"] == [{"seconds": 9}]
    assert detail["validation"]["reports"] == data["validation"]["reports"]
    package = detail["identity"]["package"]
    assert package["installed_files_sha256"] == "package-hash"
    assert package["source_provenance"] == "fixed-source"
    assert package["files_sha256"]["sha256"] == fingerprint({"a.py": "hash-a"})
    assert data["identity"]["package"]["files_sha256"] == {"a.py": "hash-a"}


def test_old_archive_remains_queryable_without_inventing_missing_details(tmp_path):
    row = summarize(result(), tmp_path/"result.json")
    del row["details"]
    path = tmp_path/"v1.json"
    path.write_text(json.dumps(dict(schema="tide-experiment-evidence-v1", results=[row],
                                   results_sha256=fingerprint([row]))))
    loaded = load([path])[0]
    assert loaded["details"] == {"scope": "unrecorded in v1 archive"}
    assert loaded["median_seconds"] == row["median_seconds"]


def test_compressed_archive_remains_queryable_and_html_folds_large_diagnostics(tmp_path):
    rows = [summarize(result(), tmp_path/"result.json")]
    path = tmp_path/"reviewed.json.gz"
    save(path, render(rows, format="archive"))
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    assert load([path]) == json.loads(json.dumps(rows))
    report = render(load([path]), format="html")
    assert '<details><summary>Details</summary>' in report


def test_abrupt_exit_keeps_completed_windows_but_never_infers_remaining_checks(tmp_path):
    data = result()
    data.update(state="running", phase="exact-resume")
    data["validation"] = dict(numeric="not_checked", discrete="not_checked", complete="not_checked")
    data["checkpoint"] = dict(state="saved", exact_resume="not_checked")
    directory = tmp_path/"case"
    (directory/"run").mkdir(parents=True)
    path = directory/"run/result.json"
    path.write_text(json.dumps(data))
    status = dict(schema="tide-durable-job-v1", state="failed", exit_code=-11, workload_exit_code=-11,
                  termination_reason="exit", remaining_live_pids=[], metadata=dict(config_id=data["config_id"]))
    (directory/"status.json").write_text(json.dumps(status))
    original = path.read_bytes()
    row = load([path])[0]
    assert row["execution_status"] == "completed" and row["median_seconds"] == pytest.approx(.3)
    assert row["qualification"] == "process-failure" and row["throughput_qualified"] is False
    assert row["numerical"] == "not_checked" and row["exact_resume"] == "not_checked"
    assert row["error"]["exit_code"] == -11 and row["error"]["last_recorded_phase"] == "exact-resume"
    assert row["details"]["state"] == "running" and row["details"]["process_exit"]["state"] == "failed"
    assert path.read_bytes() == original
    archive = tmp_path/"reviewed.json.gz"
    save(archive, render([row], format="archive"))
    path.unlink()
    (directory/"status.json").unlink()
    assert load([archive]) == [row]


def test_supervisor_never_promotes_checks_or_conflates_another_case(tmp_path):
    data = result()
    data.update(state="failed", error=dict(type="NumericMismatch"))
    data["validation"].update(numeric="failed", strict_passed=False)
    (tmp_path/"run").mkdir()
    path = tmp_path/"run/result.json"
    status = dict(schema="tide-durable-job-v1", state="failed", exit_code=1,
                  metadata=dict(config_id=data["config_id"]))
    (tmp_path/"status.json").write_text(json.dumps(status))
    row = summarize(data, path)
    assert row["qualification"] == "strict-failure" and row["error"] == data["error"]
    status.update(state="passed", exit_code=0)
    (tmp_path/"status.json").write_text(json.dumps(status))
    assert summarize(data, path)["qualification"] == "strict-failure"
    status.update(state="failed", exit_code=-11, metadata=dict(config_id="another-case"))
    (tmp_path/"status.json").write_text(json.dumps(status))
    assert "process_exit" not in summarize(result(), path)["details"]
