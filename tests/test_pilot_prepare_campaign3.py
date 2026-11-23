"""Preparation-only campaign handoff checks with real frozen temporary files."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from pilotlab import prepare3
from pilotlab import prepare_campaign3 as p
from pilotlab.scheduler import Quota


@pytest.mark.parametrize("hard,free,available", [(700, 1000, 300), (700, 250, 250), (0, 1000, 1000)])
def test_storage_observation_above_soft_quota(tmp_path, monkeypatch, hard, free, available):
    monkeypatch.setattr(p, "current_quota", lambda: Quota("fs", 400, 300, hard, 0))
    monkeypatch.setattr(p.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    record = p.storage_observation(tmp_path / "out", reserve=available-2, snapshot_bytes=1)
    assert record["available_storage_bytes"] == available
    with pytest.raises(ValueError, match="insufficient quota/filesystem"):
        p.storage_observation(tmp_path / "out", reserve=available-1, snapshot_bytes=1)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    repo, foundation = tmp_path / "repo", tmp_path / "foundation"
    files = {"__init__.py", "prepare3.py", "preflight3.py", "scheduler3.py", "worker3.py", "data3.py",
             "models3.py", "catalog3.py", "models.py", "worker.py", "io.py", "scheduler.py", "selection.py",
             "round3.py", "report3.py", "prepare_campaign3.py"}
    (repo / "pilotlab").mkdir(parents=True)
    for name in files:
        (repo / "pilotlab" / name).write_text("# qualified fixture " + name + "\n")
    (foundation / "src/tide").mkdir(parents=True)
    (foundation / "src/tide/__init__.py").write_text("# exact foundation\n")
    (foundation / "src/tide/allocation.py").write_text("# exact cap correction\n")
    data = tmp_path / "data"
    write_json(data / "manifest.json", {"fixture_data": "unchanged"})
    monkeypatch.setattr(prepare3, "verify_data", lambda path: {
        "manifest_path": str(path / "manifest.json"), "manifest_sha256": p.sha256(path / "manifest.json"), "assets": {}})
    monkeypatch.setattr(prepare3.importlib.metadata, "version", lambda name: "qualified-version")
    budget = tmp_path / "budget.json"
    prepare3.ensure_budget(budget, now=time.time() - 100)
    old_root = tmp_path / "preflight"
    prepare3.freeze(repo, foundation, old_root, python=sys.executable, data=data, budget=budget)
    old = p.load(old_root / "prepared.json")
    subjects = [{"id": f"subject-{i}", "model_key": f"effective-{i}", "group": "fc",
                 "model": {"family": "tide", "config": {"d": 64, "fixture": i}}} for i in range(216)]
    request = {"id": "pf-one", "model_id": subjects[0]["id"], "model": subjects[0]["model"],
               "source_identity": old["source_identity"], "precision": "fp32", "deterministic": False}
    output = Path(old["output"])
    task_path = write_json(output / "tasks/pf-one/task.json", request)
    memory = {"npu_total_memory_bytes": 32 * p.GIB, "npu_peak_allocated_bytes": p.GIB}
    result = {"status": "completed", "source_identity": old["source_identity"],
              "task_sha256": p.sha256(task_path), "health": {"eligible": True}, "device_memory": memory}
    result_path = write_json(output / "tasks/pf-one/attempt-0/result.json", result)
    preflight_plan = write_json(output / "preflight-plan.json", {"core_tasks": [request]})
    quota = Quota("fixture-fs", 10 * p.GIB, 1000 * p.GIB, 1200 * p.GIB, 0)
    resources = {"quota": asdict(quota), "rss_bytes": 12345, "available_storage_bytes": 990 * p.GIB}
    write_json(output / "scheduler.json", {"status": "completed", "peak_actual_npu_concurrency": 8,
                                           "resources": resources})
    report = {"status": "passed", "source_identity": old["source_identity"], "remaining_descendants": [],
              "actual_peak_npu_concurrency": 8, "plan_path": str(preflight_plan), "plan_sha256": p.sha256(preflight_plan),
              "result_files": {"pf-one": {"path": str(result_path), "sha256": p.sha256(result_path)}},
              "measurements": [{"id": "pf-one", "model_id": subjects[0]["id"], "device_memory": memory,
                                "result_path": str(result_path), "result_sha256": p.sha256(result_path),
                                "cpu_threads": 4, "sequence_length": 128}]}
    report_path = write_json(output / "preflight-report.json", report)
    monkeypatch.setattr(p, "current_quota", lambda: quota)
    monkeypatch.setattr(p.shutil, "disk_usage", lambda path: SimpleNamespace(free=2000 * p.GIB))
    monkeypatch.setattr(p, "frozen_subjects", lambda prepared: deepcopy(subjects))

    def no_processes(*args, **kwargs):
        pytest.fail("preparation must not launch a service or probe a device")

    monkeypatch.setattr(p.subprocess, "run", no_processes)
    args = dict(repo=repo, foundation=foundation, output=tmp_path / "main-new", python=sys.executable,
                data=data, budget=budget, preflight_prepared=old_root / "prepared.json", card_count=16, max_workers=16)
    return SimpleNamespace(args=args, old=old, report_path=report_path, subjects=subjects, quota=quota)


def test_campaign_draft_keeps_existing_artifacts_budget_and_explicit_launch_contract(inputs, monkeypatch):
    budget_before = inputs.args["budget"].read_bytes()
    report_before = inputs.report_path.read_bytes()
    original_freeze = prepare3.freeze
    generated_preflight_artifacts = {}

    def observe_freeze(*args, **kwargs):
        answer = original_freeze(*args, **kwargs)
        root = Path(args[2])
        for name in ("prepared.json", "launch-draft.json", "launch.sh"):
            generated_preflight_artifacts[name] = (root / name).read_bytes()
        return answer

    monkeypatch.setattr(prepare3, "freeze", observe_freeze)
    (inputs.args["repo"] / "pilotlab/scheduler3.py").write_text("# changed lifecycle, same qualified math\n")
    draft = p.prepare(**inputs.args)
    root = inputs.args["output"]
    assert draft["status"] == "not_submitted"
    assert inputs.args["budget"].read_bytes() == budget_before
    assert inputs.report_path.read_bytes() == report_before
    for name, before in generated_preflight_artifacts.items():
        assert (root / name).read_bytes() == before
    plan = p.load(draft["plan_path"])
    assert len(plan["subjects"]) == 216 and plan["cohort_models"] == 4
    assert plan["config"]["max_workers"] == 16
    assert plan["protocol"] == {"precision": "fp32", "deterministic": False, "sequence_length": 128,
                                "cpu_threads": 4, "npu_total_bytes": 32 * p.GIB, "admitted_parallelism": 8}
    assert {row["file"] for row in plan["qualification_transfer"]["lifecycle_source_changes"]} == {"scheduler3.py"}
    assert plan["config"]["deadline_unix"] == p.load(inputs.args["budget"])["deadline_unix"]
    assert "--slice=background.slice" in draft["launch_argv"]
    assert "--property=TimeoutStopSec=120" in draft["launch_argv"]
    assert f'--property=RuntimeMaxSec={draft["runtime_max_seconds"]}' in draft["launch_argv"]
    assert draft["command_argv"][2:4] == ["-m", "pilotlab.round3"]
    assert draft["plan_sha256"] == p.sha256(draft["plan_path"])
    assert draft["launcher_sha256"] == p.sha256(draft["launcher"])
    script = Path(draft["launcher"]).read_text()
    assert "module load ascend/dev-workspace-9.2.0" in script
    assert "export TORCH_DEVICE_BACKEND_AUTOLOAD=0" in script
    assert "pilotlab.preflight3" not in script


@pytest.mark.parametrize("damage", ["core", "foundation", "foundation_added", "failed_report", "result_changed", "expired_budget"])
def test_invalid_qualification_or_budget_stops_before_new_snapshot(inputs, damage):
    if damage == "core":
        (inputs.args["repo"] / "pilotlab/models3.py").write_text("# different training behavior\n")
    elif damage == "foundation":
        (inputs.args["foundation"] / "src/tide/allocation.py").write_text("# different VJP\n")
    elif damage == "foundation_added":
        (inputs.args["foundation"] / "src/tide/new_math.py").write_text("# unqualified new source\n")
    elif damage == "failed_report":
        report = p.load(inputs.report_path)
        report["status"] = "blocked"
        write_json(inputs.report_path, report)
    elif damage == "result_changed":
        result_path = p.load(inputs.report_path)["measurements"][0]["result_path"]
        write_json(result_path, {"status": "completed", "changed": True})
    else:
        ledger = p.load(inputs.args["budget"])
        ledger["started_unix"] = time.time() - 31 * 86400
        ledger["deadline_unix"] = ledger["started_unix"] + 30 * 86400
        write_json(inputs.args["budget"], ledger)
    with pytest.raises(ValueError):
        p.prepare(**inputs.args)
    assert not inputs.args["output"].exists()


@pytest.mark.parametrize("catalog_changed", [False, True])
def test_matcher_change_requires_identical_full_catalog(inputs, monkeypatch, catalog_changed):
    (inputs.args["repo"] / "pilotlab/catalog3.py").write_text("# new matcher; generator must remain equal\n")
    old_source = inputs.old["source"]

    def catalogs(prepared):
        subjects = deepcopy(inputs.subjects)
        if catalog_changed and prepared["source"] != old_source:
            subjects[-1]["model"]["config"]["d"] *= 2
        return subjects

    monkeypatch.setattr(p, "frozen_subjects", catalogs)
    if catalog_changed:
        with pytest.raises(ValueError, match="216-subject catalog changed"):
            p.prepare(**inputs.args)
        assert not (inputs.args["output"] / "campaign-plan.json").exists()
    else:
        inputs.args["card_count"] = 4
        draft = p.prepare(**inputs.args)
        plan = p.load(draft["plan_path"])
        assert plan["config"]["max_workers"] == 4
        assert [row["file"] for row in plan["qualification_transfer"]["catalog_source_changes"]] == ["catalog3.py"]


def test_actual_catalog_subprocess_uses_only_frozen_cpu_sources(inputs, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=json.dumps(inputs.subjects))

    monkeypatch.setattr(p.subprocess, "run", run)
    # Recover the real helper replaced by the fixture without executing any
    # external command; its structured argv/environment are the contract.
    import importlib.util
    spec = importlib.util.spec_from_file_location("pilotlab._prepare_campaign_probe", p.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.frozen_subjects(inputs.old)) == 216
    argv, kwargs = calls[0]
    assert argv[0] == inputs.old["python"] and argv[1] == "-c"
    assert kwargs["cwd"] == inputs.old["source"]
    assert kwargs["env"]["PYTHONPATH"] == inputs.old["environment"]["PYTHONPATH"]
    assert kwargs["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert kwargs["timeout"] == 120 and kwargs["check"] is True
