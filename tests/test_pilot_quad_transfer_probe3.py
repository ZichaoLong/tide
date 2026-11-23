"""Real CPU witness, source isolation, and finite quad-proof admission."""
from __future__ import annotations

import copy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pilotlab import quad_transfer_probe3 as probe
from pilotlab.data3 import prepare
from pilotlab.io import atomic_json, sha256
from pilotlab.prepare3 import ensure_budget
from pilotlab.transfer3 import REVIEW_SCHEMA, _validate_review
from test_pilot_data3 import write_source


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def snapshots(tmp_path):
    data = tmp_path / "data"
    prepare(write_source(tmp_path / "corpus"), data)
    budget = tmp_path / "budget.json"
    ensure_budget(budget)
    foundation = importlib.util.find_spec("tide")
    assert foundation and foundation.submodule_search_locations, "the current test foundation must be explicit on PYTHONPATH"
    base = Path(next(iter(foundation.submodule_search_locations)))
    prepared = {}
    for side in ("old", "new"):
        root = tmp_path / side
        source = root / "source"
        target = root / "foundation/src/tide"
        (source / "pilotlab").mkdir(parents=True)
        for path in (ROOT / "pilotlab").glob("*.py"):
            shutil.copyfile(path, source / "pilotlab" / path.name)
        for path in base.rglob("*.py"):
            destination = target / path.relative_to(base)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        if side == "new":
            with (target / "allocation.py").open("a") as stream:
                stream.write("\n# CPU harness fixture: mathematical code is identical.\n")
        files = [*(source / "pilotlab").glob("*.py"), *target.rglob("*.py")]
        identity = {"schema": "pilot-r3-source-v1", "source_files": {str(path): sha256(path) for path in files},
            "data_manifest_sha256": sha256(data / "manifest.json"), "python_version": sys.version.split()[0],
            "runtime_packages": {name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "numpy")}}
        value = {"schema": "pilot-r3-prepared-inputs-v1", "python": sys.executable, "source": str(source),
            "foundation": str(root / "foundation"), "data": str(data), "budget_ledger": str(budget), "source_identity": identity}
        path = root / "prepared.json"
        atomic_json(path, value)
        prepared[side] = path
    old = probe.load(prepared["old"])
    model = {"family": "tide", "config": {"topology": {"kind": "fc", "n": 2, "layers": 1}, "d": 64,
        "memory": "K", "sharing": "independent", "budget": 1, "allocation": "quad", "attention_backend": "math", "ffn_width": 64}}
    choices = {"quad-fixture": {"recipe_id": "r0", "init_recipe": "default", "confirmation": []}}
    for seed in (84, 85):
        task = {"id": f"a3-{seed}", "model_id": "quad-fixture", "phase": "a3", "model": model, "seed": seed,
            "recipe_id": "r0", "init_recipe": "default", "batch_size": 2, "sequence_length": 128,
            "stream_chunks": 3, "cpu_threads": 1, "precision": "fp32", "deterministic": False,
            "device": "npu:0",
            "lr": .001, "schedule_tokens": 12 * 2**20, "warmup_tokens": 65536, "lr_floor_ratio": .1,
            "weight_decay": .01, "grad_clip": 1., "auxiliary_loss_weight": .01, "target_tokens": 2 * 2**20,
            "source_identity": old["source_identity"], "resume_checkpoint": "must-not-be-loaded.pt"}
        task_path, result_path = tmp_path / f"task-{seed}.json", tmp_path / f"result-{seed}.json"
        atomic_json(task_path, task)
        atomic_json(result_path, {"status": "completed", "source_identity": old["source_identity"], "task_sha256": sha256(task_path)})
        choices["quad-fixture"]["confirmation"].append({"task_path": str(task_path), "result_path": str(result_path)})
    selected = tmp_path / "selected.json"
    atomic_json(selected, choices)
    return prepared, selected, budget


def test_plan_preserves_selected_recipe_and_reserves_both_sources(snapshots):
    prepared, selected, budget = snapshots
    before = budget.read_bytes()
    plan = probe.build_plan(prepared["old"], prepared["new"], selected)
    task = plan["recipes"]["quad-fixture"]
    assert plan["reserved_tokens"] == 2 * 3 * 2 * 128
    assert task["seed"] == 84 and task["cpu_threads"] == 1 and task["lr"] == .001
    assert task["device"] == "npu:0" and task["target_tokens"] == 3 * 2 * 128
    assert "resume_checkpoint" not in task and "source_identity" not in task
    assert budget.read_bytes() == before


def test_two_real_processes_use_their_own_source_and_three_stateful_updates(snapshots, tmp_path):
    prepared, selected, budget = snapshots
    new = probe.load(prepared["new"])
    old = probe.load(prepared["old"])
    before = {path: sha256(path) for identity in (old["source_identity"], new["source_identity"])
              for path in identity["source_files"]}
    output = tmp_path / "witness"
    command = [sys.executable, str(Path(new["source"]) / "pilotlab/quad_transfer_probe3.py"),
        "--old-prepared", str(prepared["old"]), "--new-prepared", str(prepared["new"]),
        "--selected", str(selected), "--output", str(output), "--timeout-seconds", "180"]
    environment = probe.child_environment(new, {"seed": 84, "cpu_threads": 1})
    completed = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=190)
    assert completed.returncode == 0, (completed.stdout, completed.stderr, (output / "report.json").read_text())
    report = probe.load(output / "report.json")
    assert report["status"] == "passed" and report["exact"]
    assert report["old_source_identity"] != report["new_source_identity"]
    assert report["remaining_descendants"] == []
    assert report["actual_completed_loss_tokens"] == report["reserved_tokens"] == 1536
    assert len(report["children"]) == 2
    observed = {side: probe.checked(report["children"][side + ":quad-fixture"]["result"]) for side in ("old", "new")}
    assert observed["old"]["process"] != observed["new"]["process"]
    for side, value in observed.items():
        own = probe.load(prepared[side])
        assert value["executed_modules"]["pilotlab.worker3"]["path"] == str(Path(own["source"]) / "pilotlab/worker3.py")
        assert value["executed_modules"]["tide.allocation"]["path"] == str(Path(own["foundation"]) / "src/tide/allocation.py")
        assert value["vendor_imported"] is False
        assert value["device"] == "cpu" and value["selected_device"] == "npu:0"
        assert value["state_digest_fields"] == ["model", "optimizer", "private_state", "stream", "rng", "tokens", "steps"]
        assert [row["state_reset"] for row in value["observations"]] == [True, False, False]
        assert len({row["state_sha256"] for row in value["observations"]}) == 3
        assert value["training"] == {"steps": 3, "segment_tokens": 768}
        lifecycle = report["children"][side + ":quad-fixture"]
        assert lifecycle["remaining_descendants"] == []
        assert lifecycle["process"]["start_ticks"] == value["process"]["start_ticks"]
        assert lifecycle["all_seen_processes"]
    assert before == {path: sha256(path) for path in before}
    assert not list(tmp_path.glob("*/source/**/__pycache__"))
    plan = probe.load(output / "plan.json")
    difference = plan["source_comparison"]["foundation_changes"][0]
    review = {"schema": REVIEW_SCHEMA, "status": "passed", "changed_nonshannon_paths": False,
              "quad_pool_contract": "unchanged", "old_allocation_sha256": difference["old_sha256"],
              "new_allocation_sha256": difference["new_sha256"]}
    _validate_review(review, report, plan["source_comparison"], old["source_identity"], new["source_identity"], ["quad-fixture"])
    ledger = probe.load(budget)
    claim = ledger["preparation_token_reservations"][str(output)]
    assert claim["status"] == "passed" and claim["reserved_tokens"] == claim["observed_completed_loss_tokens"] == 1536
    assert ledger["device_seconds"] == 0
    repeated = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0 and "existing probe" in repeated.stderr
    assert probe.load(budget) == ledger


@pytest.mark.parametrize("damage", ["shannon", "source", "batch", "seed", "recipe"])
def test_invalid_selected_evidence_fails_before_budget_or_execution(snapshots, damage):
    prepared, selected, budget = snapshots
    choice = probe.load(selected)["quad-fixture"]
    entry = choice["confirmation"][0]
    task = probe.load(entry["task_path"])
    if damage == "shannon":
        task["model"]["config"]["allocation"] = "shannon"
    elif damage == "source":
        task["source_identity"] = probe.load(prepared["new"])["source_identity"]
    elif damage == "batch":
        task["batch_size"] = 0
    elif damage == "seed":
        task["seed"] = 85
    else:
        task["lr"] = .003
    atomic_json(Path(entry["task_path"]), task)
    result = probe.load(entry["result_path"])
    result.update(task_sha256=sha256(entry["task_path"]), source_identity=task["source_identity"])
    atomic_json(Path(entry["result_path"]), result)
    before = budget.read_bytes()
    with pytest.raises(ValueError):
        probe.build_plan(prepared["old"], prepared["new"], selected)
    assert budget.read_bytes() == before


@pytest.mark.parametrize("damage", ["forward", "backward", "optimizer", "state", "input", "identity", "process", "short"])
def test_equal_exit_codes_cannot_replace_full_exact_bound_observations(damage):
    plan = {"model_ids": ["quad"], "old_source_identity": {"source": "old"}, "new_source_identity": {"source": "new"},
            "recipes": {"quad": {"target_tokens": 768}}}
    results = {}
    for index, side in enumerate(("old", "new")):
        results[side + ":quad"] = {"status": "completed", "source_identity": plan[side + "_source_identity"],
            "model_id": "quad", "side": side, "device": "cpu", "training": {"steps": 3, "segment_tokens": 768},
            "observations": [{"step": i + 1} for i in range(3)], "inputs": [{"sha256": "1" * 64}] * 3,
            "digests": dict.fromkeys(probe.PATHS, "2" * 64), "process": {"pid": 30 + index, "start_ticks": 10},
            "contract": {"seed": 84}, "initial_parameter_sha256": "3" * 64}
    assert not probe.compare_sides(plan, results)[1]
    altered = copy.deepcopy(results)
    if damage in probe.PATHS:
        altered["new:quad"]["digests"][damage] = "4" * 64
    elif damage == "input":
        altered["new:quad"]["inputs"][1] = {"sha256": "5" * 64}
    elif damage == "identity":
        altered["new:quad"]["source_identity"] = plan["old_source_identity"]
    elif damage == "process":
        altered["new:quad"]["process"] = altered["old:quad"]["process"]
    else:
        altered["new:quad"]["training"]["steps"] = 2
    assert probe.compare_sides(plan, altered)[1]


def test_help_and_import_never_import_torch_or_a_vendor():
    script = "import sys; import pilotlab.quad_transfer_probe3; assert 'torch' not in sys.modules; assert 'torch_npu' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))


def test_probe_tracks_and_reaps_a_detached_descendant_after_its_owner_exits():
    script = r'''
import subprocess, sys, time
from pilotlab.scheduler import child_subreaper
from pilotlab.quad_transfer_probe3 import observe_descendants, cleanup_descendants
previous = child_subreaper(True)
result = {"children": {"child": {}}}
code = "import os,time; pid=os.fork(); os._exit(0) if pid else None; os.setsid(); time.sleep(30)"
process = subprocess.Popen([sys.executable, "-c", code])
process.wait(timeout=10)
try:
    assert observe_descendants(result, "child")
    assert cleanup_descendants(result, current_key="child")
    assert result["all_seen_processes"]
    assert result["remaining_descendants"] == []
    assert result["children"]["child"]["remaining_descendants"] == []
finally:
    cleanup_descendants(result, current_key="child")
    child_subreaper(previous)
'''
    subprocess.run([sys.executable, "-c", script], check=True, timeout=20,
                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
