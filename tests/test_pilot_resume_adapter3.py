"""A real CPU interruption/continuation and fail-closed evidence admission."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
import torch

from pilotlab import resume_adapter3 as adapter
from pilotlab import worker3
from pilotlab.data3 import prepare
from pilotlab.io import sha256
from pilotlab.worker import digest_tree
from test_pilot_data3 import write_source
from test_pilot_worker3 import ROOT, run_worker, task_spec


@pytest.fixture(scope="module")
def interrupted(tmp_path_factory):
    root = tmp_path_factory.mktemp("bound-a1")
    data = root / "data"
    prepare(write_source(root / "corpus"), data)
    task = task_spec(data, target_tokens=128, eval_interval_tokens=256, cpu_threads=1, device="cpu")
    result, payload, output = run_worker(root, data, task, interrupt=True)
    assert result["status"] == "budget_exhausted" and result["stop_reason"] == "signal"
    assert payload["tokens"] == 32 and result["validation_initial"]["tokens"] == 16
    assert result["validation_final"]["tokens"] == 0
    return {"root": root, "data": data, "task": task, "result": result, "output": output,
            "task_path": root / "parent.json"}


def make_receipt(sample):
    return {"schema": adapter.SCHEMA, "task_id": sample["task"]["id"],
            "original_task": {"path": str(sample["task_path"]), "sha256": sha256(sample["task_path"])},
            "result": {"path": str(sample["output"] / "result.json"), "sha256": sha256(sample["output"] / "result.json")},
            "checkpoint": copy.deepcopy(sample["result"]["checkpoint"]),
            "source_identity": copy.deepcopy(sample["task"]["source_identity"]),
            "original_initial_evaluation": copy.deepcopy(sample["result"]["validation_initial"]),
            "evaluation_additional_reservation_tokens": 32}


def arguments(tmp_path, sample, receipt=None, **changes):
    value = make_receipt(sample) if receipt is None else receipt
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(value))
    args = argparse.Namespace(task=sample["task_path"], output=tmp_path / "resumed", data=sample["data"],
        device="cpu", deadline_unix=time.time() + 120, resume_checkpoint=Path(value["checkpoint"]["path"]),
        receipt=receipt_path, receipt_sha256=sha256(receipt_path), adapter_sha256=sha256(adapter.__file__))
    for key, val in changes.items():
        setattr(args, key, val)
    return args


def run_adapter(args, script=None):
    command = [sys.executable, str(script or adapter.__file__)]
    for name in ("task", "output", "data", "device", "deadline_unix", "resume_checkpoint", "receipt", "receipt_sha256", "adapter_sha256"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", ASCEND_RT_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    foundation = Path(next(iter(importlib.util.find_spec("tide").submodule_search_locations))).resolve().parent
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(foundation)))
    return subprocess.run(command, cwd=args.receipt.parent, env=env, text=True, capture_output=True, timeout=120)


def test_standalone_adapter_preserves_real_cpu_state_and_only_evaluates_final(tmp_path, interrupted):
    args = arguments(tmp_path, interrupted)
    # The entry point can live outside the package containing the training core.
    script = tmp_path / "controller-resume.py"
    shutil.copyfile(adapter.__file__, script)
    before = {path: sha256(path) for path in (interrupted["task_path"], interrupted["output"] / "result.json",
                                              interrupted["result"]["checkpoint"]["path"])}
    process = run_adapter(args, script)
    assert process.returncode == 0, (process.stdout, process.stderr)
    resumed = json.loads((args.output / "result.json").read_text())
    baseline, expected, baseline_path = run_worker(tmp_path, interrupted["data"], dict(interrupted["task"], id="baseline"))
    assert baseline["status"] == resumed["status"] == "completed"
    actual = torch.load(resumed["checkpoint"]["path"], map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "private_state", "stream", "rng"):
        assert digest_tree(actual[key]) == digest_tree(expected[key]), key
    assert actual["state_sha256"] == expected["state_sha256"]
    assert resumed["training"]["from_tokens"] == 32 and resumed["training"]["segment_tokens"] == 96
    assert resumed["resume"]["checkpoint_load"]["status"] == "exact"
    assert resumed["recovery"]["numeric_trajectory"]["status"] == "not_tested"
    assert resumed["validation_initial"]["tokens"] == 0
    assert resumed["validation_initial"]["original_observed_tokens"] == 16
    assert resumed["validation_initial"]["exposure"] == 0
    assert resumed["validation_initial"]["nll"] == interrupted["result"]["validation_initial"]["nll"]
    assert resumed["validation_final"]["tokens"] == resumed["evaluation"]["tokens"] == 16
    assert resumed["evaluation"]["cumulative_task_observed_tokens"] == 32
    assert resumed["validation_final"]["nll"] == baseline["validation_final"]["nll"]
    assert resumed["source_identity"] == interrupted["task"]["source_identity"]
    execution = resumed["execution_adapter"]
    assert execution["adapter"] == {"path": str(script), "sha256": sha256(script)}
    assert execution["worker"]["path"] == str(ROOT / "pilotlab/worker3.py")
    assert execution["receipt"]["sha256"] == args.receipt_sha256
    assert execution["evaluation_additional_reservation_tokens"] == 32
    resumed_rows = [json.loads(line) for line in (args.output / "steps.jsonl").read_text().splitlines()]
    baseline_rows = [json.loads(line) for line in (baseline_path / "steps.jsonl").read_text().splitlines()][2:]
    for left, right in zip(resumed_rows, baseline_rows, strict=True):
        for key in ("step", "tokens", "loss", "lr", "input_sha256", "rng_before_sha256", "rng_after_sha256", "state_reset"):
            assert left[key] == right[key], key
    assert before == {path: sha256(path) for path in before}


@pytest.mark.parametrize("damage", ["adapter_hash", "receipt_hash", "task_hash", "result_hash", "checkpoint_hash",
                                    "receipt_source", "initial_copy", "reservation", "cli_checkpoint", "output_reuse"])
def test_changed_receipt_or_artifacts_never_enter_worker(tmp_path, interrupted, monkeypatch, damage):
    receipt = make_receipt(interrupted)
    changes = {}
    if damage == "adapter_hash":
        changes["adapter_sha256"] = "0" * 64
    elif damage == "receipt_hash":
        changes["receipt_sha256"] = "0" * 64
    elif damage in {"task_hash", "result_hash", "checkpoint_hash"}:
        receipt[{"task_hash": "original_task", "result_hash": "result", "checkpoint_hash": "checkpoint"}[damage]]["sha256"] = "0" * 64
    elif damage == "receipt_source":
        receipt["source_identity"] = {"different": True}
    elif damage == "initial_copy":
        receipt["original_initial_evaluation"]["nll"] += 1
    elif damage == "reservation":
        receipt["evaluation_additional_reservation_tokens"] = 0
    elif damage == "cli_checkpoint":
        changes["resume_checkpoint"] = tmp_path / "different.pt"
    else:
        changes["output"] = interrupted["output"]
    args = arguments(tmp_path, interrupted, receipt, **changes)
    monkeypatch.setattr(worker3, "execute", lambda _: pytest.fail("invalid receipt entered the training worker"))
    with pytest.raises((ValueError, OSError)):
        adapter.execute(args)
    if damage != "output_reuse":
        assert not args.output.exists()


@pytest.mark.parametrize("damage", ["completed", "not_signal", "wrong_task", "source", "initial_incomplete",
                                    "evaluation_accounting", "unrelated_health_failure", "cp_exposure"])
def test_rehashed_but_inapplicable_result_is_rejected(tmp_path, interrupted, monkeypatch, damage):
    receipt = make_receipt(interrupted)
    result = copy.deepcopy(interrupted["result"])
    if damage == "completed":
        result["status"] = "completed"
    elif damage == "not_signal":
        result["stop_reason"] = "deadline_or_incomplete_checks"
    elif damage == "wrong_task":
        result["id"] = "another-task"
    elif damage == "source":
        result["source_identity"] = {"different": True}
    elif damage == "initial_incomplete":
        result["validation_initial"]["status"] = "budget_exhausted"
        receipt["original_initial_evaluation"] = copy.deepcopy(result["validation_initial"])
    elif damage == "evaluation_accounting":
        result["evaluation"]["tokens"] += 8
    elif damage == "unrelated_health_failure":
        result["health"]["blocking_reasons"].append("nonfinite gradient")
    else:
        result["training"]["tokens"] += 16
    path = tmp_path / "changed-result.json"
    path.write_text(json.dumps(result))
    receipt["result"] = {"path": str(path), "sha256": sha256(path)}
    # A sibling directory avoids triggering the independent output guard first.
    args = arguments(tmp_path, interrupted, receipt, output=tmp_path.parent / (tmp_path.name + "-attempt"))
    monkeypatch.setattr(worker3, "execute", lambda _: pytest.fail("inapplicable result entered the training worker"))
    with pytest.raises(ValueError):
        adapter.execute(args)
    assert not args.output.exists()


def test_bound_checkpoint_cannot_fall_back_after_admission(tmp_path, interrupted, monkeypatch):
    args = arguments(tmp_path, interrupted)
    load = worker3.load_checkpoint
    def fallback(*items, **keywords):
        payload, evidence = load(*items, **keywords)
        return payload, {**evidence, "fallback_used": True}
    monkeypatch.setattr(worker3, "load_checkpoint", fallback)
    result = adapter.execute(args)
    assert result["status"] == "failed"
    assert "forbids fallback" in result["error"]
    assert result["training"]["segment_tokens"] == 0
    assert not result["execution_adapter"]["initial_evaluation_inherited"]


def test_expired_deadline_preserves_noncompletion_and_does_not_relabel_final_as_initial(tmp_path, interrupted):
    args = arguments(tmp_path, interrupted, deadline_unix=time.time() - 1)
    process = run_adapter(args)
    assert process.returncode == 2, (process.stdout, process.stderr)
    result = json.loads((args.output / "result.json").read_text())
    assert result["status"] == "budget_exhausted"
    assert not result["execution_adapter"]["initial_evaluation_inherited"]
    assert result["evaluation"]["tokens"] == 0
    assert "validation_initial" not in result
    assert result["validation_final"]["status"] == "budget_exhausted"
    assert result["training"]["segment_tokens"] == 0


def test_cli_refusal_logs_nonzero_without_creating_an_attempt(tmp_path, interrupted):
    args = arguments(tmp_path, interrupted, receipt_sha256="0" * 64)
    process = run_adapter(args)
    assert process.returncode == 1 and "resume_adapter_refused" in process.stderr
    assert not args.output.exists()


def test_target_reached_before_signal_only_completes_final_evaluation(tmp_path, interrupted):
    task = dict(interrupted["task"], id="partial-final", target_tokens=32, batch_size=1)
    task_path, output = tmp_path / "partial-final.json", tmp_path / "original"
    task_path.write_text(json.dumps(task))
    # Deliver SIGTERM after a real final-evaluation batch; the original loop
    # itself publishes the partial count, status and full training checkpoint.
    harness = """
import os,signal
from pilotlab import worker3
original=worker3.evaluate
calls=0
def evaluate(model,*args,**kwargs):
    global calls
    calls+=1
    if calls!=2:
        return original(model,*args,**kwargs)
    forward=model.forward
    def stop_after_batch(*a,**k):
        value=forward(*a,**k)
        os.kill(os.getpid(),signal.SIGTERM)
        return value
    model.forward=stop_after_batch
    try:
        return original(model,*args,**kwargs)
    finally:
        model.forward=forward
worker3.evaluate=evaluate
raise SystemExit(worker3.main())
"""
    environment = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    process = subprocess.run([sys.executable, "-c", harness, "--task", str(task_path), "--output", str(output),
        "--data", str(interrupted["data"]), "--device", "cpu", "--deadline-unix", str(time.time()+120)],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
    assert process.returncode == 2, (process.stdout, process.stderr)
    previous = json.loads((output / "result.json").read_text())
    assert previous["training"]["tokens"] == 32
    assert previous["validation_final"]["tokens"] == 8
    sample = {**interrupted, "task": task, "task_path": task_path, "result": previous, "output": output}
    receipt = make_receipt(sample)
    receipt["evaluation_additional_reservation_tokens"] = 32+8
    args = arguments(tmp_path, sample, receipt)
    completed = run_adapter(args)
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    result = json.loads((args.output / "result.json").read_text())
    assert result["training"]["from_tokens"] == result["training"]["tokens"] == 32
    assert result["training"]["segment_tokens"] == result["training"]["segment_steps"] == 0
    assert result["checkpoint"]["state_sha256"] == previous["checkpoint"]["state_sha256"]
    assert result["evaluation"]["tokens"] == 16
    assert result["evaluation"]["cumulative_task_observed_tokens"] == 16+8+16
    assert result["execution_adapter"]["evaluation_additional_reservation_tokens"] == 40
