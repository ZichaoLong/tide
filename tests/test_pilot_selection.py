import copy
import json

import pytest

from pilotlab.selection import resolve_selection


def fixture(tmp_path):
    tasks = {}
    records = {}
    for lr in (0.0001, 0.0003, 0.001):
        for seed in (1, 2):
            identity = f"lr{lr}-s{seed}"
            candidate = dict(id=identity, lr=lr, seed=seed, model={"family": "test"},
                             backend="cpu", batch_size=2, sequence_length=4)
            result = dict(status="completed", training={"steps": 3, "tokens": 24},
                          recovery={"status": "within_tolerance"},
                          validation_final={"nll": {0.0001: 4, 0.0003: 3, 0.001: 5}[lr] + seed / 10})
            directory = tmp_path / "tasks" / identity
            directory.mkdir(parents=True)
            (directory / "task.json").write_text(json.dumps(candidate))
            (directory / "result.json").write_text(json.dumps(result))
            tasks[identity] = candidate
            records[identity] = {"status": "completed"}
    task = dict(tasks[next(iter(tasks))], id="continue", max_steps=9, cpu_affinity=[4, 5],
                depends_on=list(tasks), resume_select={"rule": "lowest_mean_validation_nll",
                "candidates": list(tasks), "seeds": [1, 2], "updates": 3})
    return task, records


def mutate_result(tmp_path, identity, mutate):
    path = tmp_path / "tasks" / identity / "result.json"
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))


def test_complete_seed_mean_and_resource_preservation(tmp_path):
    task, records = fixture(tmp_path)
    original = copy.deepcopy(task)
    result, review = resolve_selection(task, records, tmp_path)
    assert result["lr"] == .0003 and result["resume_task"] == "lr0.0003-s1"
    assert result["max_steps"] == 9 and result["cpu_affinity"] == [4, 5]
    assert task == original and len(review["candidates"]) == 6


@pytest.mark.parametrize("failure", ["execution", "recovery", "exposure", "nan"])
def test_bad_seed_disqualifies_whole_lr_group(tmp_path, failure):
    task, records = fixture(tmp_path)
    identity = "lr0.0003-s2"
    if failure == "execution":
        records[identity]["status"] = "failed"
    elif failure == "recovery":
        mutate_result(tmp_path, identity, lambda r: r["recovery"].update(status="comparison_failed"))
    elif failure == "exposure":
        mutate_result(tmp_path, identity, lambda r: r["training"].update(steps=2, tokens=16))
    else:
        mutate_result(tmp_path, identity, lambda r: r["validation_final"].update(nll=float("nan")))
    result, review = resolve_selection(task, records, tmp_path)
    assert result["lr"] == .0001
    assert not next(g for g in review["groups"] if g["lr"] == .0003)["eligible"]


def test_rejects_mixed_workload_and_nonterminal(tmp_path):
    task, records = fixture(tmp_path)
    records["lr0.0001-s1"]["status"] = "running"
    with pytest.raises(ValueError, match="terminal"):
        resolve_selection(task, records, tmp_path)
    records["lr0.0001-s1"]["status"] = "completed"
    path = tmp_path / "tasks/lr0.0001-s1/task.json"
    candidate = json.loads(path.read_text())
    candidate["batch_size"] = 4
    path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="comparison contract"):
        resolve_selection(task, records, tmp_path)


def test_no_valid_candidate_is_failure(tmp_path):
    task, records = fixture(tmp_path)
    for record in records.values():
        record["status"] = "failed"
    with pytest.raises(ValueError, match="no complete"):
        resolve_selection(task, records, tmp_path)
