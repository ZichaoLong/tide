"""Pure-record review checks; no torch, device, checkpoint load or experiment."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pilotlab.report2 import build_report, canonical_sha256, main, record_identity, render_markdown
from pilotlab.selection import resolve_selection


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Records:
    def __init__(self, root):
        self.root = root / "round"
        self.source = root / "frozen-v2" / "source"
        self.foundation = root / "frozen" / "foundation" / "src"
        self.tasks, self.records = [], {}
        files = {}
        for name in ("worker.py", "models.py", "data.py", "io.py"):
            path = self.source / "pilotlab" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic execution source\n")
            files[name] = record_identity(path)["sha256"]
        path = self.foundation / "tide" / "__init__.py"
        path.parent.mkdir(parents=True)
        path.write_text("# synthetic foundation\n")
        foundation_files = {"__init__.py": record_identity(path)["sha256"]}
        self.identity = {"experiment_sha256": canonical_sha256(files), "experiment_files": files,
                         "foundation_sha256": canonical_sha256(foundation_files), "foundation_files": foundation_files,
                         "data_manifest_sha256": digest("synthetic data"), "runtime_packages": {"torch": "fixture"}}
        write(self.source.parent / "identity.json", {
            "marker": "correct-v2", "source_sha256": {"source/pilotlab/" + name: value for name, value in files.items()}})
        write(root / "frozen" / "identity.json", {"marker": "wrong-old-freeze"})

    def task(self, identity, **changes):
        task = {"id": identity, "phase": "mixed", "backend": "npu", "seed": 1, "cpu_threads": 2,
                "cpu_affinity": [0, 1], "model": {"family": "transformer", "d": 16},
                "batch_size": 2, "sequence_length": 4, "stream_chunks": 4,
                "lr": 0.0, "warmup_steps": 1, "weight_decay": .01, "grad_clip": 1.0,
                "precision": "fp32", "measurement_warmup_steps": 1, "eval_batches": 16, "decode_steps": 1,
                "checkpoint_interval_steps": 128, "resume_check": True, "resume_verify_steps": 8,
                "checkpoint_identity": self.identity, "purpose": "cost_calibration", "max_steps": 4}
        task.update(changes)
        return task

    def add(self, task, *, declared=None, start=1100., seconds=2., physical=3, nll=3.,
            from_step=0, selection=None, continuation=None, rows_override=None, no_result=False):
        task = copy.deepcopy(task)
        declared = copy.deepcopy(declared or task)
        identity, count = task["id"], task["max_steps"]
        directory = self.root / "tasks" / identity
        self.tasks.append(declared)
        write(directory / "declared-task.json", declared)
        write(directory / "task.json", task)
        resolution = {"declared_task_sha256": canonical_sha256(declared),
                      "resolved_task_sha256": canonical_sha256(task)}
        if selection is not None:
            write(directory / "selection.json", selection)
            parent = selection["selected_parent"]
            resolution["selection"] = {"path": str(directory / "selection.json"), "sha256": canonical_sha256(selection)}
            resolution["resume_parent"] = {"task_id": parent, "sha256": digest(parent), "step": 257}
        process = {"id": identity, "backend": task["backend"], "phase": task["phase"], "status": "completed",
                   "cpu_affinity": task["cpu_affinity"], "descendants_terminated": True,
                   "process_started_unix": start - 10, "process_ended_unix": start + seconds * count + 10,
                   "task_resolution": resolution, "task_file_sha256": record_identity(directory / "task.json")["sha256"]}
        if no_result:
            process["status"] = "failed"
        self.records[identity] = process
        write(directory / "process.json", process)
        write(directory / "worker.json", {"pid": 100, "started_unix": start - 5, "affinity": task["cpu_affinity"]})
        if no_result:
            return
        comparisons = {key: {"atol": 1e-5, "rtol": 1e-4, "exact": True, "finite": True,
                             "within_tolerance": True, "structure_equal": True, "integer_equal": True,
                             "max_abs": 0., "fields": 4, "failed_fields": [], "failed_field_count": 0}
                       for key in ("model", "optimizer", "private_state", "logits", "gradients")}
        recovery_rows = [{"step": from_step + count + index + 1, "within_tolerance": True,
                          "next_input_exact": True, "rng_before_exact": True, "rng_after_exact": True,
                          "stream_after_exact": True, "step_exact": True, "state_reset_exact": True,
                          "continued_private_state": True, "comparisons": comparisons}
                         for index in range(task["resume_verify_steps"])]
        recovery = {"status": "within_tolerance", "from_step": from_step + count,
                    "to_step": from_step + count + task["resume_verify_steps"],
                    "verified_steps": task["resume_verify_steps"], "requested_steps": task["resume_verify_steps"],
                    "saved_data_exact": True, "next_input_exact": True, "rng_before_exact": True,
                    "rng_after_exact": True, "stream_after_exact": True, "continued_private_state": True,
                    "comparisons": comparisons, "per_step": recovery_rows,
                    "fresh_pid_result": {"pid": 101, "validation": {"identity_exact": True,
                        "contract_exact": True, "state_digest_exact": True}}}
        token_count = task["batch_size"] * task["sequence_length"]
        result = {"status": "completed", "task": task, "checkpoint_identity": self.identity,
                  "data_manifest_sha256": self.identity["data_manifest_sha256"], "cpu_threads": task["cpu_threads"],
                  "started_unix": start - 5, "ended_unix": start + seconds * count + 5,
                  "device": "npu:0" if task["backend"] == "npu" else "cpu",
                  "physical_visibility": str(physical) if task["backend"] == "npu" else "",
                  "training": {"steps": count, "from_step": from_step, "to_step": from_step + count,
                               "global_steps": from_step + count, "tokens": count * token_count,
                               "tokens_per_second": token_count / seconds, "step_seconds": [seconds] * count},
                  "training_memory": {"cpu_rss_peak_bytes": 100, "npu_peak_allocated_bytes": 200},
                  "parameter_updates": {"changed_tensor_count": 0, "total_tensor_count": 10},
                  "validation_initial": {"nll": nll + .1, "tokens": 256},
                  "validation_final": {"nll": nll, "tokens": 256}, "recovery": recovery,
                  "checkpoint": {"path": str(directory / "checkpoint.pt"), "sha256": digest(identity),
                                 "step": from_step + count, "identity": self.identity},
                  "continuation": continuation}
        write(directory / "result.json", result)
        rows = [{"step": from_step + index + 1, "tokens": token_count, "seconds": seconds,
                 "time_unix": start + seconds * (index + 1), "state_reset": index % 4 == 0,
                 "input_sha256": digest(f"input{index}"), "rng_before_sha256": digest(f"before{index}"),
                 "rng_after_sha256": digest(f"after{index}")} for index in range(count)]
        with (directory / "steps.jsonl").open("w") as stream:
            for row in rows if rows_override is None else rows_override:
                stream.write(json.dumps(row) + "\n")

    def finish(self, phases=None):
        phases = phases or [{"id": "isolated", "cpu_slots": 0, "npu_slots": 1},
                  {"id": "mixed", "cpu_slots": 1, "npu_slots": 2},
                  {"id": "lr-screen", "cpu_slots": 0, "npu_slots": 16},
                  {"id": "selected-continuation", "cpu_slots": 0, "npu_slots": 16}]
        plan = {"round_id": "synthetic-r2", "tasks": self.tasks, "phases": phases,
                "foundation_source": str(self.foundation)}
        write(self.root / "plan.json", plan)
        scheduler = {"status": "completed", "started_unix": 900, "updated_unix": 10000,
                     "all_task_processes_terminated": True, "tasks": list(self.records.values()),
                     "source": str(self.source), "foundation_source": str(self.foundation),
                     "plan_sha256": record_identity(self.root / "plan.json")["sha256"],
                     "phases": [dict(phase, status="completed", started_unix=1000, ended_unix=1200) for phase in phases],
                     "memory_guard": {"samples": 3, "peak_rss_bytes": 200,
                        "enforcement": "sampled_process_rss_software_watchdog", "kernel_memory_limit_established": False,
                        "effective": {"poll_seconds": 1}, "rss_scope": "scheduler_and_descendants"}}
        write(self.root / "scheduler.json", scheduler)
        memory = [{"sample_started_unix": 1000 + index, "scan_seconds": .01 * (index + 1),
                   "interval_since_previous_seconds": None if index == 0 else 1.2,
                   "total_rss_bytes": peak, "host": {"available_bytes": 1000 - peak},
                   "task_rss_bytes": {identity: 10 + index for identity in self.records}}
                  for index, peak in enumerate((100, 200, 180))]
        (self.root / "memory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in memory))
        return self.root


def paired_fixture(tmp_path):
    records = Records(tmp_path)
    records.add(records.task("isolated", phase="isolated", comparison_group="npu-one", comparison_role="isolated"),
                start=1000, seconds=1)
    records.add(records.task("concurrent", comparison_group="npu-one", comparison_role="concurrent"))
    records.add(records.task("peer-cpu", backend="cpu", cpu_affinity=[4, 5]))
    records.add(records.task("peer-npu", cpu_affinity=[6, 7]), physical=5)
    return records


def mutate(path, update):
    value = json.loads(path.read_text())
    update(value)
    write(path, value)


def test_matched_prefix_actual_training_overlap_and_current_freeze(tmp_path):
    records = paired_fixture(tmp_path)
    report = build_report(records.finish(), generated_unix=10001)
    pair = report["paired_comparisons"][0]
    assert pair["throughput_ratio"] == .5
    assert pair["prefix"]["common_step_count"] == 4 and pair["prefix"]["measured_prefix_steps"] == 3
    assert pair["training_overlap"]["planned_concurrency_fraction"] == 1
    assert pair["training_overlap"]["peak_simultaneous_training_tasks"] == 3
    assert report["strict_recovery_counts"]["qualified_at_least_eight_steps"] == 4
    assert report["memory_guard"]["sampled_peak_rss_bytes"] == 200
    assert report["memory_guard"]["actual_scan_interval_seconds"]["median"] == 1.2
    assert report["token_exposure"]["cost"]["tokens"] == 128
    assert report["token_exposure"]["learning"]["tokens"] is None
    assert not report["mechanism_observations"]
    provenance = report["source_provenance"]
    assert provenance["frozen_identity"]["marker"] == "correct-v2"
    assert provenance["manifest_source_files_validation"]["all_observed_files_match"]
    assert provenance["round_plan_file_matches_scheduler_sha256"]
    assert all(value["experiment"]["all_observed_files_match"] and value["foundation"]["all_observed_files_match"]
               for value in provenance["checkpoint_source_file_validations"].values())
    assert report["review_ready"]
    encoded = json.dumps(report)
    assert "raw_records" not in encoded and '"step_seconds"' not in encoded
    assert "wrong-old-freeze" not in encoded
    assert "成本计算" in render_markdown(report)


@pytest.mark.parametrize("change", ["seed", "cpu-mask", "physical", "input", "rng", "missing-rng", "source"])
def test_mismatched_comparison_never_receives_ratio(tmp_path, change):
    records = paired_fixture(tmp_path)
    root = records.finish()
    directory = root / "tasks" / "concurrent"
    if change == "seed":
        mutate(directory / "task.json", lambda value: value.update(seed=2))
    elif change == "cpu-mask":
        mutate(directory / "worker.json", lambda value: value.update(affinity=[2, 3]))
    elif change == "physical":
        mutate(directory / "result.json", lambda value: value.update(physical_visibility="4"))
    elif change == "source":
        mutate(directory / "result.json", lambda value: value["checkpoint_identity"].update(experiment_sha256=digest("different")))
    else:
        path = directory / "steps.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "input":
            rows[2]["input_sha256"] = digest("different")
        elif change == "rng":
            rows[2]["rng_before_sha256"] = digest("different")
        else:
            rows[2].pop("rng_after_sha256")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    pair = build_report(root)["paired_comparisons"][0]
    assert pair["throughput_ratio"] is None and pair["status"] == "not_qualified"


def test_worker_overlap_cannot_substitute_for_training_overlap(tmp_path):
    records = paired_fixture(tmp_path)
    root = records.finish()
    for identity in ("peer-cpu", "peer-npu"):
        path = root / "tasks" / identity / "steps.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["time_unix"] += 100
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = build_report(root)
    assert report["worker_concurrency"]["all"]["peak"] == 3
    pair = report["paired_comparisons"][0]
    assert pair["training_overlap"]["peak_simultaneous_training_tasks"] == 1
    assert pair["throughput_ratio"] is None
    assert "concurrent_training_overlap_not_observed" in pair["reasons"]


def test_cpu_repeats_share_one_baseline_but_keep_separate_overlap_evidence(tmp_path):
    records = Records(tmp_path)
    records.add(records.task("cpu-isolated", backend="cpu", phase="isolated",
                             comparison_group="cpu-one", comparison_role="isolated"), start=1000, seconds=1)
    phases = [{"id": "isolated", "cpu_slots": 1, "npu_slots": 0}]
    for repeat, start in (("a", 1100), ("b", 1200)):
        phase = "mixed-" + repeat
        phases.append({"id": phase, "cpu_slots": 2, "npu_slots": 0})
        records.add(records.task("cpu-" + repeat, backend="cpu", phase=phase,
                                 comparison_group="cpu-one", comparison_role="concurrent"), start=start)
        records.add(records.task("peer-" + repeat, backend="cpu", phase=phase, cpu_affinity=[4, 5]), start=start)
    pairs = build_report(records.finish(phases))["paired_comparisons"]
    assert len(pairs) == 2
    assert all(pair["isolated_id"] == "cpu-isolated" and pair["throughput_ratio"] == .5 for pair in pairs)
    assert [pair["training_overlap"]["overlapping_tasks"][0]["task_id"] for pair in pairs] == ["peer-a", "peer-b"]
    assert all("not captured" in pair["activity_scope"] for pair in pairs)


def test_underfilled_phase_is_labelled_and_missing_measurements_stay_unknown(tmp_path):
    records = paired_fixture(tmp_path)
    path = records.root / "tasks" / "peer-npu" / "steps.jsonl"
    path.unlink()
    records.add(records.task("failed-cost"), no_result=True)
    report = build_report(records.finish())
    pair = report["paired_comparisons"][0]
    assert pair["throughput_ratio"] == .5
    assert pair["target_load_status"] == "insufficient_or_variable_training_concurrency"
    assert pair["training_overlap"]["planned_concurrency_fraction"] == 0
    missing = next(task for task in report["tasks"] if task["id"] == "failed-cost")
    assert missing["training"] is None and missing["recovery"]["recorded_steps"] is None
    assert report["token_exposure"]["cost"]["missing_training_records"] == 1
    assert report["token_exposure"]["cost"]["tokens"] == 128


def test_per_step_discrete_or_rng_failure_blocks_strict_recovery(tmp_path):
    records = paired_fixture(tmp_path)
    root = records.finish()
    path = root / "tasks" / "isolated" / "result.json"
    def fail(value):
        row = value["recovery"]["per_step"][4]
        row["rng_before_exact"] = False
        row["comparisons"]["private_state"]["integer_equal"] = False
    mutate(path, fail)
    task = next(task for task in build_report(root)["tasks"] if task["id"] == "isolated")
    assert not task["recovery"]["eight_step_qualified"]
    assert task["recovery"]["failed_check_count"] == 2


def test_source_drift_is_retained_and_running_round_cannot_publish(tmp_path):
    records = paired_fixture(tmp_path)
    root = records.finish()
    (records.source / "pilotlab" / "worker.py").write_text("# changed after the recorded run\n")
    report = build_report(root)
    validation = report["source_provenance"]["manifest_source_files_validation"]
    assert not validation["all_observed_files_match"]
    assert validation["mismatches"][0]["path"] == "source/pilotlab/worker.py"
    mutate(root / "scheduler.json", lambda value: value.update(status="running"))
    with pytest.raises(ValueError, match="terminal scheduler"):
        main(["--round", str(root), "--output", str(tmp_path / "not-written.json"),
              "--markdown", str(tmp_path / "not-written.md")])
    assert not (tmp_path / "not-written.json").exists()


def test_eighteen_parents_six_selection_records_and_257_to_1025(tmp_path):
    records = Records(tmp_path)
    families = {}
    for family in ("fc", "x", "tf"):
        ids = []
        for lr in (.0001, .0003, .001):
            for seed in (1, 2):
                identity = f"screen-{family}-{lr}-{seed}"
                task = records.task(identity, phase="lr-screen", model={"family": family}, lr=lr, seed=seed,
                                    max_steps=257, purpose="equal_token_lr_screen", selection_family=family)
                records.add(task, nll={.0001: 4, .0003: 3, .001: 5}[lr] + seed * .01)
                ids.append(identity)
        families[family] = ids
    for family, candidates in families.items():
        for seed in (1, 2):
            identity = f"continue-{family}-{seed}"
            task = records.task(identity, phase="selected-continuation", model={"family": family}, lr=.0003,
                seed=seed, max_steps=768, purpose="selected_multistep_continuation", selection_family=family,
                depends_on=candidates, resume_select={"rule": "lowest_mean_validation_nll", "candidates": candidates,
                                                     "seeds": [1, 2], "updates": 257},
                resume_allow_budget_overrides=["max_steps"])
            resolved, selection = resolve_selection(task, records.records, records.root)
            parent = selection["selected_parent"]
            resolved.update(resume_from=str(records.root / "tasks" / parent / "checkpoint.pt"),
                            resume_checkpoint_sha256=digest(parent))
            records.add(resolved, declared=task, selection=selection, from_step=257,
                        continuation={"sha256": digest(parent), "identity_exact": True, "contract_exact": True,
                                      "state_digest_exact": True, "first_input_reused_private_history": True,
                                      "budget_overrides": ["max_steps"]})
    report = build_report(records.finish())
    selection = report["selection"]
    assert selection["parent_count"] == 18 and selection["selection_records_observed"] == 6
    assert selection["continuation_count"] == 6
    for child in selection["continuations"]:
        assert child["selection_recomputed_exact"]
        assert child["winner"]["lr"] == .0003
        assert child["winner"]["mean_validation_nll"] == pytest.approx(3.015)
        assert child["training"] == {"steps": 768, "tokens": 6144, "from_step": 257, "to_step": 1025, "global_steps": 1025}
        assert child["planned_range"]["target_global_steps"] == 1025
        assert child["inheritance"]["recipe_inherited"] and child["inheritance"]["recorded_checkpoint_hashes_consistent"]
        assert child["inheritance"]["global_start_matches_parent_checkpoint"]
        assert child["strict_eight_step_recovery"] and child["winner_all_parents_strict_eight_step"]
    assert report["token_exposure"]["learning"]["tokens"] == (18 * 257 + 6 * 768) * 8
    assert all(task["configuration"]["declared_matches_plan"] for task in report["tasks"])
    assert len(json.dumps(report)) < 600_000
