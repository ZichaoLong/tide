"""Finite CPU simulations of the campaign DAG; no worker or device is started.

The real selection, pruning, persistence and A audit operate on tiny synthetic
worker artifacts. Only execution, capacity matching and the 216-model count
boundary are substituted; gate3 has separate full-cardinality fault tests.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import heapq
import json
from pathlib import Path

import pytest

from pilotlab import gate3 as g
from pilotlab import round3 as r
from pilotlab import scheduler3 as s


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


class SimulatedSupervisor:
    injection = {}

    def __init__(self, config, output):
        self.config, self.output = config, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / "scheduler.json"
        self.state = s.read(self.state_path, {"status": "preparing", "stage_b_gate": "closed", "tasks": {}})
        self.journal_path = self.output / "simulation.json"
        self.journal = s.read(self.journal_path, {"calls": [], "executions": {}, "physical_tasks": []})
        self.resource_cache = {"available_storage_bytes": 10**12}
        self.stop_reason = None

    def _save(self):
        self.state["counts"] = dict(Counter(row["status"] for row in self.state["tasks"].values()))
        write_json(self.state_path, self.state)
        write_json(self.journal_path, self.journal)

    def run(self, stage, tasks, **kwargs):
        self.journal["calls"].append(stage)
        self._save()
        if self.injection.get("crash_before") == stage:
            self.injection.pop("crash_before")
            raise RuntimeError("injected controller restart")
        answers = {}
        for request in tasks:
            tid = request["id"]
            task_path = r.freeze_json(self.output / "tasks" / tid / "task.json", request)
            old = self.state["tasks"].get(tid)
            if old:
                assert old["status"] == "completed"
                assert old["stage"] == stage
                assert s.read(old["task_path"]) == request
                assert s.sha256(old["result_path"]) == old["result_sha256"]
                answers[tid] = s.read(old["result_path"])
                assert answers[tid]["task_sha256"] == s.sha256(task_path)
                continue
            self.journal["executions"][tid] = self.journal["executions"].get(tid, 0) + 1
            self.journal["physical_tasks"].append(deepcopy(request))
            directory = self.output / "tasks" / tid / "attempt-0"
            checkpoint = directory / "checkpoints" / "final.pt"
            previous = None
            if request.get("resume_checkpoint"):
                assert s.sha256(request["resume_checkpoint"]) == request["resume_checkpoint_sha256"]
                previous = s.read(request["resume_checkpoint"])
                for key in ("seed", "model_id", "recipe_id"):
                    assert previous.get(key) == request.get(key)
            start = previous["tokens"] if previous else 0
            payload = {"tokens": request["target_tokens"], "seed": request["seed"],
                       "model_id": request["model_id"], "recipe_id": request.get("recipe_id")}
            write_json(checkpoint, payload)
            write_json(checkpoint.parent / "previous.pt", {"older_generation": True})
            seed_index = int(request["seed"] == r.TUNING_SEEDS[1])
            recipe = request.get("recipe_id", "r0")
            if request["phase"] == "a1":
                nll = [1., 9.][seed_index] if recipe == "r0" else 5. + int(recipe[1:])
            elif request["phase"] == "a2":
                nll = [1., 9.][seed_index] if recipe == "r0" else 4.
            else:
                nll = 3.
            healthy = not (self.injection.get("unhealthy_a3") and request["phase"] == "a3")
            total = 500 if request["model_id"] == "dense-active" else 1000
            active = 1000 if request["model_id"] == "dense-total" else 500
            resume = {} if previous is None else {
                "path": request["resume_checkpoint"], "sha256": request["resume_checkpoint_sha256"],
                "from_tokens": start, "state_reused": True}
            answer = {"id": tid, "model_id": request["model_id"], "status": "completed",
                      "parameters": total+500,
                      "source_identity": request["source_identity"], "task_sha256": s.sha256(task_path),
                      "training": {"tokens": request["target_tokens"], "from_tokens": start,
                                   "segment_tokens": request["target_tokens"] - start,
                                   "tokens_per_second": 100 * request["batch_size"]},
                      "validation_final": {"nll": nll},
                      "health": {"eligible": healthy, "blocking_reasons": [] if healthy else ["bad-gradient"]},
                      "recovery": {"checkpoint_load": {"status": "exact"}}, "resume": resume,
                      "work": [{"work": {"activity": {"qualified": True, "backbone_total_parameters": total,
                                "mean_active_backbone_parameters": active, "measured_external_tokens": 128}}}],
                      "device_memory": {"npu_peak_allocated_bytes": 64 * request["batch_size"],
                                        "npu_total_memory_bytes": 10**6},
                      "checkpoint": {"path": str(checkpoint), "sha256": s.sha256(checkpoint),
                                     "size_bytes": checkpoint.stat().st_size}}
            result_path = write_json(directory / "result.json", answer)
            self.state["tasks"][tid] = {"status": "completed", "stage": stage,
                "task_path": str(task_path), "result_path": str(result_path), "result_sha256": s.sha256(result_path),
                "attempts": [{"path": str(directory), "status": "completed"}]}
            answers[tid] = answer
            self._save()
        return answers

    def run_dynamic(self, stage, producer):
        """Event simulation: independent workers finish at different times."""
        self.journal.setdefault("dynamic_calls", []).append(stage)
        events = self.journal.setdefault("dynamic_events", [])
        requests, verified, pending, running = {}, {}, [], []
        now, sequence = 0., 0
        try:
            while True:
                update = producer(deepcopy(verified))
                reused = False
                for wave in update["requests"]:
                    for task in wave["tasks"]:
                        tid = task["id"]
                        request = (wave["stage"], task, wave["allowed_failures"])
                        if tid in requests:
                            assert requests[tid] == request
                            continue
                        requests[tid] = request
                        events.append({"kind": "offered", "time": now, "stage": wave["stage"], "task_id": tid})
                        if tid in self.state["tasks"]:
                            verified.update(self.run(wave["stage"], [task], allowed_failures=wave["allowed_failures"]))
                            events.append({"kind": "reused", "time": now, "task_id": tid})
                            reused = True
                        else:
                            pending.append(request)
                if reused:
                    continue
                while pending and len(running) < self.config.get("max_workers", 1):
                    request = pending.pop(0)
                    task = request[1]
                    duration = (100. if task["model_id"] == self.injection.get("slow_a1_model")
                                and task["phase"] == "a1" else 1.)
                    heapq.heappush(running, (now+duration, sequence, request))
                    sequence += 1
                    events.append({"kind": "started", "time": now, "task_id": task["id"]})
                if update["done"]:
                    assert len(verified) == len(requests) and not pending and not running
                    self._save()
                    return verified
                assert running, "producer stalled without pending dependencies"
                now, _, (task_stage, task, allowed) = heapq.heappop(running)
                verified.update(self.run(task_stage, [task], allowed_failures=allowed))
                events.append({"kind": "verified", "time": now, "task_id": task["id"]})
                self._save()
        except BaseException as error:
            self.stop_reason = f"{type(error).__name__}: {error}"
            self.state.update(status="a_blocked_needs_repair", stage_b_gate="closed", blocking_reason=self.stop_reason)
            self._save()
            raise

    def resource_reason(self):
        return None

    def owned_members(self):
        return {}

    def finish(self, status):
        self.state["status"] = status
        self._save()

    def close(self):
        pass


@pytest.fixture
def simulation(tmp_path, monkeypatch):
    SimulatedSupervisor.injection = {}
    monkeypatch.setattr(r, "Supervisor", SimulatedSupervisor)
    groups = ["fc", "lattice", "exchange", "multiscale_settle", "multiscale_timed", "fc"]
    subjects = [{"id": f"subject-{i}", "group": group, "model_key": f"distinct-{i}",
                 "model": {"family": "tide", "config": {"fixture_graph": i,
                           "d": [64, 128, 256, 64, 128, 256][i]}}} for i, group in enumerate(groups)]
    controls = [{"id": name, "group": family, "model_key": name, "effective_key": name,
                 "model": {"family": family, "fixture_control": name}}
                for name, family in (("dense-total", "transformer"), ("dense-active", "transformer"), ("moe-joint", "moe"))]
    plan = {"config": {"max_workers": 1}, "source_identity": {"source": "fixed", "foundation": "fixed", "data": "fixed"},
            "subjects": subjects, "cohort_models": 2,
            "protocol": {"precision": "fp32", "deterministic": True, "sequence_length": 128,
                         "cpu_threads": 1, "npu_total_bytes": 10**6}}

    def matching(campaign):
        campaign.matches = [{"subject_id": model["id"], "control_id": cid, "kind": kind,
                             "qualified": True, "total_relative_error": 0., "active_relative_error": 0.,
                             "ratio_absolute_error": 0.}
                            for model in campaign.subjects
                            for cid, kind in (("dense-total", "dense_total"), ("dense-active", "dense_active"), ("moe-joint", "moe_joint"))]
        return deepcopy(controls)

    def small_audit(*args, **kwargs):
        report = g.audit_stage_a(*args, **kwargs)
        report["blocking_reasons"] = [reason for reason in report["blocking_reasons"]
            if reason not in ("subject_catalog_not_216_unique_effective_models", "fewer_than_200_qualified_subjects")]
        report["status"] = "blocked" if report["blocking_reasons"] else "passed"
        return report

    def small_gate(path, *, manifest_path, report_path, task_paths, gate_kind="formal"):
        report = s.read(report_path)
        assert report["status"] == "passed"
        assert all(s.read(p)["model_id"] in report["eligible_models"] for p in task_paths)
        return r.freeze_json(path, {"status": "passed", "kind": gate_kind, "report_sha256": s.sha256(report_path),
                                    "task_sha256": {s.read(p)["id"]: s.sha256(p) for p in task_paths}})

    monkeypatch.setattr(r.Campaign, "controls", matching)
    monkeypatch.setattr(r.Campaign, "forecast", lambda self: True)
    monkeypatch.setattr(r, "audit_stage_a", small_audit)
    monkeypatch.setattr(r, "create_gate", small_gate)

    def make():
        return r.Campaign(deepcopy(plan), tmp_path / "round")

    return make


def test_joint_selection_uses_complete_seeds_and_real_parent_checkpoints(simulation):
    campaign = simulation()
    model = campaign.subjects[0]
    campaign.train_group([model])
    selected = campaign.selected[model["id"]]
    assert selected["recipe_id"] == "r1", "a lucky r0 seed must not override its worse group mean"
    physical = campaign.supervisor.journal["physical_tasks"]
    learning = [task for task in physical if task["phase"] in {"a1", "a2", "a3"}]
    assert Counter(task["phase"] for task in learning) == {"a1": 12, "a2": 4, "a3": 2}
    assert sum(campaign.completed_result(task)["training"]["segment_tokens"] for task in learning) == 8 * r.MI
    for task in learning:
        if task["phase"] in {"a2", "a3"}:
            expected = r.MI // 4 if task["phase"] == "a2" else r.MI
            assert campaign.completed_result(task)["resume"]["from_tokens"] == expected
            assert task["evaluation_start_tokens"] == expected
    for field in ("selection_a1", "selection_a2", "confirmation"):
        assert len(selected[field]) == 2
        for entry in selected[field]:
            checkpoint = s.read(entry["result_path"])["checkpoint"]
            assert s.sha256(checkpoint["path"]) == checkpoint["sha256"]


def test_repaired_source_a_only_stops_before_any_formal_gate_or_task(simulation):
    from pilotlab.stage_a_revision3 import run_stage_a
    campaign = simulation()
    run_stage_a(campaign, {"deferred_subject_ids": ["old-exposed-subject"]})
    report = s.read(campaign.output / "report.json")
    assert report["status"] == "completed" and report["scope"] == "stage_a_subset_only"
    assert report["stage_b_gate"] == campaign.supervisor.state["stage_b_gate"] == "closed"
    assert report["deferred_subject_ids"] == ["old-exposed-subject"]
    assert not (campaign.output / "stage-b-gate.json").exists()
    assert all(not task["phase"].startswith("b") for task in campaign.supervisor.journal["physical_tasks"])
    before = deepcopy(campaign.supervisor.journal["executions"])
    run_stage_a(simulation(), {"deferred_subject_ids": ["old-exposed-subject"]})
    assert simulation().supervisor.journal["executions"] == before


def test_repaired_source_a_only_keeps_training_failure_blocking(simulation):
    from pilotlab.stage_a_revision3 import run_stage_a
    campaign = simulation()
    SimulatedSupervisor.injection = {"unhealthy_a3": True}
    with pytest.raises(s.Blocked, match="selected_recipe_confirmation_failed"):
        run_stage_a(campaign, {"deferred_subject_ids": []})
    assert campaign.supervisor.state["stage_b_gate"] == "closed"
    assert not (campaign.output / "stage-b-gate.json").exists()


def test_measured_cpu_recipe_keeps_backend_through_continuations_and_formal(simulation):
    campaign = simulation()
    campaign.plan["cpu_recipe_calibration"] = True
    model = campaign.subjects[0]
    campaign.train_group([model])
    physical = campaign.supervisor.journal["physical_tasks"]
    small = [task for task in physical if task["phase"] in {"a1", "a2", "a3"} and task["batch_size"] == 2]
    assert {task["device"] for task in small} == {"cpu"}
    assert {task["phase"] for task in small} == {"a1", "a2", "a3"}
    formal = campaign.formal_task(model, r.FORMAL_SEED, "b", campaign.output / "gate.json")
    assert formal["device"] == "cpu" and not formal.get("resume_checkpoint")
    assert formal["checkpoint_reservation_bytes"] > 0
    assert formal["source_identity"] == campaign.identity


def test_consumed_parent_retirement_keeps_a3_and_is_independently_verifiable(simulation):
    from pilotlab.catalog3 import generate_subjects
    from pilotlab.chain_receipt3 import verify_model_chain_receipt
    campaign = simulation()
    model = generate_subjects()[0]
    campaign.plan["retire_consumed_a_parents"] = True
    campaign.train_group([model])
    choice = campaign.selected[model["id"]]
    evidence = verify_model_chain_receipt(choice["chain_receipt"], model=model, choice=choice,
        source_identity=campaign.identity, tuning_seeds=r.TUNING_SEEDS, diagnostic_seeds=r.DIAGNOSTIC_SEEDS)
    assert len(evidence["retired_checkpoints"]) == 4
    assert all(Path(row["path"]).is_file() for row in evidence["retained_checkpoints"])
    executions = deepcopy(campaign.supervisor.journal["executions"])
    campaign.train_group([model])
    assert campaign.supervisor.journal["executions"] == executions


def test_final_cleanup_never_opens_or_prunes_external_transferred_evidence(simulation):
    campaign = simulation()
    campaign.train_group(campaign.subjects[:1])
    owned = deepcopy(campaign.selected[campaign.subjects[0]["id"]])
    campaign.selected["old-model"] = {"confirmation": [{"task_path": "must-not-open"}]}
    transfer = write_json(campaign.output / "transfer.json", {"model_ids": ["old-model"]})
    campaign.plan["stage_a_transfer"] = {"path": str(transfer), "sha256": s.sha256(transfer)}
    campaign.retire_completed_stage_a()
    assert all(not Path(s.read(e["result_path"])["checkpoint"]["path"]).exists()
        for field in ("selection_a1", "selection_a2", "confirmation") for e in owned[field])


def test_restart_after_a1_reuses_completed_trajectories_without_new_exposure(simulation):
    first = simulation()
    SimulatedSupervisor.injection["crash_before"] = "a2"
    with pytest.raises(RuntimeError, match="injected controller restart"):
        first.train_group(first.subjects[:1])
    recovered = simulation()
    recovered.train_group(recovered.subjects[:1])
    assert set(recovered.supervisor.journal["executions"].values()) == {1}
    assert recovered.selected[recovered.subjects[0]["id"]]["recipe_id"] == "r1"


def test_pipeline_advances_fast_model_and_later_cohort_before_slow_a1_finishes(simulation):
    campaign = simulation()
    campaign.plan.update(pipeline_stage_a=True, pipeline_model_limit=2)
    campaign.plan["config"]["max_workers"] = 64
    fast, slow, later = campaign.subjects[:3]
    SimulatedSupervisor.injection["slow_a1_model"] = slow["id"]
    campaign.train_models([fast, slow, later])
    events = campaign.supervisor.journal["dynamic_events"]

    def indices(model, phase, kind):
        tasks = {task["id"] for task in campaign.supervisor.journal["physical_tasks"]
                 if task["model_id"] == model["id"] and task["phase"] == phase}
        return [index for index, event in enumerate(events)
                if event["kind"] == kind and event["task_id"] in tasks]

    slow_a1_final = max(indices(slow, "a1", "verified"))
    assert max(indices(fast, "a1", "verified")) < min(indices(fast, "a2", "offered"))
    assert max(indices(fast, "a2", "verified")) < min(indices(fast, "a3", "offered"))
    assert max(indices(fast, "a3", "verified")) < slow_a1_final
    assert min(indices(later, "preflight", "started")) < slow_a1_final
    assert set(campaign.selected) == {fast["id"], slow["id"], later["id"]}


def test_pipeline_cpu_baseline_is_offered_before_npu_capacity_completes(simulation):
    campaign = simulation()
    campaign.plan.update(pipeline_stage_a=True, cpu_recipe_calibration=True)
    campaign.profile_models(campaign.subjects[:1])
    events = campaign.supervisor.journal["dynamic_events"]
    offered_cpu = next(i for i, event in enumerate(events)
                       if event["kind"] == "offered" and event["stage"] == "capacity-cpu")
    completed_npu = next(i for i, event in enumerate(events)
                         if event["kind"] == "verified" and event["task_id"].endswith("capacity-small"))
    assert offered_cpu < completed_npu
    assert campaign.resources[campaign.subjects[0]["id"]]["device_by_batch"]["2"] == "cpu"


def test_pipeline_default_window_is_not_limited_to_old_cohort_size(simulation):
    campaign = simulation()
    campaign.plan.update(pipeline_stage_a=True, cohort_models=1)
    campaign.plan["config"].update(max_workers=64, max_cpu_workers=40)
    campaign.profile_models(campaign.subjects)
    events = campaign.supervisor.journal["dynamic_events"]
    first_completion = next(i for i, event in enumerate(events) if event["kind"] == "verified")
    offered = {event["task_id"] for event in events[:first_completion] if event["kind"] == "offered"}
    assert len(offered) == len(campaign.subjects)


@pytest.mark.parametrize("migrate", [False, True])
def test_pipeline_restart_reuses_retired_probes_and_original_cohort_evidence(simulation, migrate):
    first = simulation()
    first.plan["cpu_recipe_calibration"] = True
    SimulatedSupervisor.injection["crash_before"] = "a2"
    with pytest.raises(RuntimeError, match="injected controller restart"):
        first.train_group(first.subjects[:2])
    old_records = deepcopy(first.supervisor.state["tasks"])
    old_tasks = {tid: Path(record["task_path"]).read_bytes() for tid, record in old_records.items()}
    old_decision = first.output/"decisions"/f'init-{first.subjects[0]["id"]}.json'
    old_decision_bytes = old_decision.read_bytes()
    probes = [record for record in old_records.values() if record["stage"].startswith("capacity-") or record["stage"] == "init"]
    assert probes and all(not Path(s.read(row["result_path"])["checkpoint"]["path"]).exists() for row in probes)
    surviving_payloads = {str(path): s.sha256(path) for record in old_records.values()
        for attempt in record["attempts"] for path in (Path(attempt["path"])/"checkpoints").glob("*.pt")}
    output = first.output.with_name("migrated") if migrate else first.output
    if migrate:
        state = deepcopy(first.supervisor.state)
        for record in state["tasks"].values():
            record["imported_completion"] = True
            record["prior_attempts"] = record["attempts"]
            record["attempts"] = []
        write_json(output/"scheduler.json", state)
        write_json(output/"model-resources.json", first.resources)
    recovered = r.Campaign(deepcopy(first.plan), output)
    recovered.plan.update(pipeline_stage_a=True, pipeline_model_limit=2)
    recovered.plan["config"]["max_workers"] = 64
    recovered.train_models(recovered.subjects[:2])
    assert set(recovered.supervisor.journal["executions"].values()) == {1}
    for tid, record in old_records.items():
        assert Path(record["task_path"]).read_bytes() == old_tasks[tid]
        assert recovered.supervisor.state["tasks"][tid]["stage"] == record["stage"]
        if migrate:
            assert tid not in recovered.supervisor.journal["executions"]
    assert old_decision.read_bytes() == old_decision_bytes
    assert (output/"decisions"/f'init-model-{first.subjects[0]["id"]}.json').is_file()
    if migrate:
        assert all(Path(path).is_file() and s.sha256(path) == digest for path, digest in surviving_payloads.items())
    physical = recovered.supervisor.journal["physical_tasks"]
    continued_small = [task for task in physical if task["phase"] in {"a2", "a3"} and task["batch_size"] == 2]
    assert continued_small and all(task["device"] == "cpu" for task in continued_small)


def test_prune_skips_external_attempts_even_when_not_marked_imported(simulation):
    first = simulation()
    first.train_group(first.subjects[:1])
    entry = first.selected[first.subjects[0]["id"]]["confirmation"][0]
    task = s.read(entry["task_path"])
    result = s.read(entry["result_path"])
    target = Path(result["checkpoint"]["path"])
    digest = s.sha256(target)
    output = first.output.with_name("migrated")
    write_json(output/"scheduler.json", first.supervisor.state)
    recovered = r.Campaign(deepcopy(first.plan), output)
    recovered.prune([task], reason="must never consume another round's final")
    assert target.is_file() and s.sha256(target) == digest
    assert recovered.retirement["files"] == []


def test_mixed_imported_chain_is_verified_without_retiring_either_owners_parents(simulation):
    from pilotlab.catalog3 import generate_subjects
    from pilotlab.chain_receipt3 import verify_model_chain_receipt
    first = simulation()
    model = generate_subjects()[0]
    SimulatedSupervisor.injection["crash_before"] = "a2"
    with pytest.raises(RuntimeError, match="injected controller restart"):
        first.train_group([model])
    output = first.output.with_name("migrated")
    write_json(output/"scheduler.json", first.supervisor.state)
    write_json(output/"model-resources.json", first.resources)
    recovered = r.Campaign(deepcopy(first.plan), output)
    recovered.plan.update(pipeline_stage_a=True, retire_consumed_a_parents=True)
    recovered.train_models([model])
    choice = recovered.selected[model["id"]]
    assert "parent_checkpoint_retirement" not in choice
    report = verify_model_chain_receipt(choice["chain_receipt"], model=model, choice=choice,
        source_identity=recovered.identity, tuning_seeds=r.TUNING_SEEDS, diagnostic_seeds=r.DIAGNOSTIC_SEEDS)
    assert report["retired_checkpoints"] == []
    assert all(Path(s.read(entry["result_path"])["checkpoint"]["path"]).is_file()
               for field in ("selection_a1", "selection_a2", "confirmation") for entry in choice[field])


def test_pipeline_failure_closes_gate_and_does_not_admit_later_models(simulation):
    campaign = simulation()
    campaign.plan.update(pipeline_stage_a=True, pipeline_model_limit=2)
    campaign.plan["config"]["max_workers"] = 64
    SimulatedSupervisor.injection.update(unhealthy_a3=True, slow_a1_model=campaign.subjects[1]["id"])
    with pytest.raises(s.Blocked, match="confirmation_failed"):
        campaign.run()
    assert campaign.supervisor.state["stage_b_gate"] == "closed"
    assert campaign.supervisor.state["status"] == "a_blocked_needs_repair"
    assert not (campaign.output/"stage-b-gate.json").exists()
    assert not any(stage.startswith("b-") for stage in campaign.supervisor.journal["calls"])
    assert campaign.subjects[2]["id"] not in {task["model_id"] for task in campaign.supervisor.journal["physical_tasks"]}


def test_pipeline_retains_control_capacity_forecast_and_whole_a_gate(simulation, monkeypatch):
    campaign = simulation()
    campaign.plan.update(pipeline_stage_a=True, pipeline_model_limit=2)
    campaign.plan["config"]["max_workers"] = 64
    SimulatedSupervisor.injection["slow_a1_model"] = campaign.subjects[1]["id"]
    forecasts = []

    def forecast():
        forecasts.append({"models": {model["id"] for model in campaign.models},
                          "resources": set(campaign.resources), "selected": set(campaign.selected)})
        return True

    monkeypatch.setattr(campaign, "forecast", forecast)
    campaign.run()
    subject_ids = {model["id"] for model in campaign.subjects}
    assert forecasts[0]["models"] == forecasts[0]["resources"]
    assert forecasts[0]["selected"] == subject_ids
    assert forecasts[-1]["models"] == forecasts[-1]["selected"]
    calls = campaign.supervisor.journal["calls"]
    assert calls.index("b-formal") > max(i for i, stage in enumerate(calls) if stage == "a3")
    assert s.read(campaign.output/"stage-a-report.json")["status"] == "passed"
    assert campaign.supervisor.state["status"] == "completed"


def test_worker_success_with_failed_a3_health_never_releases_b(simulation):
    campaign = simulation()
    SimulatedSupervisor.injection["unhealthy_a3"] = True
    with pytest.raises(s.Blocked, match="confirmation_failed"):
        campaign.run()
    assert not any(stage.startswith("b-") for stage in campaign.supervisor.journal["calls"])
    assert not (campaign.output / "stage-b-gate.json").exists()


def test_full_small_dag_and_terminal_restart_keep_exposure_and_evaluation_budgets(simulation):
    campaign = simulation()
    campaign.run()
    journal = deepcopy(campaign.supervisor.journal)
    assert campaign.supervisor.state["status"] == "completed"
    calls = journal["calls"]
    assert calls.index("b-formal") > max(i for i, stage in enumerate(calls) if stage == "a3")
    assert calls.index("b-repeat") > calls.index("b-formal")
    assert calls.index("b-confirmation") > calls.index("b-repeat")
    evaluations = defaultdict(int)
    for task in journal["physical_tasks"]:
        if task.get("evaluation_budget_key"):
            evaluations[task["evaluation_budget_key"]] += s.evaluation_bound(task)
            assert evaluations[task["evaluation_budget_key"]] <= task["evaluation_token_limit"]
        if task["phase"] in {"b", "b_repeat"}:
            assert task["target_tokens"] == 12 * r.MI
            assert not task.get("resume_checkpoint")
            assert task["seed"] not in r.DIAGNOSTIC_SEEDS
        if task["phase"] == "b_confirmation":
            assert campaign.completed_result(task)["training"]["segment_tokens"] == 0
            assert task["evaluation_start_tokens"] == 12 * r.MI
    assert not any(Path(s.read(entry["result_path"])["checkpoint"]["path"]).exists()
                   for choice in campaign.selected.values() for entry in choice["selection_a1"])
    recovered = simulation()
    recovered.run()
    assert recovered.supervisor.journal == journal


def test_retirement_preserves_failed_attempt_payload_after_successful_retry(simulation):
    campaign = simulation()
    campaign.train_group(campaign.subjects[:1])
    entry = campaign.selected[campaign.subjects[0]["id"]]["confirmation"][0]
    request = s.read(entry["task_path"])
    record = campaign.supervisor.state["tasks"][request["id"]]
    directory = campaign.output / "tasks" / request["id"] / "failed-earlier"
    payload = write_json(directory / "checkpoints" / "failure.pt", {"decisive_failure": True})
    record["attempts"].insert(0, {"path": str(directory), "status": "failed"})
    campaign.prune([request], keep_final=True, reason="successful final generation cleanup")
    assert payload.is_file()


def inject_capacity_observations(campaign, monkeypatch, *, private_bytes_per_sample,
                                free_bytes=128 * 2**30, slower_expansion=False, fail_batch=None):
    """Only replace synthetic measurements; Campaign.profile still makes decisions."""
    original = campaign.supervisor.run

    def run(stage, requests, **kwargs):
        answers = original(stage, requests, **kwargs)
        for request in requests:
            answer = answers[request["id"]]
            batch = request["batch_size"]
            answer["parameters"] = 1000
            answer["device_memory"] = {"npu_peak_allocated_bytes": batch * 2**29,
                "npu_total_memory_bytes": 128 * 2**30, "npu_free_memory_bytes": free_bytes,
                "npu_reserved_bytes": 2**30}
            answer["checkpoint"]["size_bytes"] = 12 * answer["parameters"] + batch * private_bytes_per_sample
            answer["training"]["tokens_per_second"] = 100 * batch
            if slower_expansion and batch > 2:
                answer["training"]["tokens_per_second"] = 100
            record = campaign.supervisor.state["tasks"][request["id"]]
            if batch == fail_batch:
                assert "capacity_error" in kwargs.get("allowed_failures", ())
                answer.update(status="failed", failure_class="capacity_error")
                answer["health"] = {"eligible": False, "blocking_reasons": ["allocator OOM"]}
                record["status"] = record["attempts"][-1]["status"] = "failed"
            write_json(record["result_path"], answer)
            record["result_sha256"] = s.sha256(record["result_path"])
        campaign.supervisor._save()
        return answers

    monkeypatch.setattr(campaign.supervisor, "run", run)


def test_private_checkpoint_growth_prevents_blind_batch128_admission(simulation, monkeypatch):
    campaign = simulation()
    model = campaign.subjects[0]
    # B2 fits HBM easily, but its retained private state already exceeds the
    # 32 MiB per-checkpoint target. The second real probe must therefore be B1.
    inject_capacity_observations(campaign, monkeypatch, private_bytes_per_sample=24 * r.MI)
    campaign.profile([model])
    tasks = campaign.supervisor.journal["physical_tasks"]
    assert [task["batch_size"] for task in tasks] == [2, 1]
    assert all(task["target_tokens"] == task["batch_size"] * task["sequence_length"] * 4 for task in tasks)
    resource = campaign.resources[model["id"]]
    assert resource["batches"] == [1, 2]
    assert resource["checkpoint_bytes"] == 12000 + 48 * r.MI
    assert resource["max_measured_batch"] == 2
    assert len(campaign.recipes(model)) == 6


def test_device_free_hbm_limits_expansion_even_when_total_hbm_is_large(simulation, monkeypatch):
    campaign = simulation()
    model = campaign.subjects[0]
    # 128 GiB total is irrelevant when only 3 GiB is currently free and this
    # worker owns 1 GiB. Baseline B2 uses 1 GiB, leaving room to test B4.
    inject_capacity_observations(campaign, monkeypatch, private_bytes_per_sample=128 * 1024,
                                 free_bytes=3 * 2**30)
    campaign.profile([model])
    assert [task["batch_size"] for task in campaign.supervisor.journal["physical_tasks"]] == [2, 4]
    assert campaign.resources[model["id"]]["batches"] == [2, 4]


def test_insufficient_actual_free_hbm_blocks_before_second_probe(simulation, monkeypatch):
    campaign = simulation()
    inject_capacity_observations(campaign, monkeypatch, private_bytes_per_sample=128 * 1024, free_bytes=0)
    with pytest.raises(s.Blocked, match="baseline_capacity_exceeds_headroom"):
        campaign.profile(campaign.subjects[:1])
    assert [task["batch_size"] for task in campaign.supervisor.journal["physical_tasks"]] == [2]


def test_slower_successful_expansion_still_enters_joint_recipe_selection(simulation, monkeypatch):
    campaign = simulation()
    model = campaign.subjects[0]
    inject_capacity_observations(campaign, monkeypatch, private_bytes_per_sample=128 * 1024,
                                 slower_expansion=True)
    campaign.profile([model])
    measured = sorted(task["batch_size"] for task in campaign.supervisor.journal["physical_tasks"])
    assert measured == [2, 128]
    assert campaign.resources[model["id"]]["chosen_throughput_batch"] == 2
    assert sorted({recipe["batch_size"] for recipe in campaign.recipes(model)}) == measured


def test_larger_oom_is_retained_capacity_rejection_without_losing_baseline(simulation, monkeypatch):
    campaign = simulation()
    model = campaign.subjects[0]
    inject_capacity_observations(campaign, monkeypatch, private_bytes_per_sample=128 * 1024, fail_batch=128)
    campaign.profile([model])
    resource = campaign.resources[model["id"]]
    failure = s.read(resource["larger_capacity_failure"]["result_path"])
    assert failure["status"] == "failed" and failure["failure_class"] == "capacity_error"
    assert Path(failure["checkpoint"]["path"]).is_file(), "failed evidence must survive successful baseline pruning"
    assert resource["max_measured_batch"] == resource["chosen_throughput_batch"] == 2
    assert resource["checkpoint_bytes"] == 12000 + 256 * 1024


def test_smaller_batch_failure_after_baseline_success_blocks_model(simulation, monkeypatch):
    campaign = simulation()
    model = campaign.subjects[0]
    inject_capacity_observations(campaign, monkeypatch, private_bytes_per_sample=24 * r.MI, fail_batch=1)
    with pytest.raises(s.Blocked, match="lower_batch_capacity_failed_after_baseline_passed"):
        campaign.profile([model])
    assert model["id"] not in campaign.resources


@pytest.fixture
def forecast_campaign(tmp_path, monkeypatch):
    """Use the actual forecast, real retained file sizes and a finite ledger."""
    from pilotlab import catalog3
    monkeypatch.setattr(r, "Supervisor", SimulatedSupervisor)
    monkeypatch.setattr(catalog3, "exact_parameter_counts", lambda spec, vocab: {"whole_model_parameters": 1000})
    monkeypatch.setattr(r.time, "time", lambda: 1000.0)
    SimulatedSupervisor.injection = {}
    budget = tmp_path / "budget.json"
    write_json(budget, {"started_unix": 100., "deadline_unix": 1000000.,
                        "max_device_seconds": 1000000., "device_seconds": 100.})
    models = [{"id": name, "model": {"family": "fixture"}} for name in ("selected", "unselected")]
    plan = {"source_identity": {"fixture": "fixed"}, "subjects": models,
        "config": {"max_workers": 16, "storage_reserve_bytes": 16 * r.MI, "budget_ledger": str(budget)},
        "protocol": {"admitted_parallelism": 8}}
    campaign = r.Campaign(plan, tmp_path / "round")
    campaign.supervisor.state["peak_actual_npu_concurrency"] = 2
    campaign.resources = {model["id"]: {"checkpoint_bytes": 2 * r.MI, "tokens_per_second": r.MI,
        "tokens_per_second_by_batch": {"2": r.MI / 2, "8": r.MI}} for model in models}
    entries = []
    for index, mib in enumerate((1, 2, 3, 4, 5, 40)):
        checkpoint = campaign.output / f"selected-{index}.pt"
        with checkpoint.open("wb") as stream:
            stream.truncate(mib * r.MI)
        result_path = write_json(campaign.output / f"selected-{index}.json",
            {"checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size},
             "training": {"tokens_per_second": r.MI / 4 if index == 5 else r.MI / 2}})
        task_path = write_json(campaign.output / f"selected-task-{index}.json", {"batch_size": 2})
        entries.append({"result_path": str(result_path), "task_path": str(task_path)})
    campaign.selected = {"selected": {field: entries[i:i + 2]
        for field, i in (("selection_a1", 0), ("selection_a2", 2), ("confirmation", 4))}}
    return campaign


def test_forecast_uses_selected_checkpoint_peak_and_only_additional_storage(forecast_campaign):
    campaign = forecast_campaign
    assert campaign.forecast()
    report = s.read(campaign.output / "forecast.json")
    assert report["checkpoint_estimates"]["selected"] == 48 * r.MI
    assert report["already_retained_decisive_checkpoint_bytes"] == 55 * r.MI
    additional = report["estimated_peak_new_output_bytes"] - 55 * r.MI
    campaign.supervisor.resource_cache["available_storage_bytes"] = additional + campaign.plan["config"]["storage_reserve_bytes"]
    assert campaign.forecast(), "existing A files are already charged to quota and must not be charged twice"
    campaign.supervisor.resource_cache["available_storage_bytes"] -= 1
    assert not campaign.forecast()
    assert s.read(campaign.output / "forecast.json")["storage_fits"] is False


def test_forecast_retains_unfinished_parent_retirement_cost_after_restart(forecast_campaign):
    campaign = forecast_campaign
    campaign.plan["retire_consumed_a_parents"] = True
    assert campaign.forecast()
    report = s.read(campaign.output / "forecast.json")
    assert report["retained_checkpoint_generations_by_model"] == {"selected": 7, "unselected": 3}
    parent = campaign.selected["selected"]["selection_a1"][0]
    Path(s.read(parent["result_path"])["checkpoint"]["path"]).unlink()
    assert campaign.forecast()
    partial = s.read(campaign.output / "forecast.json")
    assert partial["retained_checkpoint_generations_by_model"]["selected"] == 6
    assert partial["already_retained_decisive_checkpoint_bytes"] == 54 * r.MI


def test_forecast_uses_remaining_device_hours_instead_of_original_ceiling(forecast_campaign):
    campaign = forecast_campaign
    assert campaign.forecast()
    report = s.read(campaign.output / "forecast.json")
    required = report["estimated_remaining_device_seconds_with_margin"]
    ledger_path = campaign.plan["config"]["budget_ledger"]
    ledger = s.read(ledger_path)
    ledger["device_seconds"] = ledger["max_device_seconds"] - required + 1
    write_json(ledger_path, ledger)
    assert not campaign.forecast()
    blocked = s.read(campaign.output / "forecast.json")
    assert blocked["storage_fits"] and blocked["wall_budget_fits"]
    assert blocked["device_budget_fits"] is False
    assert blocked["remaining_device_seconds"] == pytest.approx(required - 1)


def test_forecast_discards_retired_unselected_batch_storage(forecast_campaign):
    campaign = forecast_campaign
    campaign.resources["selected"].update(checkpoint_bytes=512*r.MI,
        checkpoint_bytes_by_batch={"2": 32*r.MI, "128": 512*r.MI})
    assert campaign.forecast()
    assert s.read(campaign.output/"forecast.json")["checkpoint_estimates"]["selected"] == 48*r.MI


def test_forecast_deadline_uses_observed_concurrency_and_remaining_wall_time(forecast_campaign):
    campaign = forecast_campaign
    assert campaign.forecast()
    report = s.read(campaign.output / "forecast.json")
    assert report["forecast_parallelism"] == 2, "16 worker slots or eight declared devices are not observed concurrency"
    assert report["estimated_remaining_wall_seconds"] == report["estimated_remaining_device_seconds_with_margin"] / 2
    ledger_path = campaign.plan["config"]["budget_ledger"]
    ledger = s.read(ledger_path)
    ledger["deadline_unix"] = 1000 + report["estimated_remaining_wall_seconds"] - 0.5
    write_json(ledger_path, ledger)
    assert not campaign.forecast()
    blocked = s.read(campaign.output / "forecast.json")
    assert blocked["device_budget_fits"] and blocked["storage_fits"]
    assert blocked["wall_budget_fits"] is False


def test_resource_stop_is_saved_before_unpopulated_resource_cache_is_read(forecast_campaign, monkeypatch):
    campaign = forecast_campaign
    campaign.supervisor.resource_cache = {}
    monkeypatch.setattr(campaign.supervisor, "resource_reason", lambda: "global_deadline")
    assert campaign.forecast() is False
    assert s.read(campaign.output / "forecast.json")["resource_blocking_reason"] == "global_deadline"


def test_forecast_without_measured_throughput_fails_closed(forecast_campaign):
    campaign = forecast_campaign
    del campaign.resources["unselected"]["tokens_per_second"]
    del campaign.resources["unselected"]["tokens_per_second_by_batch"]
    with pytest.raises(s.Blocked, match="forecast_requires_measured_capacity:unselected"):
        campaign.forecast()


def test_forecast_uses_slower_selected_seed_and_slowest_unselected_batch(forecast_campaign):
    campaign = forecast_campaign
    assert campaign.forecast()
    first = s.read(campaign.output / "forecast.json")
    assert first["tokens_per_second_by_model"] == {"selected": r.MI / 4, "unselected": r.MI / 2}
    # Inflating the fastest capacity result cannot make the formal recipe or
    # the potentially slower future recipe appear quicker than its evidence.
    for resource in campaign.resources.values():
        resource["tokens_per_second"] = 100 * r.MI
        resource["tokens_per_second_by_batch"]["8"] = 100 * r.MI
    assert campaign.forecast()
    second = s.read(campaign.output / "forecast.json")
    for key in ("estimated_remaining_device_seconds_with_margin", "estimated_repeat_device_seconds_without_margin",
                "estimated_remaining_wall_seconds"):
        assert second[key] == first[key]
