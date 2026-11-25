"""Authorized finite round-three DAG, with an audited whole-stage A/B barrier.

The controller never edits code or invents candidates after a failure. Model
catalogs, adaptive choices and their decisive inputs are durable records.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict, deque
import json
import math
import os
from pathlib import Path
import signal
import statistics
import time

from .io import atomic_json, sha256
from .gate3 import MI, activity_observation, audit_stage_a, choose_recipes, combine_activity, create_gate, eligible
from .scheduler3 import Blocked, Supervisor, freeze_json, read, worker_limit

TUNING_SEEDS = [2026112001, 2026112002]
DIAGNOSTIC_SEEDS = [*TUNING_SEEDS, 2026112003]
FORMAL_SEED = 2026112011
REPEAT_SEEDS = [2026112012, 2026112013]
REPEAT_SLOTS = [("fc", 64), ("lattice", 128), ("exchange", 256),
                ("multiscale_settle", 64), ("multiscale_timed", 128)]


def repeat_representatives(subjects, results):
    by_id = {model["id"]: model for model in subjects}
    ranks = sorted((result["validation_final"]["nll"], result["model_id"])
                   for result in results.values() if result["model_id"] in by_id)
    representatives = []
    for group, width in REPEAT_SLOTS:
        candidates = [mid for _, mid in ranks if by_id[mid]["group"] == group
                      and by_id[mid]["model"]["config"]["d"] == width]
        if not candidates:
            raise Blocked(f"repeat_coverage_slot_unfilled:{group}:d{width}")
        representatives.append(candidates[0])
    remaining = [mid for _, mid in ranks if mid not in representatives]
    if not remaining:
        raise Blocked("sixth_repeat_coverage_slot_unfilled")
    return representatives+[remaining[0]]


def last_activity(result, task=None):
    try:
        activity = activity_observation(result, task)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise Blocked(f"missing_actual_activity_measurement: {exc}") from exc
    return activity


def mean_activity(results, tasks=None):
    if tasks is None:
        tasks = [None] * len(results)
    if len(tasks) != len(results):
        raise Blocked("incomplete A3 activity seed group")
    rows = [last_activity(result, task) for result, task in zip(results, tasks)]
    if not rows or len({row["backbone_total_parameters"] for row in rows}) != 1:
        raise Blocked("inconsistent_activity_parameter_identity")
    counts = combine_activity(rows)
    for key in ("version", "method", "whole_model_parameters", "vocabulary_parameters"):
        if any(row.get(key) != rows[0].get(key) for row in rows):
            raise Blocked("inconsistent_activity_measurement_definition:"+key)
    shared = {k:v for k,v in rows[0].items() if k != "measurement_origin" and all(r.get(k)==v for r in rows)}
    if "vocabulary_parameters" in shared:
        shared["mean_active_whole_model_parameters"] = counts["mean_active_backbone_parameters"] + shared["vocabulary_parameters"]
    for key, reduce_samples in (("min_active_backbone_parameters", min), ("max_active_backbone_parameters", max),
                                ("semantic_full_owner_tick_rows", sum)):
        if all(key in row for row in rows):
            shared[key] = reduce_samples(row[key] for row in rows)
    return {**shared, **counts, "aggregation": "last prescribed sample of each fixed A3 seed"}


class Campaign:
    def __init__(self, plan, output):
        self.plan = plan
        from .hbm_profile3 import frozen_profile
        inherited_hbm = frozen_profile(plan.get("hbm_profile"))
        if plan["config"].get("runq_root"):
            from .runq_campaign import QueueSupervisor
            self.supervisor = QueueSupervisor(plan["config"], output, inherited_hbm=inherited_hbm)
        elif plan["config"].get("shared_execution"):
            from .scheduler_shared3 import SharedSupervisor
            self.supervisor = SharedSupervisor(plan["config"], output)
        else:
            self.supervisor = Supervisor(plan["config"], output)
        self.output = self.supervisor.output
        self.inherited_hbm = inherited_hbm
        self.identity = plan["source_identity"]
        self.protocol = plan["protocol"]
        self.subjects = plan["subjects"]
        self.models = list(self.subjects)
        self.selected = read(self.output/"selected.json", {})
        self.resources = read(self.output/"model-resources.json", {})
        self.matches = []
        self.candidate_tasks = {}
        self.retirement = read(self.output/"checkpoint-retirement.json", {"schema": "pilot-r3-retirement-v1", "files": []})

    def task(self, model, phase, seed, *, recipe=None, target=0, suffix=""):
        value = dict(id=f'{model["id"]}-{phase}-{seed}{suffix}', model_id=model["id"],
            phase=phase, model=model["model"], seed=seed, source_identity=self.identity,
            target_tokens=target, schedule_tokens=12*MI, precision=self.protocol["precision"],
            deterministic=self.protocol["deterministic"], sequence_length=self.protocol["sequence_length"],
            stream_chunks=4, cpu_threads=self.protocol["cpu_threads"], batch_size=4,
            init_recipe="default", lr=3e-4, weight_decay=0.01, grad_clip=1.0,
            warmup_tokens=65536, lr_floor_ratio=0.1, auxiliary_loss_weight=0.01,
            eval_tokens=16384, eval_interval_tokens=MI, health_interval_tokens=MI//4,
            checkpoint_interval_seconds=180, closing_reserve_seconds=180,
            device="npu:0", record_step_fingerprints=False,
            evaluation_budget_key=model["id"], evaluation_token_limit=MI,
            evaluation_start_tokens=0)
        if recipe:
            value.update(recipe)
        self.reserve_resources(value)
        if phase == "preflight":
            value.pop("evaluation_budget_key")
        return value

    def reserve_resources(self, value):
        resource = self.resources.get(value["model_id"], {})
        batch = str(value["batch_size"])
        if resource:
            peak = resource.get("npu_peak_bytes_by_batch", {}).get(batch, 0)
            rss = resource.get("rss_peak_bytes_by_batch", {}).get(batch, 0)
            checkpoint = resource.get("checkpoint_bytes_by_batch", {}).get(batch, resource.get("checkpoint_bytes", 0))
            if peak:
                value["reserved_hbm_bytes"] = max(1024**3, math.ceil(1.5*peak) + 512*2**20)
            if rss:
                value["reserved_rss_bytes"] = max(2*1024**3, math.ceil(1.5*rss) + 256*2**20)
            if checkpoint:
                value["checkpoint_reservation_bytes"] = max(64*2**20, math.ceil(3.6*checkpoint))

    def evidence(self, task):
        state = self.supervisor.state["tasks"][task["id"]]
        return {"task_path": state["task_path"], "result_path": state["result_path"]}

    def completed_result(self, task):
        state = self.supervisor.state["tasks"][task["id"]]
        return read(state["result_path"])

    def prune(self, tasks, *, keep_final=False, reason):
        """Only successful newly generated checkpoint files in this round."""
        for task in tasks:
            record = self.supervisor.state["tasks"][task["id"]]
            if record["status"] != "completed":
                continue
            attempts = [attempt for attempt in record["attempts"]
                        if attempt.get("status") == "completed"
                        and Path(attempt["path"]).resolve().is_relative_to(self.output)]
            if not attempts:
                continue  # Migrated completed evidence retains its old owner.
            result = read(record["result_path"])
            if sha256(record["result_path"]) != record["result_sha256"]:
                raise Blocked("cannot_retire_changed_success_record")
            final = Path(result["checkpoint"]["path"]).resolve()
            for attempt in attempts:
                directory = Path(attempt["path"])/"checkpoints"
                if not directory.resolve().is_relative_to(self.output):
                    raise Blocked("checkpoint_retirement_outside_new_round")
                for file in directory.glob("*.pt"):
                    if keep_final and file.resolve() == final:
                        continue
                    item = {"path": str(file.resolve()), "sha256": sha256(file),
                            "bytes": file.stat().st_size, "task_id": task["id"],
                            "reason": reason, "retired_unix": time.time()}
                    self.retirement["files"].append(item)
                    atomic_json(self.output/"checkpoint-retirement.json", self.retirement)
                    file.unlink()

    @staticmethod
    def request(stage, tasks, *, allowed_failures=()):
        return {"stage": stage, "tasks": tasks, "allowed_failures": list(allowed_failures)}

    def drain_flow(self, flow):
        """Keep the ordinary blocking entry points for explicit small batches."""
        answers = None
        while True:
            try:
                requests = flow.send(answers)
            except StopIteration:
                return
            answers = {}
            for request in requests:
                answers.update(self.supervisor.run(request["stage"], request["tasks"],
                    allowed_failures=request["allowed_failures"]))

    def stage_a_pipelined(self):
        return self.plan.get("pipeline_stage_a", self.plan["config"].get("shared_execution", False))

    def run_model_flows(self, models, factory, *, stage):
        """Advance independent models using one finite resource supervisor.

        Only a model's own complete seed wave is a dependency. The global
        audit and control-capacity forecast remain outside this pipeline.
        """
        models = list(models)
        if len({model["id"] for model in models}) != len(models):
            raise Blocked("duplicate_pipeline_model_identity")
        config = self.plan["config"]
        default_limit = max(self.plan.get("cohort_models", 4), worker_limit(config),
                            2*config.get("max_cpu_workers", 32))
        limit = self.plan.get("pipeline_model_limit", default_limit)
        if type(limit) is not int or limit < 1:
            raise ValueError("pipeline_model_limit must be a positive integer")
        rejoining = self.supervisor.inflight_model_ids() if config.get("runq_root") else set()
        # Keep plan order within each group. Already dispatched work can span
        # more models than the window now admits.
        pending = deque(sorted(models, key=lambda model: model["id"] not in rejoining))
        active = {}

        blocked_models = {}

        def advance(mid, flow, answers, requests):
            while True:
                try:
                    wave = flow.send(answers)
                except Blocked as error:
                    if not config.get("runq_root"):
                        raise
                    blocked_models[mid] = str(error)
                    return None
                except StopIteration:
                    return None
                task_ids = {task["id"] for request in wave for task in request["tasks"]}
                if task_ids:
                    requests.extend(wave)
                    return task_ids
                answers = {}

        def produce(verified):
            dispatch = getattr(self.supervisor, "dispatch_placement", None)
            requests = []
            # Continue ready chains before admitting replacement models.
            # A scheduler result is visible here only after owner cleanup.
            for mid, (flow, waiting) in list(active.items()):
                if waiting.intersection(getattr(self.supervisor, "failed_independent_tasks", ())):
                    flow.close()
                    del active[mid]
                    continue
                if not waiting.issubset(verified):
                    continue
                next_wave = advance(mid, flow, {tid: verified[tid] for tid in waiting}, requests)
                if next_wave is None:
                    del active[mid]
                else:
                    active[mid] = flow, next_wave
            while pending and (pending[0]["id"] in rejoining or len(active) < limit):
                model = pending.popleft()
                flow = factory([model])
                waiting = advance(model["id"], flow, None, requests)
                if waiting is not None:
                    active[model["id"]] = flow, waiting
            if dispatch and stage == "stage-a-training-pipeline":
                requests[:0] = dispatch.supplemental_requests(verified)
            return {"requests": requests, "done": not pending and not active}

        try:
            self.supervisor.run_dynamic(stage, produce)
            if blocked_models:
                raise Blocked(f"model dependencies blocked after queue drain: {blocked_models}")
        finally:
            for flow, _ in active.values():
                flow.close()

    def profile(self, models):
        self.drain_flow(self.profile_flow(models))

    def profile_models(self, models):
        if self.stage_a_pipelined():
            self.run_model_flows(models, self.profile_flow, stage="stage-a-capacity-pipeline")
            return
        size = self.plan.get("cohort_models", 4)
        for start in range(0, len(models), size):
            self.profile(models[start:start+size])

    def profile_flow(self, models):
        """Two finite per-model batch measurements, within preparation budget."""
        pending = [model for model in models if model["id"] not in self.resources]
        if not pending:
            return
        length = self.protocol["sequence_length"]
        small = 2
        first = [self.task(model, "preflight", 2026112021, recipe=dict(batch_size=small,
                    eval_tokens=256, eval_interval_tokens=0, purpose="capacity_calibration"),
                    target=small*length*(16 if self.plan.get("cpu_recipe_calibration") else 4), suffix="-capacity-small") for model in pending]
        cpu_tasks = []
        if self.plan.get("cpu_recipe_calibration"):
            cpu_tasks = [self.task(model, "preflight", 2026112021, recipe=dict(batch_size=small,
                    device="cpu", eval_tokens=256, eval_interval_tokens=0,
                    purpose="capacity_calibration", closing_reserve_seconds=60),
                    target=small*length*16, suffix="-capacity-cpu") for model in pending]
        # These independent fixed probes share no state. Starting the CPU
        # measurement now avoids holding host slots behind the NPU expansion.
        requests = [self.request("capacity-small", first)]
        if cpu_tasks:
            requests.append(self.request("capacity-cpu", cpu_tasks))
        observed = yield requests
        results = {task["id"]: observed[task["id"]] for task in first}
        cpu_results = {task["id"]: observed[task["id"]] for task in cpu_tasks}
        for task in first:
            if eligible(task, results[task["id"]]):
                raise Blocked(f'capacity_baseline_not_healthy:{task["model_id"]}')
        for task in cpu_tasks:
            if eligible(task, cpu_results[task["id"]]):
                raise Blocked(f'cpu_capacity_probe_unhealthy:{task["model_id"]}')
        second = []
        for model, task in zip(pending, first):
            result = results[task["id"]]
            peak = result["device_memory"]["npu_peak_allocated_bytes"]
            total = result["device_memory"].get("npu_total_memory_bytes", self.protocol["npu_total_bytes"])
            available = min(total, result["device_memory"].get("npu_free_memory_bytes", total)
                            +result["device_memory"].get("npu_reserved_bytes", 0))
            if not peak or peak > .8*available:
                raise Blocked(f'baseline_capacity_exceeds_headroom:{model["id"]}')
            ratio = max(1, int(.60*available/peak))
            # Retaining six decisive A checkpoints makes private-state I/O
            # a real capacity limit. Avoid choosing a large NPU batch whose
            # cache copies would exhaust the account quota. The first probe
            # covers four full stream chunks; this estimate is remeasured by
            # the second probe and still subject to the whole-round forecast.
            parameter_bytes = 12*result["parameters"]
            private_bytes = max(1, result["checkpoint"]["size_bytes"]-parameter_bytes)
            private_limit = (self.plan.get("profile_private_checkpoint_target_bytes", 32*2**20)
                             + self.plan.get("profile_private_parameter_factor", 2)*parameter_bytes)
            ratio = min(ratio, max(1, int(private_limit/private_bytes)))
            candidate = min(128 if length == 128 else 64, small*2**int(math.log2(ratio)))
            if candidate == small:
                candidate = 1
            if candidate != small:
                second.append(self.task(model, "preflight", 2026112021, recipe=dict(batch_size=candidate,
                    eval_tokens=256, eval_interval_tokens=0, purpose="capacity_calibration", capture_failure=False),
                    target=candidate*length*4, suffix="-capacity-large"))
        larger = (yield [self.request("capacity-large", second,
                                    allowed_failures={"capacity_error"})]) if second else {}
        cpu_by_model = {task["model_id"]: task for task in cpu_tasks}
        second_by_model = {task["model_id"]: task for task in second}
        for model, first_task in zip(pending, first):
            rows = [(first_task, results[first_task["id"]])]
            alternative = second_by_model.get(model["id"])
            if alternative:
                candidate_result = larger[alternative["id"]]
                if alternative["batch_size"] < small and candidate_result["status"] != "completed":
                    raise Blocked(f'lower_batch_capacity_failed_after_baseline_passed:{model["id"]}')
                if candidate_result["status"] == "completed":
                    if eligible(alternative, candidate_result):
                        raise Blocked(f'large_capacity_probe_unhealthy:{model["id"]}')
                    rows.append((alternative, candidate_result))
            best_task, best_result = max(rows, key=lambda pair: pair[1]["training"]["tokens_per_second"])
            batches = sorted({row[0]["batch_size"] for row in rows})
            if len(batches) == 1:
                batches = [max(1, small//2), small]
            devices = {str(batch): "npu:0" for batch in batches}
            cpu_task = cpu_by_model.get(model["id"])
            cpu_result = cpu_results[cpu_task["id"]] if cpu_task else None
            if cpu_result and cpu_result["training"]["tokens_per_second"] >= (
                    self.plan.get("cpu_min_exclusive_rate_ratio", 1.0) * results[first_task["id"]]["training"]["tokens_per_second"]):
                devices[str(small)] = "cpu"
            self.resources[model["id"]] = {"batches": batches, "chosen_throughput_batch": best_task["batch_size"],
                "device_by_batch": devices,
                "tokens_per_second": best_result["training"]["tokens_per_second"],
                "tokens_per_second_by_batch": {str(row[0]["batch_size"]): row[1]["training"]["tokens_per_second"]
                                                for row in rows},
                "checkpoint_bytes": max(row[1]["checkpoint"]["size_bytes"] for row in rows),
                "checkpoint_bytes_by_batch": {str(row[0]["batch_size"]): row[1]["checkpoint"]["size_bytes"]
                                               for row in rows},
                "npu_peak_bytes_by_batch": {str(row[0]["batch_size"]): max(row[1]["device_memory"].get("npu_peak_allocated_bytes", 0),
                    row[1]["device_memory"].get("npu_peak_reserved_bytes", 0)) for row in rows},
                "rss_peak_bytes_by_batch": {str(row[0]["batch_size"]): row[1]["device_memory"].get("cpu_rss_peak_bytes", 0) for row in rows},
                "max_measured_batch": max(row[0]["batch_size"] for row in rows),
                "private_checkpoint_target_bytes": (self.plan.get("profile_private_checkpoint_target_bytes", 32*2**20)
                    + self.plan.get("profile_private_parameter_factor", 2)*12*rows[0][1]["parameters"]),
                "measurements": [self.evidence(row[0]) for row in rows],
                "cpu_measurement": self.evidence(cpu_task) if cpu_task else None,
                "cpu_tokens_per_second": cpu_result["training"]["tokens_per_second"] if cpu_result else None,
                "backend_selection": "CPU small-batch only when qualified pre-training measurement reaches declared NPU-rate ratio; each trajectory and selected formal recipe keep their backend",
                "larger_capacity_failure": self.evidence(alternative) if alternative and len(rows) == 1 else None,
                "selection": "two batches bounded by actual HBM and projected private checkpoint storage; highest measured training throughput plus lower batch tuned; whole-round storage forecast remains mandatory"}
            if cpu_result:
                resource = self.resources[model["id"]]
                resource["rss_peak_bytes_by_batch"][str(small)] = max(
                    resource["rss_peak_bytes_by_batch"].get(str(small), 0),
                    cpu_result["device_memory"].get("cpu_rss_peak_bytes", 0))
                resource["checkpoint_bytes_by_batch"][str(small)] = max(
                    resource["checkpoint_bytes_by_batch"].get(str(small), 0),
                    cpu_result["checkpoint"]["size_bytes"])
        atomic_json(self.output/"model-resources.json", self.resources)
        self.prune(first+second+cpu_tasks, reason="completed capacity measurements; fixed seed reproducible, no training dependency")

    def recipes(self, model):
        batches = self.resources[model["id"]]["batches"]
        return [dict(recipe_id=f"r{index}", batch_size=batch, lr=lr, init_recipe="default",
                     device=self.resources[model["id"]].get("device_by_batch", {}).get(str(batch), "npu:0"))
                for index, (batch, lr) in enumerate((b, lr) for b in batches for lr in (1e-4, 3e-4, 1e-3))]

    def train_group(self, models):
        self.drain_flow(self.training_flow(models))

    def training_flow(self, models, *, per_model_decisions=False):
        models = [model for model in models if model["id"] not in self.selected]
        if not models:
            return
        yield from self.profile_flow(models)
        diagnostics = [self.task(model, "init", seed, recipe=dict(
            batch_size=max(self.resources[model["id"]]["batches"]), eval_tokens=256,
            eval_interval_tokens=0, health_interval_tokens=8192), target=32768)
            for model in models for seed in DIAGNOSTIC_SEEDS]
        initialized = yield [self.request("init", diagnostics)]
        for task in diagnostics:
            if eligible(task, initialized[task["id"]]):
                raise Blocked(f'initialization_not_qualified:{task["id"]}')
        # Old blocking cohorts used one decision named after their first
        # model. Keep that immutable evidence intact when importing a cohort
        # into the per-model pipeline; tasks and decisive results are shared.
        prefix = "init-model" if per_model_decisions else "init"
        freeze_json(self.output/"decisions"/f'{prefix}-{models[0]["id"]}.json',
                    {"diagnostics": [self.evidence(task) for task in diagnostics], "status": "passed"})
        self.prune(diagnostics, reason="successful initialization diagnostics recorded; no checkpoint consumer")
        a1 = [self.task(model, "a1", seed, recipe=recipe, target=MI//4, suffix="-"+recipe["recipe_id"])
              for model in models for recipe in self.recipes(model) for seed in TUNING_SEEDS]
        first = yield [self.request("a1", a1)]
        kept1, a2 = [], []
        for model in models:
            trials = [task for task in a1 if task["model_id"] == model["id"]]
            selection = choose_recipes(trials, first, TUNING_SEEDS, keep=2)
            freeze_json(self.output/"decisions"/(model["id"]+"-a1.json"), selection)
            if len(selection["selected"]) != 2:
                raise Blocked(f'no_two_qualified_recipe_groups:{model["id"]}')
            ids = {row["recipe_id"] for row in selection["selected"]}
            for parent in trials:
                if parent["recipe_id"] not in ids:
                    continue
                kept1.append(parent)
                cp = first[parent["id"]]["checkpoint"]
                child = {**parent, "id": parent["id"].replace("-a1-", "-a2-"), "phase": "a2", "target_tokens": MI,
                         "evaluation_start_tokens": MI//4,
                         "resume_checkpoint": cp["path"], "resume_checkpoint_sha256": cp["sha256"]}
                a2.append(child)
        self.prune([task for task in a1 if task not in kept1], reason="A1 complete seed groups not selected for A2")
        self.prune(kept1, keep_final=True, reason="successful A1 complete; retain its decisive final checkpoint")
        second = yield [self.request("a2", a2)]
        kept2, a3, choices = [], [], {}
        for model in models:
            trials = [task for task in a2 if task["model_id"] == model["id"]]
            selection = choose_recipes(trials, second, TUNING_SEEDS, keep=1)
            freeze_json(self.output/"decisions"/(model["id"]+"-a2.json"), selection)
            if not selection["selected"]:
                raise Blocked(f'no_qualified_recipe:{model["id"]}')
            recipe_id = selection["selected"][0]["recipe_id"]
            choices[model["id"]] = recipe_id
            for parent in trials:
                if parent["recipe_id"] != recipe_id:
                    continue
                kept2.append(parent)
                cp = second[parent["id"]]["checkpoint"]
                a3.append({**parent, "id": parent["id"].replace("-a2-", "-a3-"), "phase": "a3", "target_tokens": 2*MI,
                           "evaluation_start_tokens": MI,
                           "resume_checkpoint": cp["path"], "resume_checkpoint_sha256": cp["sha256"]})
        discarded1 = [task for task in kept1 if task["recipe_id"] != choices[task["model_id"]]]
        self.prune(discarded1+[task for task in a2 if task not in kept2], reason="complete A2 comparison selected a different recipe")
        self.prune(kept2, keep_final=True, reason="successful A2 complete; retain its decisive final checkpoint")
        third = yield [self.request("a3", a3)]
        for task in a3:
            if eligible(task, third[task["id"]], require_resume=True):
                raise Blocked(f'selected_recipe_confirmation_failed:{task["id"]}')
        self.prune(a3, keep_final=True, reason="successful A3 complete; keep complete final checkpoint until B")
        for model in models:
            mid, recipe_id = model["id"], choices[model["id"]]
            selected_a3 = [task for task in a3 if task["model_id"] == mid]
            self.selected[mid] = {"recipe_id": recipe_id, "init_recipe": "default",
                "diagnostics": [self.evidence(t) for t in diagnostics if t["model_id"] == mid],
                "selection_a1": [self.evidence(t) for t in kept1 if t["model_id"] == mid and t["recipe_id"] == recipe_id],
                "selection_a2": [self.evidence(t) for t in kept2 if t["model_id"] == mid],
                "confirmation": [self.evidence(t) for t in selected_a3],
                "activity": mean_activity([third[t["id"]] for t in selected_a3], selected_a3)}
            atomic_json(self.output/"selected.json", self.selected)
            if self.plan.get("retire_consumed_a_parents"):
                self.retire_selected_parents(model)

    def retire_selected_parents(self, model, *, source_identity=None):
        """Seal independent chain verification before retiring consumed A1/A2."""
        from .chain_receipt3 import write_model_chain_receipt, retire_consumed_parents
        mid = model["id"]
        choice = self.selected[mid]
        reference = write_model_chain_receipt(self.output/"decisions"/(mid+"-chain-receipt.json"),
            model=model, choice=choice, source_identity=source_identity or self.identity,
            tuning_seeds=TUNING_SEEDS, diagnostic_seeds=DIAGNOSTIC_SEEDS)
        choice["chain_receipt"] = reference
        atomic_json(self.output/"selected.json", self.selected)
        receipt = read(reference["path"])
        if (any(not Path(row[field]).resolve().is_relative_to(self.output)
                for row in receipt["evidence"] for field in ("task_path", "result_path"))
                or any(not Path(row["path"]).resolve().is_relative_to(self.output)
                       for row in receipt["retirable_checkpoints"])):
            # A mixed imported/new chain remains verifiable, but its old
            # owner's evidence cannot authorize retirement in this round.
            # The forecast charges every surviving parent generation.
            return
        checkpoints = retire_consumed_parents(reference, choice, self.output)
        retirement_path = freeze_json(self.output/"decisions"/(mid+"-parent-retirement.json"),
            {"schema": "pilot-r3-parent-checkpoint-retirement-v1", "receipt": reference,
             "output_root": str(self.output), "checkpoints": checkpoints})
        choice["parent_checkpoint_retirement"] = {"path": str(retirement_path), "sha256": sha256(retirement_path)}
        atomic_json(self.output/"selected.json", self.selected)
        parents = {read(e["result_path"])["checkpoint"]["path"]: read(e["task_path"])["id"]
                   for key in ("selection_a1", "selection_a2") for e in choice[key]}
        for checkpoint in checkpoints:
            path = Path(checkpoint["path"])
            if sha256(path) != checkpoint["sha256"] or path.stat().st_size != checkpoint["size_bytes"]:
                raise Blocked("consumed_parent_changed_before_retirement")
            self.retirement["files"].append({"path": str(path), "sha256": checkpoint["sha256"],
                "bytes": checkpoint["size_bytes"], "task_id": parents[str(path)],
                "reason": "independently verified A1/A2 chain consumed by exact A2/A3 loads; immutable receipt retained",
                "receipt": reference, "retired_unix": time.time()})
            atomic_json(self.output/"checkpoint-retirement.json", self.retirement)
            path.unlink()

    def train_models(self, models):
        if self.stage_a_pipelined():
            self.run_model_flows(models, lambda group: self.training_flow(group, per_model_decisions=True),
                                 stage="stage-a-training-pipeline")
            return
        for start in range(0, len(models), self.plan.get("cohort_models", 4)):
            self.train_group(models[start:start+self.plan.get("cohort_models", 4)])

    def controls(self):
        from .catalog3 import match_control_catalog
        activities = {subject["id"]: self.selected[subject["id"]]["activity"] for subject in self.subjects}
        catalog = match_control_catalog(self.subjects, activities, window=self.protocol["sequence_length"])
        freeze_json(self.output/"control-matching.json", catalog)
        for subject_id, match in catalog["per_subject"].items():
            freeze_json(self.output/"matches"/(subject_id+".json"), match)
        if not catalog["qualified"]:
            raise Blocked(f'no_legal_matched_control_catalog:{catalog["reasons"][:3]}')
        controls = catalog["controls"]
        if len(controls) > self.plan.get("catalog_contract", {}).get("control_count_limit", 648) or len(catalog["matches"]) != 3*len(self.subjects):
            raise Blocked("control_count_or_coverage_outside_approved_plan")
        self.matches = catalog["matches"]
        freeze_json(self.output/"control-catalog.json", {"models": controls, "matches": self.matches,
            "matching_report": str(self.output/"control-matching.json"),
            "matching_report_sha256": sha256(self.output/"control-matching.json")})
        return controls

    def manifest(self):
        return {"schema": "pilot-r3-resolved-model-manifest-v1", "source_identity": self.identity,
                **({"fresh_shannon": self.plan["fresh_shannon"]} if "fresh_shannon" in self.plan else {}),
                **({"catalog_contract": self.plan["catalog_contract"]} if "catalog_contract" in self.plan else {}),
                **({"stage_a_transfer": self.plan["stage_a_transfer"]} if "stage_a_transfer" in self.plan else {}),
                "subjects": self.subjects, "models": self.models, "tuning_seeds": TUNING_SEEDS,
                "diagnostic_seeds": DIAGNOSTIC_SEEDS, "formal_seed": FORMAL_SEED, "repeat_seeds": REPEAT_SEEDS,
                "formal_tokens": 12*MI, "maximum_tuning_tokens_per_model": 8*MI,
                "repeat_coverage_slots": REPEAT_SLOTS,
                "repeat_selection_rule": "minimum formal selection NLL per fixed topology/width slot; sixth best remaining; model id tie break",
                "protocol": self.protocol, "model_count": len(self.models)}

    def forecast(self):
        from .catalog3 import exact_parameter_counts
        estimates = {}
        for model in self.models:
            count = exact_parameter_counts(model["model"], 8192)["whole_model_parameters"]
            measured = self.resources.get(model["id"], {}).get("checkpoint_bytes", 0)
            if model["id"] in self.selected:
                # Discarded larger-batch probes do not remain a storage
                # dependency after the chosen batch is fixed.
                selected_task = read(self.selected[model["id"]]["confirmation"][0]["task_path"])
                measured = self.resources.get(model["id"], {}).get("checkpoint_bytes_by_batch", {}).get(
                    str(selected_task["batch_size"]), 0)
                measured = max(measured, *(read(entry["result_path"])["checkpoint"]["size_bytes"]
                    for field in ("selection_a1", "selection_a2", "confirmation")
                    for entry in self.selected[model["id"]][field]))
            # FP32 parameters and Adam's two full moments need 12 bytes per
            # parameter. Actual populated private state/serialization is
            # measured at the selected largest batch; add 20% headroom.
            # The peak for overlapping generations is estimated below.
            estimates[model["id"]] = max(count*12+16*2**20, math.ceil(measured*1.20))
        # Keep A1/A2/A3 two-seed decisive checkpoints through the gate, and
        # estimate formal/repeat storage plus concurrent atomic generations.
        config = self.plan["config"]
        active_peak = max(estimates.values())*3*worker_limit(config)
        write_limit = config.get("max_inflight_checkpoint_bytes")
        if config.get("shared_execution") and write_limit is not None:
            active_peak = min(active_peak, write_limit)
        repeated = sum(sorted(estimates.values(), reverse=True)[:24])*2
        transferred = set(read(self.plan["stage_a_transfer"]["path"]).get("model_ids", [])) if self.plan.get("stage_a_transfer") else set()
        retained_generations = {mid: 3 if self.plan.get("retire_consumed_a_parents") and mid not in transferred else 7
                                for mid in estimates}
        # A restart can observe the selection before all authorized parent
        # retirements finished. Those surviving files still consume space and
        # must not become a fictitious credit when subtracting existing bytes.
        for mid, choice in self.selected.items():
            if self.plan.get("retire_consumed_a_parents") and mid not in transferred:
                retained_generations[mid] += sum(Path(read(e["result_path"])["checkpoint"]["path"]).exists()
                    for field in ("selection_a1", "selection_a2") for e in choice[field])
        required = sum(size*retained_generations[mid] for mid, size in estimates.items())+repeated+active_peak
        resource_reason = self.supervisor.resource_reason()
        if resource_reason:
            self.supervisor.stop_reason = resource_reason
            atomic_json(self.output/"forecast.json", {"resource_blocking_reason": resource_reason,
                "storage_fits": False, "device_budget_fits": False, "wall_budget_fits": False})
            return False
        reserve = config.get("storage_reserve_bytes", 0)
        available = self.supervisor.resource_cache["available_storage_bytes"] - reserve
        # Existing successful A checkpoints are already in quota usage.
        existing = 0
        for choice in self.selected.values():
            for field in ("selection_a1", "selection_a2", "confirmation"):
                for evidence in choice[field]:
                    result = read(evidence["result_path"])
                    checkpoint = Path(result["checkpoint"]["path"])
                    if checkpoint.exists():
                        existing += checkpoint.stat().st_size
        remaining_estimates, forecast_rates, forecast_backends = {}, {}, {}
        for model in self.models:
            resource = self.resources.get(model["id"], {})
            if model["id"] in self.selected:
                rates = [read(entry["result_path"])["training"]["tokens_per_second"]
                         for entry in self.selected[model["id"]]["confirmation"]]
                backend = read(self.selected[model["id"]]["confirmation"][0]["task_path"]).get("device", "npu:0")
            else:
                rates = list(resource.get("tokens_per_second_by_batch", {}).values())
                backend = "npu:0"
            rate = min(rates) if rates else resource.get("tokens_per_second", 0)
            if not rate or rate <= 0:
                raise Blocked(f'forecast_requires_measured_capacity:{model["id"]}')
            forecast_rates[model["id"]] = rate
            forecast_backends[model["id"]] = backend
            # Formal learning, remaining A rungs, initialization and evaluation.
            tokens = (12 if model["id"] in self.selected else 20.25)*MI+MI
            remaining_estimates[model["id"]] = tokens/rate
        repeat_seconds = sum(sorted((12.25*MI/forecast_rates[m["id"]]
                                     for m in self.models), reverse=True)[:24])*2
        npu_seconds = sum(seconds for mid, seconds in remaining_estimates.items() if forecast_backends[mid].startswith("npu"))
        cpu_seconds = sum(seconds for mid, seconds in remaining_estimates.items() if forecast_backends[mid] == "cpu")
        # Treat every remaining NPU worker as exclusive for a conservative
        # card-hour bound. Shared execution is debited once per actual lease.
        # Keep the entire possible repeat allowance on the NPU side as well.
        device_seconds = (npu_seconds+repeat_seconds)*1.5
        ledger = read(self.plan["config"]["budget_ledger"])
        remaining = ledger["max_device_seconds"]-ledger.get("device_seconds", 0)
        elapsed = time.time()-ledger["started_unix"]
        remaining_wall = ledger["deadline_unix"]-time.time()
        parallelism = max(1, min(worker_limit(self.plan["config"], 1),
            self.supervisor.state.get("peak_actual_npu_concurrency", 0)
            or self.protocol.get("admitted_parallelism", 1)))
        # Queue contention can make this longer; runtime still enforces the
        # absolute deadline, and the gate never treats this as a guarantee.
        cpu_parallelism = max(1, min(self.plan["config"].get("max_cpu_workers", 1),
            self.supervisor.state.get("peak_cpu_worker_concurrency", 0) or 1))
        wall_estimate = max(device_seconds/parallelism, 1.5*cpu_seconds/cpu_parallelism)
        forecast = {"checkpoint_estimates": estimates, "estimated_peak_new_output_bytes": required,
                    "tokens_per_second_by_model": forecast_rates,
                    "throughput_basis": "slower prescribed A3 seed of the selected batch; before selection, slowest successful measured batch; 50% time margin",
                    "already_retained_decisive_checkpoint_bytes": existing,
                    "available_additional_bytes_after_reserve": available,
                    "estimated_remaining_device_seconds_with_margin": device_seconds,
                    "estimated_cpu_worker_seconds_with_margin": cpu_seconds*1.5,
                    "forecast_backend_by_model": forecast_backends,
                    "forecast_cpu_parallelism": cpu_parallelism,
                    "retained_checkpoint_generations_by_model": retained_generations,
                    "estimated_repeat_device_seconds_without_margin": repeat_seconds,
                    "remaining_device_seconds": remaining, "elapsed_wall_seconds": elapsed,
                    "estimated_remaining_wall_seconds": wall_estimate,
                    "remaining_wall_seconds": remaining_wall, "forecast_parallelism": parallelism,
                    "resource_blocking_reason": resource_reason,
                    "storage_fits": required-existing <= available,
                    "storage_forecast_gates_release": reserve > 0 or write_limit is not None,
                    "device_budget_fits": device_seconds <= remaining,
                    "wall_budget_fits": wall_estimate <= remaining_wall}
        atomic_json(self.output/"forecast.json", forecast)
        storage_ok = not forecast["storage_forecast_gates_release"] or forecast["storage_fits"]
        return storage_ok and forecast["device_budget_fits"] and forecast["wall_budget_fits"]

    def formal_task(self, model, seed, phase, gate_file, suffix=""):
        selected = read(self.selected[model["id"]]["confirmation"][0]["task_path"])
        task = {**selected, "id": f'{model["id"]}-{phase}-{seed}{suffix}', "phase": phase,
                "source_identity": self.identity,
                "seed": seed, "target_tokens": 12*MI, "gate_file": str(gate_file),
                "eval_interval_tokens": MI, "health_interval_tokens": MI//2,
                "evaluation_start_tokens": 0}
        if phase == "b_repeat":
            task.update(eval_tokens=8192, evaluation_token_limit=MI//4,
                        evaluation_budget_key=f'{model["id"]}-repeat-{seed}')
        task.pop("resume_checkpoint", None)
        task.pop("resume_checkpoint_sha256", None)
        self.reserve_resources(task)
        if task.get("device", "npu:0").startswith("npu"):
            from .hbm_profile3 import HbmProfile, hbm_shape
            # Frozen formal requests must replay byte-for-byte on controller restart.
            frozen = read(self.output/"tasks"/task["id"]/"task.json")
            if frozen:
                if "reserved_hbm_bytes" in frozen:
                    task["reserved_hbm_bytes"] = frozen["reserved_hbm_bytes"]
                else:
                    task.pop("reserved_hbm_bytes", None)
            else:
                shape = hbm_shape(task)
                estimate = getattr(self, "inherited_hbm", {}).get(shape, 0)
                profile = getattr(self.supervisor, "hbm_profile", None)
                if profile:
                    estimate = max(estimate, self.supervisor.measured_hbm(shape))
                elif (self.output/"hbm-profile.sqlite3").is_file():
                    local = HbmProfile(self.output, readonly=True)
                    try:
                        estimate = max(estimate, local.summaries().get(shape, {}).get("reservation_bytes", 0))
                    finally:
                        local.db.close()
                if estimate:
                    task["reserved_hbm_bytes"] = max(task.get("reserved_hbm_bytes", 0), estimate)
        return task

    def retire_completed_stage_a(self):
        transferred = set(read(self.plan["stage_a_transfer"]["path"])["model_ids"]) if self.plan.get("stage_a_transfer") else set()
        for mid, choice in self.selected.items():
            if mid in transferred:
                continue  # Transferred old payloads are never this round's cleanup targets.
            tasks = [read(e["task_path"]) for field in ("selection_a1", "selection_a2", "confirmation") for e in choice[field]]
            self.prune(tasks, reason="formal and confirmation completed; selected A trajectory no longer needed")

    def run(self):
        if self.supervisor.state.get("status") == "completed":
            report_path = self.output/"report.json"
            if (not report_path.is_file() or sha256(report_path) != self.supervisor.state.get("terminal_report_sha256")
                    or read(report_path).get("status") != "completed"):
                raise Blocked("completed_campaign_report_missing_or_changed")
            # Successful A checkpoints may have been deliberately retired.
            # Validate the committed terminal records and retained formal
            # finals instead of reopening A or submitting new trajectories.
            for record in self.supervisor.state["tasks"].values():
                if record["status"] != "completed":
                    continue
                result = read(record["result_path"])
                task = read(record["task_path"])
                if (sha256(record["result_path"]) != record["result_sha256"]
                        or result.get("task_sha256") != sha256(record["task_path"])
                        or result.get("source_identity") != self.identity):
                    raise Blocked("completed_campaign_evidence_changed")
                if task["phase"] in {"b", "b_repeat"}:
                    checkpoint = result["checkpoint"]
                    if not Path(checkpoint["path"]).is_file() or sha256(checkpoint["path"]) != checkpoint["sha256"]:
                        raise Blocked("retained_formal_final_checkpoint_missing_or_changed")
            if self.supervisor.owned_members():
                raise Blocked("completed_campaign_has_live_descendants")
            return
        for subject in self.subjects:
            subject["effective_key"] = subject["model_key"]
        self.train_models(self.subjects)
        controls = self.controls()
        self.models += controls
        self.profile_models(controls)
        if not self.forecast():
            raise Blocked("resolved_controls_exceed_remaining_storage_or_device_budget")
        self.train_models(controls)
        manifest = self.manifest()
        manifest_path = freeze_json(self.output/"model-manifest.json", manifest)
        report = audit_stage_a(manifest, self.selected, self.matches, remaining_resources_ok=self.forecast())
        report_path = self.output/"stage-a-report.json"
        freeze_json(report_path, report)
        if report["status"] != "passed":
            raise Blocked(f'stage_a_gate_failed:{report["blocking_reasons"][:3]}')
        gate_path = self.output/"stage-b-gate.json"
        formal = [self.formal_task(model, FORMAL_SEED, "b", gate_path) for model in self.models]
        paths = [freeze_json(self.output/"tasks"/task["id"]/"task.json", task) for task in formal]
        create_gate(gate_path, manifest_path=manifest_path, report_path=report_path, task_paths=paths)
        self.supervisor.state["stage_b_gate"] = "passed"
        first = self.supervisor.run("b-formal", formal)
        for task in formal:
            if eligible(task, first[task["id"]]):
                raise Blocked(f'formal_task_unqualified:{task["id"]}')
        self.prune(formal, keep_final=True, reason="formal trajectory complete; keep complete final checkpoint")
        # Five declared topology/width slots cover all three widths. Selection
        # uses the selection split, before any confirmation evaluation.
        representatives = repeat_representatives(self.subjects, first)
        repeated_models = set(representatives)
        repeated_models.update(m["control_id"] for m in self.matches if m["subject_id"] in representatives)
        repeat_gate = self.output/"stage-b-repeat-gate.json"
        repeated = [self.formal_task(model, seed, "b_repeat", repeat_gate)
                    for model in self.models if model["id"] in repeated_models for seed in REPEAT_SEEDS]
        repeat_list = freeze_json(self.output/"repeat-selection.json", {"representatives": representatives,
            "model_ids": sorted(repeated_models), "task_ids": [t["id"] for t in repeated],
            "coverage_slots": REPEAT_SLOTS,
            "rule": "best selection NLL within fixed topology/width slots, then best remaining; model id tie break; matching controls included"})
        paths = [freeze_json(self.output/"tasks"/t["id"]/"task.json", t) for t in repeated]
        create_gate(repeat_gate, manifest_path=manifest_path, report_path=report_path,
                    task_paths=paths, gate_kind="repeat")
        repeat_results = self.supervisor.run("b-repeat", repeated)
        for task in repeated:
            if eligible(task, repeat_results[task["id"]]):
                raise Blocked(f'repeat_task_unqualified:{task["id"]}')
        self.prune(repeated, keep_final=True, reason="repeat trajectory complete; retain final state")
        confirmation_gate = self.output/"stage-b-confirmation-gate.json"
        confirmation = []
        for parent in formal+repeated:
            source = (first if parent["phase"] == "b" else repeat_results)[parent["id"]]
            cp = source["checkpoint"]
            task = {**parent, "id": parent["id"]+"-confirmation", "phase": "b_confirmation",
                    "gate_file": str(confirmation_gate), "allow_confirmation": True,
                    "confirmation_tokens": 131072, "eval_tokens": 256, "evaluation_start_tokens": 12*MI,
                    "resume_checkpoint": cp["path"], "resume_checkpoint_sha256": cp["sha256"]}
            confirmation.append(task)
        sealed_list = freeze_json(self.output/"confirmation-list.json", {"repeat_selection_path": str(repeat_list),
            "repeat_selection_sha256": sha256(repeat_list), "allowed_task_ids": [t["id"] for t in confirmation]})
        authorization = {"manifest_path": str(sealed_list), "manifest_sha256": sha256(sealed_list),
                         "allowed_task_ids": [t["id"] for t in confirmation]}
        for task in confirmation:
            task["confirmation_authorization"] = authorization
        paths = [freeze_json(self.output/"tasks"/t["id"]/"task.json", t) for t in confirmation]
        create_gate(confirmation_gate, manifest_path=manifest_path, report_path=report_path,
                    task_paths=paths, gate_kind="confirmation")
        confirmed = self.supervisor.run("b-confirmation", confirmation)
        self.prune(confirmation, reason="confirmation evaluated the retained formal final state without additional learning")
        summary = {"schema": "pilot-r3-final-summary-v1", "status": "completed",
            "subject_count": len(self.subjects), "control_count": len(controls),
            "formal_trajectories": len(formal), "repeat_trajectories": len(repeated),
            "formal_unique_loss_tokens": sum(r["training"]["tokens"] for r in [*first.values(), *repeat_results.values()]),
            "confirmed_trajectories": len(confirmed), "stage_a_report": str(report_path),
            "numerical_scope": "checkpoint state loading and numerical trajectory reproducibility are distinct; original strict failures retained"}
        atomic_json(self.output/"report.json", summary)
        self.supervisor.state["terminal_report_sha256"] = sha256(self.output/"report.json")
        self.supervisor.finish("completed")
        # The committed terminal report makes restart independent of retired
        # A checkpoints. A cleanup interruption leaves extra files, not a new
        # training request or an invalid partially reopened admission gate.
        self.retire_completed_stage_a()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = Campaign(read(args.plan), args.output)
    signal.signal(signal.SIGTERM, campaign.supervisor.request_stop)
    signal.signal(signal.SIGINT, campaign.supervisor.request_stop)
    try:
        campaign.run()
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        campaign.supervisor.stop_reason = campaign.supervisor.stop_reason or f"{type(exc).__name__}: {exc}"
        campaign.supervisor.stop_all()
        atomic_json(campaign.output/"report.json", {"schema": "pilot-r3-final-summary-v1",
            "status": campaign.supervisor.state["status"], "blocking_reason": str(exc),
            "stage_b_started": any(r["stage"].startswith("b-") and r.get("attempts")
                for r in campaign.supervisor.state["tasks"].values()),
            "counts": campaign.supervisor.state["counts"], "traceback": traceback.format_exc()})
        return 1
    finally:
        campaign.supervisor.close()


if __name__ == "__main__":
    raise SystemExit(main())
