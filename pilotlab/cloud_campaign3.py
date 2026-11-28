"""Advance the frozen 344-subject R3 DAG from durable project and cloud evidence.

advance() creates finite task requests and immutable model decisions. It never
launches workers. All paths it owns live below state_dir/campaign; the original
campaign files and training results remain byte-for-byte inputs.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import fcntl
import math
import os
from pathlib import Path
import statistics
import time

from .gate3 import (MI, audit_stage_a, create_gate, eligible, gate_path_for_source,
                    manifest_digest, read_evidence, stage_source_contract)
from .hbm_profile3 import DEFAULT_POLICY, HbmProfile, hbm_shape, reservation
from .io import atomic_json, sha256
from .prepare_resume3 import reference
from .round3 import Campaign, DIAGNOSTIC_SEEDS, FORMAL_SEED, REPEAT_SEEDS, REPEAT_SLOTS, repeat_representatives
from .scheduler3 import Blocked, freeze_json, read
from .source_migration3 import read_bound

REPO = Path(os.environ.get("TIDE_CLOUD_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
PAUSE = REPO / "docs/evidence/pilot-r3-stage-a-checkpoint-pause-20261127.json"
MIB = 2**20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def initial_sources(output):
    path = output / "original-inputs.json"
    if path.exists():
        return read(path)
    stopped = read(PAUSE)
    groups = {}
    for name, group in stopped["groups"].items():
        folder = Path(group["scheduler_output"])
        groups[name] = dict(scheduler=group["scheduler"], selected=reference(folder / "selected.json"),
                            resources=reference(folder / "model-resources.json"), original_output=str(folder))
    deferred = Path(groups["deferred"]["original_output"]).parent / "plan.json"
    plan = read(deferred)
    review = read_bound(plan["review"])
    source = dict(pause=reference(PAUSE), groups=groups, deferred_plan=reference(deferred),
                  primary_plan=review["primary_plan"], review=plan["review"])
    return read(freeze_json(path, source))


@lru_cache(maxsize=12)
def bound(path, checksum):
    return read_bound({"path": path, "sha256": checksum})


def fixed(ref):
    return bound(str(REPO / ref["path"]), ref["sha256"])


def terms(original, resources, rows, legacy):
    """Original and cloud resource reservations remain separate from task bytes."""
    task, ram, hbm = original, original.get("reserved_rss_bytes", 0), original.get("reserved_hbm_bytes", 0)
    batch = str(task["batch_size"])
    profile = resources.get(task["model_id"], {})
    ram = max(ram, math.ceil(1.5 * profile.get("rss_peak_bytes_by_batch", {}).get(batch, 0)) + 256 * MIB)
    hbm = max(hbm, legacy.get(hbm_shape(task), 0))
    for row in rows:
        memory = row.get("device_memory", {})
        if memory:
            ram = max(ram, math.ceil(1.5 * memory.get("cpu_rss_peak_bytes", 0)) + 256 * MIB)
            peak = max((int(value or 0) for key, value in memory.items()
                        if key.startswith(("npu_peak_", "npu_startup_peak_"))), default=0)
            hbm = max(hbm, reservation(peak, *DEFAULT_POLICY))
    rate = profile.get("tokens_per_second_by_batch", {}).get(batch)
    measured = [row["training"].get("tokens_per_second") for row in rows
                if row.get("training", {}).get("tokens_per_second")]
    backend = task["device"].split(":")[0]
    if measured:
        rate = min(measured)
    elif backend == "cpu":
        rate = profile.get("cpu_tokens_per_second") or rate
    exposure = task["target_tokens"] - task.get("evaluation_start_tokens", 0)
    seconds = math.ceil((exposure + 2 * task["eval_tokens"]) / rate) if rate and rate > 0 else None
    return dict(backend=backend, threads=task["cpu_threads"], ram_mib=max(4096, math.ceil(ram/MIB)),
                hbm_mib=math.ceil((hbm or 8*2**30) / MIB) if backend == "npu" else 0,
                estimated_seconds=seconds)


class CloudCampaign(Campaign):
    def __init__(self, output, initial, accepted):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.inputs = initial_sources(output)
        self.plan = fixed(self.inputs["primary_plan"])
        self.deferred = fixed(self.inputs["deferred_plan"])
        self.review = fixed(self.inputs["review"])
        self.identity, self.protocol = self.plan["source_identity"], self.plan["protocol"]
        self.resources, self.selected, self.records, self.original_tasks = {}, {}, {}, {}
        self.inherited_hbm, self.peaks, self.entries, self.pending, self.blocked = {}, {}, {}, {}, []
        self.accepted = accepted
        self.initial = {row["id"]: row for row in initial}
        self.supervisor = None
        self.matches = []
        migration = fixed(self.plan["stage_a_migration"])
        for tid, item in migration["completed"].items():
            self.original_tasks[tid] = item["task"]["path"]
            self.records[tid] = dict(task_path=item["task"]["path"], result_path=item["result"]["path"],
                                     task_sha256=item["task"]["sha256"], result_sha256=item["result"]["sha256"])
        for name, group in self.inputs["groups"].items():
            self.resources.update(fixed(group["resources"]))
            self.selected.update(fixed(group["selected"]))
            for tid, row in fixed(group["scheduler"])["tasks"].items():
                if row["status"] == "superseded":
                    continue
                if tid in self.records and tid not in self.plan.get("qualified_attempts", {}):
                    continue
                file = row.get("task_path") or str(Path(group["original_output"]) / "tasks" / tid / "task.json")
                self.original_tasks[tid] = file
                if row["status"] in {"completed", "recipe_rejected"}:
                    self.records[tid] = dict(task_path=file, result_path=row["result_path"],
                                             result_sha256=row["result_sha256"])
            if (Path(group["original_output"]) / "hbm-profile.sqlite3").is_file():
                profile = HbmProfile(group["original_output"], readonly=True)
                try:
                    for shape, meta in profile.summaries().items():
                        self.inherited_hbm[shape] = max(meta["reservation_bytes"], self.inherited_hbm.get(shape, 0))
                finally:
                    profile.db.close()
        self.resources.update(read(output / "model-resources.json", {}))
        self.selected.update(read(output / "selected.json", {}))
        old = fixed(self.review["original_inputs"]["r07_selected"])
        self.selected.update({mid: old[mid] for mid, row in self.review["subjects"].items() if row["mode"] == "carry_recipe"})
        self.subjects = [dict(m, effective_key=m["model_key"]) for m in self.plan["subjects"]]
        for model in self.deferred["subjects"]:
            mid, spec = model["id"], model["model"]
            target = {**spec, "config": {**spec["config"], **self.review["subjects"][mid]["model_policy"]}}
            self.subjects.append({**model, "effective_key": model["model_key"], "model": target})
        self.models = list(self.subjects)
        self.rollout = fixed(self.plan["fresh_shannon"])
        self.fresh = set(self.rollout["model_ids"])
        self.arms = {name: fixed(a["identity"]) for name, a in self.review["sources"].items()}
        self.weak = fixed(self.rollout["worker_identity"])

    def prune(self, tasks, **kwargs):
        # Archive ownership is separate. A/B needs the complete scientific evidence.
        return

    def retire_selected_parents(self, model, **kwargs):
        return

    def task(self, model, phase, seed, **kwargs):
        task = super().task(model, phase, seed, **kwargs)
        mid = model["id"]
        if mid in self.fresh:
            task["id"] = mid + "-weak-log-v1" + task["id"][len(mid):]
            task["source_identity"] = self.weak
        elif mid in self.review["subjects"]:
            row = self.review["subjects"][mid]
            task["source_identity"] = self.arms[row["target_arm"]]
            request = row["batch_requests"].get(str(task["batch_size"]), {})
            if task["device"].startswith("npu"):
                task["reserved_hbm_bytes"] = max(task.get("reserved_hbm_bytes", 0), request.get("hbm", 0),
                                                  row["unmeasured_npu_request"] if not request.get("peak") else 0)
            if request.get("rss"):
                task["reserved_rss_bytes"] = max(request["rss"], task.get("reserved_rss_bytes", 0))
        return task

    def recipes(self, model):
        rows = super().recipes(model)
        review = self.review["subjects"].get(model["id"])
        return [r for r in rows if not review or r["recipe_id"] in review["recipes"]]

    def evidence(self, task):
        return self.entries[task["id"]]

    def manifest(self):
        return {**super().manifest(), "stage_a_dispositions": self.inputs["review"]}

    def remember_measurement(self, task, result):
        key = hbm_shape(task), task["device"].split(":")[0]
        observation = self.peaks.setdefault(key, {"training": {}, "device_memory": {}})
        rate = result.get("training", {}).get("tokens_per_second")
        if rate and rate > 0:
            previous = observation["training"].get("tokens_per_second", rate)
            observation["training"]["tokens_per_second"] = min(rate, previous)
        for name, value in result.get("device_memory", {}).items():
            if (name.startswith(("npu_peak_", "npu_startup_peak_")) or name == "cpu_rss_peak_bytes") and value:
                memory = observation["device_memory"]
                memory[name] = max(value, memory.get(name, 0))

    def consume(self, task, entry):
        """Expose a relocated checkpoint to the old generator without editing raw evidence."""
        _, result, _ = read_evidence(entry)
        self.entries[task["id"]] = entry
        self.remember_measurement(task, result)
        cp = result.get("checkpoint")
        if cp and entry.get("checkpoint_path"):
            result["checkpoint"] = {**cp, "path": entry["checkpoint_path"]}
        return result

    def resolution(self, task):
        tid = task["id"]
        path = self.original_tasks.get(tid)
        if path:
            original = read(path)
            require(original and all(original.get(k) == task.get(k) for k in
                    ("id", "phase", "model_id", "seed", "target_tokens", "batch_size", "init_recipe", "lr", "recipe_id")),
                    "generated Stage A rung does not match its frozen original: " + tid)
            if task.get("resume_checkpoint"):
                require(task["resume_checkpoint_sha256"] == original.get("resume_checkpoint_sha256"),
                        "original child refers to a different selected parent: " + tid)
            task.clear()
            task.update(original)
        else:
            path = str(freeze_json(self.output / "tasks" / tid / "task.json", task))
        fp = sha256(path)
        if fp in self.accepted:
            value = self.accepted[fp]
            entry = dict(task_path=path, task_sha256=fp, result_path=value["result_path"],
                         runtime_plan=reference(value["runtime_plan_path"]), cloud_verdict=reference(value["verdict_path"]),
                         cloud_receipt=reference(value["receipt_path"]))
            if value.get("checkpoint_receipt"):
                entry["checkpoint_receipt"] = value["checkpoint_receipt"]
            if value.get("checkpoint_path"):
                entry.update(checkpoint_path=value["checkpoint_path"],
                             checkpoint_sha256=read(value["result_path"])["checkpoint"]["sha256"])
            result = self.consume(task, entry)
            allowed_negative = (result.get("status") == "recipe_rejected" and
                                result.get("failure_class") == "declared_optimizer_instability" and task["phase"] in {"a1","a2"}
                                or result.get("failure_class") == "capacity_error" and tid.endswith("-capacity-large"))
            if result["status"] != "completed" and not allowed_negative:
                raise Blocked("unqualified cloud result: " + tid + ": " + str(result.get("stop_reason")))
            return result
        if tid in self.records:
            return self.consume(task, self.records[tid])
        if tid in self.initial:
            row = self.initial[tid]
            require(fp == row["fingerprint"], "queued task fingerprint changed: " + tid)
            self.pending[tid] = {**row, "task_path": path}
        else:
            measure = self.peaks.get((hbm_shape(task), task["device"].split(":")[0]))
            self.pending[tid] = {"id": tid, "task_path": path, "path": path, "fingerprint": fp,
                                 "source": manifest_digest(task["source_identity"]), "phase": task["phase"],
                                 "group": "deferred" if task["model_id"] in self.review["subjects"] else "primary",
                                 "prior_results": [], "reserved_evaluation_tokens": 0,
                                 **terms(task, self.resources, [measure] if measure else [], self.inherited_hbm)}
        return None

    def run_flow(self, model, *, profile_only=False):
        mid = model["id"]
        if mid in self.selected:
            return
        review = self.review["subjects"].get(mid, {})
        active_plan = self.plan
        if review:
            self.plan = self.deferred
        flow = self.profile_flow([model]) if profile_only else self.training_flow(
            [model], per_model_decisions=True, overlap_initialization=bool(review.get("historical_init_qualified")),
            a2_recipe_count=min(2, len(review.get("recipes", range(6)))))
        answers = None
        try:
            while True:
                wave = flow.send(answers)
                answers = {}
                waiting = False
                for request in wave:
                    for task in request["tasks"]:
                        result = self.resolution(task)
                        if result is None:
                            waiting = True
                        else:
                            answers[task["id"]] = result
                if waiting:
                    return
        except StopIteration:
            return
        except (Blocked, ValueError, KeyError, OSError) as exc:
            self.blocked.append(f"{mid}: {exc}")
        finally:
            flow.close()
            self.plan = active_plan

    def run_all(self, tasks):
        results = {}
        for task in tasks:
            try:
                result = self.resolution(task)
            except (Blocked, ValueError, KeyError, OSError) as exc:
                self.blocked.append(f"{task['id']}: {exc}")
                continue
            if result is not None:
                confirmation = result.get("confirmation_final", {})
                if eligible(task, result) or (task["phase"] == "b_confirmation" and
                        (confirmation.get("status") != "completed" or confirmation.get("split") != "confirmation"
                         or confirmation.get("tokens") != task["confirmation_tokens"]
                         or not math.isfinite(confirmation.get("nll", math.nan)) or confirmation["nll"] < 0
                         or result["training"].get("segment_tokens") != 0)):
                    self.blocked.append("unqualified " + task["phase"] + " evidence for " + task["id"])
                else:
                    results[task["id"]] = result
        return results

    def response(self, phase):
        mids = {m["id"] for m in self.subjects}
        return {"phase": phase, "finished": phase == "completed", "blocked": self.blocked,
                "selected_subjects": len(mids.intersection(self.selected)),
                "selected_controls": len(set(self.selected) - mids), "tasks": list(self.pending.values())}

    def formal_tasks(self, models, seeds, phase, gate, contract):
        tasks = []
        for model in models:
            item = contract["models"][model["id"]]
            path = gate_path_for_source(gate, item["b_source"], self.identity)
            for observation in self.selected[model["id"]]["confirmation"]:
                selected_task, selected_result, _ = read_evidence(observation)
                self.remember_measurement(selected_task, selected_result)
            for seed in seeds:
                task = self.formal_task(model, seed, phase, path)
                task.update(source_identity=item["b_source"], model=item["b_model"])
                task.pop("stage_a_migration", None)
                tasks.append(task)
        return tasks

    def permit(self, gate, phase, tasks, manifest_path, report_path):
        paths = [freeze_json(self.output / "tasks" / t["id"] / "task.json", t) for t in tasks]
        # An already frozen grant was issued only after the full A evidence audit.
        if not gate.exists():
            create_gate(gate, manifest_path=manifest_path, report_path=report_path,
                        task_paths=paths, gate_kind=phase)
        stored = read(gate)
        require(stored["task_sha256"] == {t["id"]: sha256(p) for t, p in zip(tasks, paths)}
                and stored["manifest_sha256"] == sha256(manifest_path)
                and stored["report_sha256"] == sha256(report_path), "frozen B grant or task list changed")

    def forecast(self, resources):
        unselected = {m["id"]: self.resources[m["id"]] for m in self.models if m["id"] not in self.selected}
        return cloud_forecast(self.selected, resources, unselected, self.accepted)

    def stages(self, resources):
        for model in self.subjects:
            self.run_flow(model)
        if any(m["id"] not in self.selected for m in self.subjects):
            return self.response("stage-a-subjects")
        controls = self.controls()
        self.models.extend(controls)
        for model in controls:
            self.run_flow(model, profile_only=True)
        if any(m["id"] not in self.resources for m in controls):
            return self.response("stage-a-control-capacity")
        forecast_path = self.output / "control-resource-forecast.json"
        if not forecast_path.exists():
            estimate = self.forecast(resources)
            if not estimate["fits"]:
                atomic_json(self.output / "forecast.json", estimate)
                self.blocked.extend(estimate["reasons"])
                return self.response("stage-a-control-budget")
            freeze_json(forecast_path, estimate)
        for model in controls:
            self.run_flow(model)
        if any(m["id"] not in self.selected for m in controls):
            return self.response("stage-a-controls")
        manifest = self.manifest()
        manifest_path = freeze_json(self.output / "model-manifest.json", manifest)
        report_path = self.output / "stage-a-report.json"
        if not report_path.exists():
            estimate = self.forecast(resources)
            atomic_json(self.output / "forecast.json", estimate)
            if not estimate["fits"]:
                self.blocked.extend(estimate["reasons"])
                return self.response("stage-a-gate")
            report = audit_stage_a(manifest, self.selected, self.matches, remaining_resources_ok=True)
            if report["status"] != "passed":
                atomic_json(self.output / "stage-a-audit-blocked.json", report)
                self.blocked.extend(str(v) for v in report["blocking_reasons"])
                return self.response("stage-a-gate")
            freeze_json(self.output / "gate-resource-forecast.json", estimate)
            freeze_json(report_path, report)
        contract = stage_source_contract(manifest, self.selected)
        gate = self.output / "stage-b-gate.json"
        formal = self.formal_tasks(self.models, [FORMAL_SEED], "b", gate, contract)
        self.permit(gate, "formal", formal, manifest_path, report_path)
        first = self.run_all(formal)
        if len(first) < len(formal):
            return self.response("stage-b-formal")
        reps = repeat_representatives(self.subjects, first)
        repeated_ids = set(reps) | {m["control_id"] for m in self.matches if m["subject_id"] in reps}
        repeat_gate = self.output / "stage-b-repeat-gate.json"
        repeated = self.formal_tasks([m for m in self.models if m["id"] in repeated_ids], REPEAT_SEEDS,
                                     "b_repeat", repeat_gate, contract)
        selection = freeze_json(self.output / "repeat-selection.json", {"representatives": reps,
            "model_ids": sorted(repeated_ids), "task_ids": [t["id"] for t in repeated], "coverage_slots": REPEAT_SLOTS,
            "formal_evidence": {t["id"]: self.evidence(t) for t in formal}})
        self.permit(repeat_gate, "repeat", repeated, manifest_path, report_path)
        second = self.run_all(repeated)
        if len(second) < len(repeated):
            return self.response("stage-b-repeat")
        parents = formal + repeated
        seal = freeze_json(self.output / "confirmation-list.json", {"repeat_selection_path": str(selection),
            "repeat_selection_sha256": sha256(selection), "allowed_task_ids": [t["id"] + "-confirmation" for t in parents]})
        auth = {"manifest_path": str(seal), "manifest_sha256": sha256(seal), "allowed_task_ids": read(seal)["allowed_task_ids"]}
        gate = self.output / "stage-b-confirmation-gate.json"
        confirmed = []
        for parent in parents:
            cp = (first if parent["phase"] == "b" else second)[parent["id"]]["checkpoint"]
            path = gate_path_for_source(gate, parent["source_identity"], self.identity)
            confirmed.append({**parent, "id": parent["id"] + "-confirmation", "phase": "b_confirmation",
                "gate_file": str(path), "allow_confirmation": True, "confirmation_tokens": 131072,
                "eval_tokens": 256, "evaluation_start_tokens": 12*MI, "resume_checkpoint": cp["path"],
                "resume_checkpoint_sha256": cp["sha256"], "confirmation_authorization": auth})
        self.permit(gate, "confirmation", confirmed, manifest_path, report_path)
        final = self.run_all(confirmed)
        if len(final) < len(confirmed):
            return self.response("stage-b-confirmation")
        freeze_json(self.output / "report.json", dict(schema="pilot-r3-cloud-final-v1", status="completed",
            stage_a_report=reference(report_path), formal=len(first), repeats=len(second), confirmed=len(final),
            subjects=len(self.subjects), controls=len(controls),
            confirmation_evidence={t["id"]: self.evidence(t) for t in confirmed}))
        return self.response("completed")


def cloud_samples(accepted):
    """Count each worker's actual update time inside its node's overlapping wall clock."""
    by_shape, by_node = defaultdict(list), defaultdict(list)
    for fp, entry in accepted.items():
        task = read(entry["task_path"])
        raw = read(entry["result_path"])
        receipt = read(entry["receipt_path"])
        require(sha256(entry["task_path"]) == fp == raw["task_sha256"] == receipt["fingerprint"]
                and sha256(entry["result_path"]) == receipt["artifacts"]["result.json"]["sha256"]
                and task["id"] == raw["id"] == receipt["task"], "resource observation belongs to another task")
        train = raw.get("training", {})
        seconds, tokens = train.get("seconds", 0), train.get("segment_tokens", 0)
        if not (seconds and tokens and seconds > 0 and tokens > 0):
            continue
        start, finish = receipt["started"], receipt["finished"]
        require(start < finish and 0 < seconds <= finish-start+1, "cloud training time outside allocation receipt")
        backend = task["device"].split(":")[0]
        device = (str(receipt.get("board")), str(receipt.get("chip"))) if backend == "npu" else None
        by_node[(receipt["job"], backend)].append((start, finish, seconds, device))
        if (receipt["status"] == "completed" and task["phase"] in {"a1", "a2", "a3"}):
            by_shape[(backend, hbm_shape(task))].append(tokens / seconds)
    return by_shape, by_node


def measured_parallelism(nodes, by_node, cpu_request):
    widths = {b: [0., 0.] for b in ("cpu", "npu")}
    evidence = {}
    for name, node in nodes.items():
        workers = [w for w in node.get("workers", []) if w.get("state") == "serving"
                   and time.time() - w.get("at", 0) < 180]
        threads = sum(w.get("capacity", {}).get("threads", 0) for w in workers)
        boards = [card for w in workers for card in w.get("cards", {}).values()]
        chips = sum(len(c.get("chips") or {}) for c in boards)
        evidence[name] = dict(threads=threads, boards=len(boards), logical_chips=chips, backends={})
        for backend, capacity in (("cpu", threads // cpu_request), ("npu", chips)):
            rows = by_node.get((name, backend), [])
            if not capacity:
                continue
            if not rows:
                widths[backend][0] += capacity
                widths[backend][1] += capacity
                evidence[name]["backends"][backend] = {"workers": 0, "basis": "available hardware, one worker per slot"}
                continue
            first = min(r[0] for r in rows)
            last = max(r[1] for r in rows)
            train = sum(r[2] for r in rows)
            realized = train / (last-first)
            devices = {r[3] for r in rows}
            if backend == "npu":
                observed = len(devices)
                # Each populated chip contributes the concurrent worker training-seconds
                # divided by its real wall span. Never infer extra sharing for an unmeasured chip.
                occupied_width = sum(sum(r[2] for r in rows if r[3] == d) /
                                     (max(r[1] for r in rows if r[3] == d) - min(r[0] for r in rows if r[3] == d))
                                     for d in devices)
                unmeasured = max(0, capacity - observed)
                conservative = realized + unmeasured * min(1., occupied_width / observed)
                optimistic = occupied_width + unmeasured * (occupied_width / observed)
            else:
                # CPU workers can have different thread requests. The observed whole-node
                # concurrency includes competition from colocated NPU training.
                conservative = min(realized, capacity)
                optimistic = max(conservative, capacity)
            widths[backend][0] += conservative
            widths[backend][1] += max(conservative, optimistic)
            evidence[name]["backends"][backend] = dict(workers=len(rows), wall_seconds=last-first,
                training_worker_seconds=train, observed_concurrent_workers=realized,
                observed_chips=len(devices) if backend == "npu" else None,
                conservative_parallelism=conservative, optimistic_parallelism=optimistic)
    return widths, evidence


def cloud_forecast(selected, resources, unselected=None, accepted=None):
    """Estimate the sequential stages using matching-model throughput and measured node overlap."""
    reasons, now = [], time.time()
    if not resources or not 0 <= now - resources.get("observed_unix", 0) <= 600:
        return {"fits": False, "reasons": ["cloud allocation or budget observation is missing or stale"]}
    boards = resources.get("allocated_physical_boards", 0)
    nodes = {k: n for k, n in resources.get("node_capacity", {}).items() if n.get("phase", "").lower() == "running"}
    if not boards or not nodes:
        return {"fits": False, "reasons": ["cloud allocation has no measured board or CPU capacity"]}
    by_shape, by_node = cloud_samples(accepted or {})
    rates, ratios = {}, defaultdict(list)
    threads = 1
    for mid, choice in selected.items():
        observations = [(read(e["task_path"]), read(e["result_path"])) for e in choice["confirmation"]]
        available = [r["training"]["tokens_per_second"] for _, r in observations if r["training"].get("tokens_per_second")]
        if not available or min(available) <= 0:
            reasons.append("missing A3 training rate: " + mid)
            continue
        task = observations[0][0]
        backend = task["device"].split(":")[0]
        local = [r["training"]["tokens_per_second"] for _, r in observations if not r.get("execution_runtime")
                 and r["training"].get("tokens_per_second")]
        matched = by_shape.get((backend, hbm_shape(task)), [])
        rates[mid] = dict(backend=backend, reference_rate=min(available), matched_cloud_workers=len(matched),
                          slow_rate=min(matched) if matched else None, fast_rate=max(matched) if matched else None)
        if matched and local:
            ratios[backend].append(statistics.median(matched) / min(local))
        if backend == "cpu":
            threads = max(threads, task["cpu_threads"])
    for mid, profile in (unselected or {}).items():
        measured = [v for v in profile.get("tokens_per_second_by_batch", {}).values() if v and v > 0]
        if not measured:
            reasons.append("missing measured control capacity: " + mid)
            continue
        rates[mid] = dict(backend="npu", reference_rate=min(measured), matched_cloud_workers=0,
                          slow_rate=min(measured), fast_rate=max(measured))
    ratio_range = {}
    for backend, values in ratios.items():
        ranked = sorted(values)
        ratio_range[backend] = dict(pairs=len(ranked), low=ranked[math.floor((len(ranked)-1)*.1)],
                                    high=ranked[math.ceil((len(ranked)-1)*.9)])
    for value in rates.values():
        if not value["slow_rate"]:
            ratio = ratio_range.get(value["backend"], {"low": 1., "high": 1.})
            value["slow_rate"] = value["reference_rate"] * min(1., ratio["low"])
            value["fast_rate"] = value["reference_rate"] * max(1., ratio["high"])
    widths, node_evidence = measured_parallelism(nodes, by_node, threads)
    if sum(n["boards"] for n in node_evidence.values()) != boards or len(nodes) != resources.get("active_nodes"):
        reasons.append("fresh runq chip maps and physical allocation counts disagree")
    def duration(index, rate):
        formal, repeats = defaultdict(list), defaultdict(list)
        for model, value in rates.items():
            seconds = MI / value[rate]
            backend = value["backend"]
            formal[backend].append((21.25 if model in (unselected or {}) else 13) * seconds)
            repeats[backend].append(12.25 * seconds)
        for backend in formal:
            if widths[backend][index] <= 0:
                reasons.append("no observed execution capacity for " + backend)
        def wave(rows, backend):
            return max(max(rows), sum(rows) / max(.01, widths[backend][index]))
        first = max((wave(rows, b) for b, rows in formal.items()), default=0.)
        # The two repeats of a model are independent, but start after formal ranking.
        second = max((wave(sorted(rows, reverse=True)[:24]*2, b) for b, rows in repeats.items()), default=0.)
        return 1.5 * (first + second)
    upper = duration(0, "slow_rate")
    lower = duration(1, "fast_rate")
    wall, remaining = resources["deadline_unix"] - now, resources["remaining_device_seconds"]
    if upper > wall:
        reasons.append("conservative makespan estimate exceeds the absolute campaign deadline")
    if upper * boards > remaining:
        reasons.append("conservative allocated physical-board seconds exceed the remaining campaign budget")
    return dict(fits=not reasons, reasons=sorted(set(reasons)), observed_unix=now, cloud=resources,
                estimated_remaining_wall_seconds=upper, estimated_remaining_device_seconds=upper*boards,
                wall_seconds_range=[lower, upper], allocated_board_seconds_range=[lower*boards, upper*boards],
                remaining_wall_seconds=wall, remaining_device_seconds=remaining, rates=rates,
                matched_runtime_speed_ratios=ratio_range, node_measurements=node_evidence,
                parallelism_by_backend=widths, physical_boards_charged=boards,
                logical_chips=sum(n["logical_chips"] for n in node_evidence.values()),
                accepted_attempts=len(accepted or {}), training_observations=sum(map(len, by_node.values())),
                scope="Slower/faster matching-shape measurements, wall overlap on each node, 50% margin; estimates are not guarantees.")


def advance(state_dir, initial_inventory, accepted):
    """Return currently runnable native TIDE tasks with the exact frozen request path."""
    state_dir = Path(state_dir).resolve()
    inventory = read(initial_inventory) if isinstance(initial_inventory, (str, Path)) else initial_inventory
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "advance.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        campaign = CloudCampaign(state_dir / "campaign", inventory, accepted)
        try:
            return campaign.stages(read(state_dir / "resources.json"))
        except (Blocked, ValueError, KeyError, OSError) as exc:
            campaign.blocked.append(str(exc))
            return campaign.response("stage-a-or-b-evidence")
