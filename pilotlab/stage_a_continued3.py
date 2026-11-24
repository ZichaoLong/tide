"""Run fresh A subjects and interrupted A trajectories under one resource owner."""
from pathlib import Path

from .adapter_recovery3 import VERIFIER_CONTINUATION
from .cutover_review3 import CONTROLLER_CONTINUATIONS, HBM_CONTINUATION, reviewed_storage_settings
from .occupied_npu3 import apply_reviewed_policy
from .prepare_resume3 import reference
from .round3 import Campaign
from .resource_placement3 import POLICY as DIRECT_POLICY, ResourcePlacement
from .scheduler3 import freeze_json, read
from .shannon_recipes3 import DEFAULT_POLICY, cohort_runtime, cohort_subjects, verify_cohort
from .source_migration3 import ACTIVITY_CONTINUATION, POLICY, POOL_ACCEPTANCE, QUAD_TOTAL_ACCEPTANCE, REGIONAL_PRIOR_ACCEPTANCE, RECEIPT, digest, read_bound, revised_model, verify_recipe


def verify_scope(plan):
    batch = read_bound(plan["stage_a_migration"])
    review = read_bound(batch["readiness"])
    old = read_bound(batch["old_plan"])
    acceptance = read_bound(batch["qualification"])
    config = plan["config"]
    prior = {**old["config"], **reviewed_storage_settings(review)}
    apply_reviewed_policy(prior, review, config.get("run_id"))
    from .runq_campaign import apply_runq
    apply_runq(prior, review)
    placement = review.get("stage_a_cpu_placement", old.get("stage_a_cpu_placement")) or {}
    if placement.get("policy") == DIRECT_POLICY:
        prior["resource_dispatch"] = True
        if (plan.get("cpu_recipe_calibration") is not False or config.get("resource_dispatch") is not True
                or placement.get("unprofiled_batches") != [2, 4]):
            raise ValueError("direct resource placement requires calibration-free admission")
    pool_floor = acceptance.get("schema") == POOL_ACCEPTANCE
    expected = [{**row, "model":revised_model(row["model"], pool_floor=pool_floor)} for row in old["subjects"]]
    rollout = read_bound(plan["fresh_shannon"]) if "fresh_shannon" in plan else None
    if rollout:
        verify_cohort(rollout, plan["source_identity"])
    planned = cohort_subjects(expected, rollout) if rollout else expected
    if (batch["schema"] != "pilot-r3-stage-a-migration-plan-v1" or planned != plan["subjects"]
            or digest(expected) != review["stage_a_scope"]["prospective_recipe_check"]["canonical_prospective_subjects_sha256"]
            or review["inputs"]["campaign_plan"]["sha256"] != batch["old_plan"]["sha256"]
            or review["inputs"]["foundation_acceptance"]["sha256"] != batch["qualification"]["sha256"]
            or batch["new_source_digest"] != digest(plan["source_identity"])
            or batch["foundation_commit"] != review["foundation"]["published_head"]
            or old["stage_a_revision"] != plan["stage_a_revision"]
            or config.get("max_inflight_checkpoint_bytes") != prior.get("max_inflight_checkpoint_bytes")
            or any(config[k] != prior[k] for k in ("python","data","budget_ledger","deadline_unix",
                   "target_npu_devices","max_cpu_task_threads","max_rss_bytes",
                   "shared_max_hbm_bytes","host_cpu_affinity","max_workers","max_cpu_workers"))
            or "resume_import" in config):
        raise ValueError("A migration changed its reviewed scope, source or resource contract")
    if review.get("schema") in CONTROLLER_CONTINUATIONS:
        from .prepare_campaign3 import source_inventory
        if (plan["source_identity"] != old["source_identity"]
                or config.get("worker_source") != prior.get("worker_source", prior["source"])
                or config.get("worker_foundation") != prior.get("worker_foundation", prior["foundation"])
                or config.get("resume_adapter") != prior["resume_adapter"]
                or config["shared_max_hbm_bytes"] != 48 * 2**30 or config.get("exclusive_max_hbm_bytes") != 52 * 2**30
                or config.get("dynamic_shared_admission") is not True):
            raise ValueError("controller continuation changed its frozen worker or admission limits")
        if review["schema"] != HBM_CONTINUATION:
            runtime = {"source", "foundation", "worker_source", "worker_foundation", "run_id",
                       "prior_evaluation_reservations", "migration_evaluation_reservations"}
            if {k:v for k,v in config.items() if k not in runtime} != {k:v for k,v in prior.items() if k not in runtime}:
                raise ValueError("controller continuation changed the prior resource or training configuration")
        # A controller-only restart can use a new immutable code directory while the
        # same live workers, task configuration and budget still own their records.
        controller_source = plan.get("controller_source", config["source"])
        source_inventory({"source":controller_source, "foundation":rollout["foundation"] if rollout else config["foundation"],
                          "source_identity":plan["controller_source_identity"]})
        placement = plan.get("stage_a_cpu_placement")
        if placement != review.get("stage_a_cpu_placement", old.get("stage_a_cpu_placement")):
            raise ValueError("CPU placement differs from the reviewed A continuation")
        if placement is not None:
            from .cpu_placement3 import POLICY as cpu_policy
            from .dispatch_placement3 import DISPATCH_POLICIES
            if placement["policy"] in DISPATCH_POLICIES:
                calibration = placement["calibration_models"]
                if (not isinstance(calibration, list) or len(calibration) != len(set(calibration))
                        or not set(calibration) <= {m["id"] for m in expected}):
                    raise ValueError("invalid dispatch calibration scope")
            if (placement["policy"] not in (cpu_policy, DIRECT_POLICY, *DISPATCH_POLICIES) or not set(placement["priority_models"]) <= {m["id"] for m in expected}
                    or not config.get("dynamic_shared_admission")):
                raise ValueError("invalid A-only CPU placement contract")
    elif "worker_source" in config or "worker_foundation" in config:
        raise ValueError("source migration must execute its qualified frozen source")
    elif review.get("schema") == ACTIVITY_CONTINUATION:
        runtime = {"source", "foundation", "run_id", "resume_adapter", "prior_evaluation_reservations",
                   "migration_evaluation_reservations"}
        previous = read_bound(old["stage_a_migration"])
        if (batch["foundation_commit"] != previous["foundation_commit"] or batch["qualification"] != previous["qualification"]
                or plan["stage_a_cpu_placement"] != old["stage_a_cpu_placement"]
                or config["resume_adapter"] != reference(Path(config["source"])/"pilotlab/resume_adapter3.py")
                or {k:v for k,v in config.items() if k not in runtime} != {k:v for k,v in old["config"].items() if k not in runtime}):
            raise ValueError("activity continuation changed its training or resource contract")
    elif review.get("schema") == VERIFIER_CONTINUATION:
        runtime = {"source", "foundation", "worker_source", "worker_foundation", "run_id", "resume_adapter",
                   "prior_evaluation_reservations", "migration_evaluation_reservations"}
        prior = old["config"]
        tid = review["priority_task_id"]
        recovered = read_bound(review["recovered_result"])
        evidence = recovered["post_run_verification"]
        source = Path(config["source"]) / "pilotlab"
        if (evidence["schema"] != VERIFIER_CONTINUATION or evidence["original_return_code"] != 1
                or evidence["scheduler"] != batch["old_scheduler"] or recovered["id"] != tid
                or batch["completed"][tid]["result"] != review["recovered_result"] or tid in batch["outstanding"]
                or config["resume_adapter"] != reference(source/"resume_adapter3.py")
                or any(plan["source_identity"]["source_files"][str(source/k)] != v for k,v in evidence["verifier_sha256"].items())
                or plan["stage_a_cpu_placement"] != old["stage_a_cpu_placement"]
                or {k:v for k,v in config.items() if k not in runtime} != {k:v for k,v in prior.items() if k not in runtime}):
            raise ValueError("verifier continuation changed the recovered evidence, worker, placement or resource contract")
    base = Path(config.get("worker_foundation", config["foundation"]))/"src/tide"
    actual = {Path(k).relative_to(base).as_posix():v for k,v in plan["source_identity"]["source_files"].items()
              if Path(k).is_relative_to(base)}
    if (actual != acceptance["authority"]["files_sha256"] or
            acceptance.get("schema") in ("pilot-r3-quad-reception-acceptance-v1", "pilot-r3-timed-ingress-acceptance-v1", POOL_ACCEPTANCE, QUAD_TOTAL_ACCEPTANCE, REGIONAL_PRIOR_ACCEPTANCE)
            and batch["foundation_commit"] != acceptance["authority"]["commit"]):
        raise ValueError("training foundation differs from the qualified release")
    return old["stage_a_revision"]


class MigrationCampaign(Campaign):
    def __init__(self, plan, output):
        self.migration = read_bound(plan["stage_a_migration"])
        self.pool_floor = any(row["model"].get("config", {}).get("pool_support_policy") == POLICY for row in plan["subjects"])
        super().__init__(plan, output)
        self.repaired_tasks = {}
        self.supervisor.worker_runtimes = {}
        self.fresh_ids = set()
        if "fresh_shannon" in plan:
            rollout = read_bound(plan["fresh_shannon"])
            self.fresh_ids = set(rollout["model_ids"])
            self.fresh_identity = read_bound(rollout["worker_identity"])
            self.fresh_runtime = cohort_runtime(rollout)
        for tid, ref in plan.get("qualified_attempts", {}).items():
            from .source_migration3 import migration_inputs
            task = read_bound(ref)
            receipt = read_bound(task["stage_a_migration"])
            original, _ = migration_inputs(task, receipt)
            record = self.supervisor.state["tasks"][tid]
            if (task["phase"] != "a2" or tid != task["id"] or receipt["mode"] != "backward_range_repair"
                    or original["source_identity"] != self.identity or not plan["config"].get("runq_root")):
                raise ValueError("unbound repaired Stage A request")
            if record.get("qualified_task") != ref:
                if (record["status"] != "failed" or record.get("observation_error")
                        or reference(record["result_path"]) != receipt["result"]
                        or reference(record["task_path"]) != receipt["original_task"]):
                    raise ValueError("repair no longer matches the failed queue attempt")
                record.update(status="pending", reason=None, qualified_task=ref, task_path=ref["path"])
            self.repaired_tasks[tid] = task
            files = task["source_identity"]["source_files"]
            def root_for(ending):
                path, = (Path(name) for name in files if Path(name).parts[-len(ending):] == ending)
                return path
            worker = root_for(("pilotlab","worker3.py")).parents[1]
            foundation = root_for(("src","tide","attention.py")).parents[2]
            runtime = {"source":str(worker), "foundation":str(foundation),
                       "adapter":reference(worker/"pilotlab"/"resume_adapter3.py")}
            for phase_id in (tid, tid.replace("-a2-", "-a3-")):
                self.supervisor.worker_runtimes[phase_id] = runtime
        self.cpu_placement = None
        self.resource_placement = None
        if plan.get("stage_a_cpu_placement", {}).get("policy") == DIRECT_POLICY:
            self.resource_placement = self.supervisor.resource_placement = ResourcePlacement(self)
        elif plan.get("stage_a_cpu_placement"):
            from .cpu_placement3 import CpuRamp, RecipePlacement
            from .dispatch_placement3 import DispatchPlacement, DISPATCH_POLICIES
            if plan["stage_a_cpu_placement"]["policy"] in DISPATCH_POLICIES:
                self.cpu_placement = self.supervisor.dispatch_placement = DispatchPlacement(self)
            else:
                self.cpu_placement = RecipePlacement(self)
            self.supervisor.cpu_ramp = CpuRamp(plan["config"])

    def profile_flow(self, models):
        if not self.plan["config"].get("resource_dispatch"):
            yield from super().profile_flow(models)
            return
        from .io import atomic_json
        missing = [model["id"] for model in models if model["id"] not in self.resources]
        if missing:
            for mid in missing:
                batches = self.plan["stage_a_cpu_placement"]["unprofiled_batches"]
                # A registered larger batch is part of that model's existing
                # plan even if its optional sizing run was still in the queue.
                tid = f"{mid}-preflight-2026112021-capacity-large"
                old = self.migration.get("retired_calibrations", {}).get(tid) or self.migration["completed"].get(tid)
                if old:
                    large = read_bound(old["task"])["batch_size"]
                    if large != 2:
                        batches = sorted([2, large])
                self.resources[mid] = {"batches": list(batches), "placement_policy": DIRECT_POLICY}
            atomic_json(self.output / "model-resources.json", self.resources)

    def recipes(self, model):
        recipes = super().recipes(model)
        if self.cpu_placement:
            from .round3 import TUNING_SEEDS
            return self.cpu_placement.apply(model, recipes, TUNING_SEEDS)
        return recipes

    def verify_outstanding(self, row, task):
        original = read_bound(row["task"])
        placement = getattr(self, "resource_placement", None) or self.cpu_placement
        if placement and placement.reassigned.get(task["id"]) == (row["task"], task["device"]):
            original = {**original, "device":task["device"]}
        verify_recipe(original, task)

    def evidence(self, task):
        row = self.migration["completed"].get(task["id"])
        if row:
            return {"task_path":row["task"]["path"], "result_path":row["result"]["path"]}
        return super().evidence(task)

    def prune(self, tasks, **kwargs):
        super().prune([t for t in tasks if t["id"] not in self.migration["completed"]], **kwargs)

    def retire_selected_parents(self, model):
        if model["id"] in self.fresh_ids:
            super().retire_selected_parents(model, source_identity=self.fresh_identity)
        elif model["id"] not in self.migration["exposed_subject_ids"]:
            super().retire_selected_parents(model)

    def task(self, model, phase, seed, **kwargs):
        task = super().task(model, phase, seed, **kwargs)
        if model["id"] in self.fresh_ids:
            task["id"] = f'{model["id"]}-{DEFAULT_POLICY}' + task["id"][len(model["id"]):]
            task["source_identity"] = self.fresh_identity
        return task

    def review_source_chains(self, report):
        """Independently verify mixed A evidence; the full formal gate stays closed."""
        from .chain_receipt3 import verify_model_chain_receipt
        from .continuation3 import continuation_origin
        from .gate3 import MI, RECIPE_FIELDS, combine_activity, eligible, last_activity, read_evidence, verify_activity_sources
        from .round3 import DIAGNOSTIC_SEEDS, TUNING_SEEDS
        historical = read_bound(self.migration["old_selected"])
        models = {m["id"]:m for m in self.subjects}
        findings = {}
        for reason in report["blocking_reasons"]:
            if reason == "subject_catalog_not_344_unique_effective_models":
                continue
            if not isinstance(reason, dict) or reason.get("model_id") not in self.migration["exposed_subject_ids"]:
                return False
            mid = reason["model_id"]
            choice = self.selected[mid]
            if mid in historical and choice != historical[mid]:
                raise ValueError("archived selection changed: "+mid)
            if mid in historical and choice.get("chain_receipt"):
                old_model = read_bound(choice["chain_receipt"])["model"]
                entry = choice["selection_a1"][0]
                selected_task = read(entry["task_path"])
                frozen = self.migration["completed"][selected_task["id"]]
                if read_bound(frozen["task"]) != selected_task or frozen["result"]["path"] != entry["result_path"]:
                    raise ValueError("original selection lacks its bound source task: "+mid)
                verify_model_chain_receipt(choice["chain_receipt"],model=old_model,choice=choice,
                    source_identity=selected_task["source_identity"],tuning_seeds=TUNING_SEEDS,diagnostic_seeds=DIAGNOSTIC_SEEDS)
                findings[mid] = {"scope":"original source selection","receipt":choice["chain_receipt"]}
                continue
            parents, evidence, activities = {}, [], []
            for field,phase,target in (("diagnostics","init",32768),("selection_a1","a1",MI//4),
                                       ("selection_a2","a2",MI),("confirmation","a3",2*MI)):
                seeds = []
                for entry in choice[field]:
                    task,result,obs = read_evidence(entry,checkpoint=phase!="init")
                    old = self.migration["completed"].get(task["id"])
                    if old:
                        if (read_bound(old["task"]) != task or read_bound(old["result"]) != result
                                or revised_model(task["model"], pool_floor=self.pool_floor) != models[mid]["model"]):
                            raise ValueError("mixed chain old reference mismatch")
                    elif task["source_identity"] != self.identity or task["model"] != models[mid]["model"]:
                        raise ValueError("mixed chain new source or policy mismatch")
                    if (task["phase"] != phase or task["model_id"] != mid or task["target_tokens"] != target
                            or task["init_recipe"] != choice["init_recipe"]
                            or phase != "init" and task["recipe_id"] != choice["recipe_id"]
                            or eligible(task,result,require_resume=phase in ("a2","a3"))):
                        raise ValueError("unqualified mixed-chain task: "+task["id"])
                    seed = task["seed"]
                    seeds.append(seed)
                    if phase in ("a2","a3"):
                        parent,cp = parents[seed]
                        load = continuation_origin(task,result)["resume"]
                        if (any(task.get(k) != parent.get(k) for k in RECIPE_FIELDS if k != "model")
                                or (task["resume_checkpoint"],task["resume_checkpoint_sha256"]) != (cp["path"],cp["sha256"])
                                or (load["path"],load["sha256"],load["from_tokens"]) != (cp["path"],cp["sha256"],parent["target_tokens"])):
                            raise ValueError("mixed-chain rung did not consume its selected parent")
                    if phase != "init":
                        parents[seed] = task,result["checkpoint"]
                    if phase == "a3":
                        activities.append(last_activity(result, task))
                    evidence.append(obs)
                if sorted(seeds) != (DIAGNOSTIC_SEEDS if phase=="init" else TUNING_SEEDS):
                    raise ValueError("incomplete mixed-chain seed group")
            measured = combine_activity(activities)
            verify_activity_sources(choice, measured)
            findings[mid] = {"scope":"source transition recorded, formal transfer review pending","evidence":evidence,
                             "activity":measured}
        freeze_json(self.output/"stage-a-source-chains.json",
                    {"stage_b_gate":"closed","prior_and_mixed_model_evidence":findings})
        return True

    def train_models(self, models):
        # Put the fault and independent fresh capacity probes into the first
        # pipeline window together. Each model's own prescription is intact.
        priority = self.migration.get("priority_model") or self.migration["fault_model"]
        if "continuing_model_ids" in self.migration:
            continuing = set(self.migration["continuing_model_ids"])
            preferred = self.plan.get("stage_a_cpu_placement", {}).get("priority_models", [])
            order = sorted(models, key=lambda m: (m["id"] != priority, m["id"] not in continuing,
                preferred.index(m["id"]) if m["id"] in preferred else len(preferred)))
            return super().train_models(order)
        known = set(self.migration["exposed_subject_ids"])
        resumed = [m for m in models if m["id"] in known and m["id"] != priority and m["id"] not in self.selected]
        fresh = [m for m in models if m["id"] not in known]
        order = [m for m in models if m["id"] == priority]
        for i in range(max(len(resumed), len(fresh))):
            order.extend(resumed[i:i+1])
            order.extend(fresh[i:i+1])
        super().train_models(order)

    def resolve_task(self, task):
        """Resolve each rung after its actual parent is known, before freezing."""
        tid = task["id"]
        if task["model_id"] in getattr(self, "fresh_ids", ()):
            if (task["model"]["config"]["shannon_support_policy"] != DEFAULT_POLICY
                    or task["source_identity"] != self.fresh_identity or task.get("stage_a_migration")
                    or not tid.startswith(f'{task["model_id"]}-{DEFAULT_POLICY}-')):
                raise ValueError("fresh Shannon task crossed recipe or source boundaries")
            self.supervisor.worker_runtimes[tid] = self.fresh_runtime
            self.resource_placement.prepare_task(task)
            return None
        task.pop("stage_a_migration", None)
        task.update(source_identity=self.identity, model=revised_model(task["model"], pool_floor=self.pool_floor))
        repaired = getattr(self, "repaired_tasks", {})
        if tid in repaired:
            verify_recipe(repaired[tid], task)
            task.update(repaired[tid])
            return None
        if task["phase"] == "a3" and (parent_id := tid.replace("-a3-", "-a2-")) in repaired:
            parent = self.supervisor.state["tasks"][parent_id]
            cp = read(parent["result_path"])["checkpoint"]
            if (parent["status"] != "completed" or cp["path"] != task["resume_checkpoint"]
                    or cp["sha256"] != task["resume_checkpoint_sha256"]):
                raise ValueError("A3 did not select the repaired parent's checkpoint")
            task["source_identity"] = repaired[parent_id]["source_identity"]
            verify_recipe(repaired[parent_id], task, parent=True)
            return None
        if getattr(self, "resource_placement", None):
            self.resource_placement.prepare_task(task)
        old_row = self.migration["completed"].get(tid)
        if old_row:
            original, result = read_bound(old_row["task"]), read_bound(old_row["result"])
            verify_recipe(original, task)
            task.clear()
            task.update(original)
            return result
        row = self.migration["outstanding"].get(tid)
        if row:
            self.verify_outstanding(row, task)
            if row.get("receipt"):
                task["stage_a_migration"] = row["receipt"]
                return None
        parent_phase = {"a2":"a1", "a3":"a2"}.get(task["phase"])
        if parent_phase:
            parent_id = tid.replace("-"+task["phase"]+"-", "-"+parent_phase+"-")
            parent = self.migration["completed"].get(parent_id)
            if parent:
                result = read_bound(parent["result"])
                receipt = {"schema":RECEIPT, "mode":"stage_parent", "task_id":tid,
                    "new_source_digest":self.migration["new_source_digest"], "original_task":parent["task"],
                    "result":parent["result"], "checkpoint":result["checkpoint"],
                    "initial_evaluation":{"mode":"rerun","exposure":task["evaluation_start_tokens"]},
                    "evaluation_additional_reservation_tokens":self.plan["config"]["migration_evaluation_reservations"].get(tid,0)}
                from .source_migration3 import migration_inputs
                migration_inputs(task, receipt)
                task["stage_a_migration"] = reference(freeze_json(self.output/"source-migrations"/(tid+".json"), receipt))
        return None

    def training_flow(self, models, **kwargs):
        flow = super().training_flow(models, **kwargs)
        answers = None
        try:
            while True:
                try:
                    wave = flow.send(answers)
                except StopIteration:
                    return
                answers, requests = {}, []
                for request in wave:
                    pending = []
                    for task in request["tasks"]:
                        completed = self.resolve_task(task)
                        if completed is None:
                            pending.append(task)
                        else:
                            answers[task["id"]] = completed
                    if pending:
                        requests.append({**request, "tasks":pending})
                if requests:
                    answers.update((yield requests))
        finally:
            flow.close()
