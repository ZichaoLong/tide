"""Run only the adjudicated early Stage A chains in the existing project queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time

from .hbm_profile3 import frozen_profile, hbm_shape
from .io import atomic_json, sha256
from .prepare_resume3 import reference
from .round3 import Campaign
from .runq_campaign import QueueSupervisor, device_cost
from .runq_store import lookup
from .scheduler3 import Blocked, freeze_json, read
from .source_migration3 import read_bound


def gate_observation(db, gate):
    row = lookup(db, gate["job"])
    spec = json.loads(row["spec"])
    if (row["job_key"] != gate["job_key"] or spec["owner"] != gate["owner"]
            or spec["task_id"] != gate["task_id"] or spec["allowed"] != ["npu"]):
        raise Blocked("NPU predecessor identity changed")
    if row["state"] not in ("running", "completed") or not row["started"]:
        return None
    if (row["backend"] != "npu" or row["generation"] < gate["minimum_generation"]
            or not row["pid"] or not row["ticks"] or not row["lease"]):
        raise Blocked("NPU predecessor lacks a bound worker")
    card = db.execute("SELECT card FROM leases WHERE id=?", (row["lease"],)).fetchone()
    if not card:
        raise Blocked("NPU predecessor lease missing")
    return dict(predecessor=gate, observed_unix=time.time(), state=row["state"], physical_card=card[0],
                **{k: row[k] for k in ("started", "generation", "pid", "ticks", "lease")})


class DeferredSupervisor(QueueSupervisor):
    def device_seconds(self):
        return device_cost(self.db, self.owner, time.time(), alongside=self.config["predecessor"]["owner"])

    def preparation_device_seconds(self):
        return device_cost(self.db, self.owner, time.time(), preparation=True,
                           alongside=self.config["predecessor"]["owner"])

    def start_ready_tasks(self, tasks, slots):
        gate = self.config["predecessor"]
        receipt = self.state.get("npu_release")
        if receipt is None:
            path = self.output / "npu-release.json"
            stored = read(path)
            if stored is None:
                stored = gate_observation(self.db, gate)
                if stored is not None:
                    freeze_json(path, stored)
            if stored is not None:
                if stored["predecessor"] != gate:
                    raise Blocked("NPU release belongs to a different predecessor")
                receipt = self.state["npu_release"] = reference(path)
        if not receipt:
            cpu = []
            for task in tasks:
                if task["device"] == "cpu":
                    cpu.append(task)
                elif self.state["tasks"][task["id"]]["status"] == "pending":
                    self.state["tasks"][task["id"]]["queue_wait"] = f'waiting for primary NPU job {gate["job"]} to start'
            tasks = cpu
        super().start_ready_tasks(tasks, slots)


class DeferredCampaign(Campaign):
    def __init__(self, plan, output):
        self.review = read_bound(plan["review"])
        self.rows = self.review["subjects"]
        self.identities = {}
        self.runtimes = {}
        for name, src in self.review["sources"].items():
            self.identities[name] = read_bound(src["identity"])
            self.runtimes[name] = {k: src[k] for k in ("source", "foundation", "adapter")}
        super().__init__(plan, output, queue_supervisor=DeferredSupervisor)
        initial = read_bound(plan["initial_resources"])
        resource_path = self.output / "model-resources.json"
        if not resource_path.exists():
            atomic_json(resource_path, initial)
            self.resources = initial
        if any(self.resources.get(mid) != resource for mid, resource in initial.items()):
            raise Blocked("historical capacity table differs from the frozen input")
        self.supervisor.worker_runtimes = {}
        for identity in self.identities.values():
            self.supervisor.inherited_hbm.update(frozen_profile(plan["hbm_profile"], identity))
        self.floors = read_bound(plan["inherited_hbm_floors"])
        if self.supervisor.state.get("npu_release"):
            release = read_bound(self.supervisor.state["npu_release"])
            if release["predecessor"] != self.plan["config"]["predecessor"]:
                raise Blocked("restored NPU release does not match this plan")

    def recipes(self, model):
        included = self.rows[model["id"]]["recipes"]
        return [r for r in super().recipes(model) if r["recipe_id"] in included]

    def task(self, model, phase, seed, **kwargs):
        task = super().task(model, phase, seed, **kwargs)
        row = self.rows[model["id"]]
        name = row["target_arm"]
        task["source_identity"] = self.identities[name]
        expected = {**model["model"]["config"], **row["model_policy"]}
        task["model"] = {**model["model"], "config": expected}
        self.supervisor.worker_runtimes[task["id"]] = self.runtimes[name]
        if phase == "a1":
            for child in ("a2", "a3"):
                self.supervisor.worker_runtimes[task["id"].replace("-a1-", f"-{child}-")] = self.runtimes[name]
        size = row["batch_requests"].get(str(task["batch_size"]), {})
        if task["device"] == "npu:0":
            # Unmeasured shapes use the declared probe envelope until full-stage observations exist.
            fallback = row["unmeasured_npu_request"] if not size.get("peak") or (
                phase == "preflight" and model["id"] not in self.resources) else 0
            task["reserved_hbm_bytes"] = max(task.get("reserved_hbm_bytes", 0), size.get("hbm", 0),
                                             self.floors.get(hbm_shape(task), 0), fallback)
        if size.get("rss"):
            task["reserved_rss_bytes"] = max(task.get("reserved_rss_bytes", 0), size["rss"])
        return task

    def run_followup(self):
        models = [m for m in self.subjects if self.rows[m["id"]]["mode"] != "carry_recipe"]
        ids = {m["id"] for m in models}

        def flow(group):
            entry = self.rows[group[0]["id"]]
            return self.training_flow(group, per_model_decisions=True,
                                      overlap_initialization=bool(entry["historical_init_qualified"]),
                                      a2_recipe_count=min(2, len(entry["recipes"])))

        self.run_model_flows(models, flow, stage="early-stage-a-followup")
        if set(self.selected) != ids:
            raise Blocked("early follow-up lacks a selection for a scheduled subject")
        choices = {mid: {"scope": self.rows[mid]["mode"], "recipe": value["recipe_id"]}
                   for mid, value in self.selected.items()}
        freeze_json(self.output / "report.json", dict(status="completed", stage_b_gate="closed",
                    review=self.plan["review"], selections=choices, formal_trajectories=0,
                    carried_recipes={mid: r["recipes"][0] for mid, r in self.rows.items() if r["mode"] == "carry_recipe"}))
        self.supervisor.finish("completed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    plan = read_bound({"path": str(args.plan), "sha256": args.sha256})
    if (Path(__file__).resolve().parents[1] != Path(plan["controller_source"])
            or any(sha256(p) != h for p, h in plan["controller_files"].items())):
        raise ValueError("controller differs from frozen Stage A follow-up")
    output = Path(plan["output"])
    campaign = DeferredCampaign(plan, output)
    signal.signal(signal.SIGTERM, campaign.supervisor.request_stop)
    signal.signal(signal.SIGINT, campaign.supervisor.request_stop)
    try:
        campaign.run_followup()
    except BaseException as error:
        campaign.supervisor.stop_reason = campaign.supervisor.stop_reason or str(error)
        campaign.supervisor.stop_all()
        atomic_json(output / "blocked.json", dict(status=campaign.supervisor.state["status"],
                    reason=campaign.supervisor.stop_reason, stage_b_gate="closed"))
        raise
    finally:
        campaign.supervisor.close()


if __name__ == "__main__":
    main()
