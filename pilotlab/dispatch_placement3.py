"""Late device binding for unopened A1 groups in the shared Stage A pipeline."""
from collections import defaultdict
import math
import statistics
import time

from .cpu_placement3 import EXPECTED_FOLLOWUP, EXPOSURE, RecipePlacement, WAVE_SIZE, completion_times, frozen_task
from .gate3 import eligible, read_evidence
from .io import atomic_json
from .prepare_resume3 import reference
from .scheduler3 import freeze_json, read
from .source_migration3 import RESOURCE_FIELDS, read_bound

POLICY = "stage-a-live-dispatch-v2"
BACKFILL_POLICY = "stage-a-live-backfill-v3"
DISPATCH_POLICIES = (POLICY, BACKFILL_POLICY)
UNBOUND = "unbound"
DEVICES = ("cpu", "npu:0")


def best_offer(forecasts, *, backfill=False):
    """Give a contested lane to the group with the greatest comparative saving."""
    choices = []
    for mid, sides in forecasts.items():
        offers = [dev for dev in DEVICES if sides[dev]["offer"] is not None]
        if not offers:
            continue
        candidates = offers if backfill else [dev for dev in DEVICES if dev in offers or sides[dev].get("predictable_wait")]
        dev = min(candidates, key=lambda d: (sides[d]["expected_seconds"], d not in offers))
        if dev not in offers:
            continue
        other = next(d for d in DEVICES if d != dev)
        saving = sides[other]["expected_seconds"] - sides[dev]["expected_seconds"]
        choices.append((-saving, sides[dev]["expected_seconds"], mid, dev))
    if choices:
        return min(choices)[2:]


class DispatchPlacement(RecipePlacement):
    policy = POLICY

    def __init__(self, campaign):
        super().__init__(campaign)
        plan = campaign.plan["stage_a_cpu_placement"]
        self.policy = plan.get("policy", POLICY)
        self.calibration_models = set(plan["calibration_models"])
        self.groups = {}
        self.sides = {}
        self.probes = {}
        self.probes_registered = set()
        self.carried_probes = {self.previous["tasks"][tid]["model_id"] for tid in self.outstanding
                               if self.previous["tasks"][tid]["stage"] == "dispatch-calibration"}
        self.claimed_intents = {}
        self.samples = {}
        self.live_rates = {}

    def spec(self, model, dev):
        recipe = dict(batch_size=2, device=dev, eval_tokens=256, eval_interval_tokens=0, purpose="capacity_calibration")
        if dev == "cpu":
            recipe["closing_reserve_seconds"] = 60
        task = self.campaign.task(model, "preflight", 2026112021, recipe=recipe,
            target=2*self.campaign.protocol["sequence_length"]*16, suffix="-dispatch-"+dev.split(":")[0])
        previous = self.campaign.migration["completed"].get(task["id"])
        if previous and frozen_task(previous["task"]["path"])["source_identity"] != task["source_identity"]:
            # Keep the completed probe under its original source. The current
            # worker needs its own timing receipt before dispatch can use it.
            task["id"] += "-" + self.campaign.migration["new_source_digest"][:16]
        return task

    def measurement(self, expected, entry):
        task, result, receipt = read_evidence(entry)
        if any(entry.get(k, v) != v for k, v in receipt.items()):
            raise ValueError("dispatch calibration receipt changed")
        ignored = RESOURCE_FIELDS | {"id", "closing_reserve_seconds"}
        if {k:v for k,v in task.items() if k not in ignored} != {k:v for k,v in expected.items() if k not in ignored}:
            return None
        train = result["training"]
        if (eligible(task, result) or train["segment_tokens"] != expected["target_tokens"]
                or train["segment_steps"] != 16 or not math.isfinite(train["tokens_per_second"])
                or train["tokens_per_second"] <= 0):
            return None
        return {"rate": train["tokens_per_second"], "overhead": max(0., result["wall_seconds"]-train["seconds"]),
                "evidence": receipt}

    def calibrations(self, model):
        mid = model["id"]
        if mid not in self.sides:
            profile = self.campaign.resources[mid]
            sides = {}
            for dev in DEVICES:
                entries = ([profile.get("cpu_measurement")] if dev == "cpu" else profile["measurements"])
                extra = profile.get("dispatch_measurements", {}).get(dev)
                for entry in [extra, *entries]:
                    if entry and (sample := self.measurement(self.spec(model, dev), entry)):
                        sides[dev] = sample
                        break
            self.sides[mid] = sides
        return self.sides[mid]

    def accept_probe(self, task, entry):
        sample = self.measurement(task, entry)
        if sample is None:
            raise ValueError("unqualified dispatch calibration: "+task["id"])
        mid, dev = task["model_id"], task["device"]
        self.sides[mid][dev] = sample
        self.campaign.resources[mid].setdefault("dispatch_measurements", {})[dev] = sample["evidence"]
        atomic_json(self.campaign.output/"model-resources.json", self.campaign.resources)

    def supplemental_requests(self, verified):
        campaign = self.campaign
        # Measure a missing side only when an unopened A1 group can actually use
        # it. Previously submitted probes also survive a controller cutover.
        required = self.calibration_models
        if self.policy == BACKFILL_POLICY:
            required = required.intersection(self.groups) | self.carried_probes
        needed = required - self.probes_registered
        new = []
        if needed:
            for model in campaign.subjects:
                if model["id"] not in needed:
                    continue
                sides = self.calibrations(model)
                for dev in DEVICES:
                    task = self.spec(model, dev)
                    if dev in sides and task["id"] not in self.outstanding:
                        continue
                    completed = campaign.resolve_task(task)
                    if completed is None:
                        self.probes[task["id"]] = task
                        new.append(task)
                    else:
                        self.accept_probe(task, campaign.evidence(task))
                self.probes_registered.add(model["id"])
        for tid in self.probes.keys() & verified.keys():
            task = self.probes.pop(tid)
            self.accept_probe(task, campaign.evidence(task))
            campaign.prune([task], reason="verified dispatch calibration; no training checkpoint consumer")
        return [campaign.request("dispatch-calibration", new)] if new else []

    def decision_path(self, mid):
        return self.campaign.output/"decisions"/(mid+"-a1-batch2-dispatch.json")

    def apply(self, model, recipes, seeds):
        mid = model["id"]
        group = [r for r in recipes if r["batch_size"] == 2]
        ids = sorted(f'{mid}-a1-{seed}-{r["recipe_id"]}' for r in group for seed in seeds)
        if not group:
            return recipes
        if len(set(ids)) != WAVE_SIZE:
            raise ValueError("dispatch requires the full six-member A1 group")
        resource = self.campaign.resources[mid]
        local = self.decision_path(mid)
        saved = resource.get("a1_placements", {}).get("2")
        prior = read_bound(saved) if saved else None
        # A sealed v2 claim is sticky across controller generations. A v1
        # preassignment may be released only while all its members are untouched.
        decision = read(local) if local.exists() else prior
        if decision and decision["policy"] in DISPATCH_POLICIES:
            if (decision["model_id"] != mid or decision["task_ids"] != ids or decision["batch_size"] != 2
                    or decision["device"] not in DEVICES):
                raise ValueError("dispatch receipt differs from the A1 group")
            dev = decision["device"]
            if local.exists():
                self.claimed_intents.update(decision["intents"])
                self.publish_device(mid, local, dev)
        else:
            state = self.campaign.supervisor.state.get("tasks", {})
            if any(tid in state and (state[tid].get("attempts") or state[tid]["status"] != "pending") for tid in ids):
                return recipes
            # Re-registering unbound intents after a restart is safe. No prior
            # attempt, receipt, checkpoint or completion can be reassigned.
            for tid in ids:
                old = self.outstanding.get(tid)
                if tid in self.campaign.migration["completed"]:
                    return recipes
                if old:
                    record = self.previous["tasks"][tid]
                    if (old["disposition"] != "not_started" or old.get("receipt") or record.get("attempts")
                            or record.get("heartbeat") or record["status"] != "pending"
                            or frozen_task(old["task"]["path"]).get("resume_checkpoint")):
                        return recipes
            if len(self.calibrations(model)) < 2 and mid not in self.calibration_models:
                return recipes
            dev = UNBOUND
            self.groups[mid] = ids
        for recipe in group:
            recipe["device"] = dev
        for tid in ids:
            old = self.outstanding.get(tid)
            if old and old["disposition"] == "not_started" and not old.get("receipt"):
                if dev == UNBOUND or decision["prior_requests"].get(tid) == old["task"]:
                    self.reassigned[tid] = old["task"], dev
        return recipes

    def publish_device(self, mid, path, dev):
        resource = self.campaign.resources[mid]
        ref = reference(path)
        if resource.get("a1_placements", {}).get("2") != ref or resource["device_by_batch"]["2"] != dev:
            resource.setdefault("a1_placements", {})["2"] = ref
            resource["device_by_batch"]["2"] = dev
            atomic_json(self.campaign.output/"model-resources.json", self.campaign.resources)

    def register(self, task, record):
        mid, tid = task["model_id"], task["id"]
        directory = self.campaign.output/"tasks"/tid
        if task["device"] == UNBOUND:
            if tid not in self.groups.get(mid, ()) or record.get("attempts"):
                raise ValueError("undeclared or already started dispatch intent")
            path = freeze_json(directory/"intent.json", task)
        else:
            if tid in self.claimed_intents:
                intent = read_bound(self.claimed_intents[tid])
                if intent["device"] != UNBOUND or {**intent, "device":task["device"]} != task:
                    raise ValueError("concrete task differs from its sealed intent")
            path = freeze_json(directory/"task.json", task)
        record.update(task_path=str(path), device=task["device"])

    def observe_rates(self, records, now):
        recent = defaultdict(list)
        active = {}
        for row in records:
            hb = row.get("heartbeat") or {}
            if hb.get("status") != "training" or now-hb["updated_unix"] > 45:
                continue
            pid = hb["pid"]
            stamp, tokens = hb["updated_unix"], hb["tokens"]
            key = (hb["task_id"], pid)
            before, start, rate = self.samples.get(key, (stamp, tokens, None))
            if stamp-before >= 15 and tokens > start:
                rate = (tokens-start)/(stamp-before)
                before, start = stamp, tokens
            active[key] = before, start, rate
            if rate and row.get("task_path"):
                task = frozen_task(row["task_path"])
                recent[task["model_id"], task["device"], task["batch_size"]].append(rate)
        self.samples = active
        self.live_rates = {key:statistics.median(v) for key,v in recent.items()}

    def dispatch_rank(self, tasks, cpu_limit, offers):
        by_mid = {}
        for task in tasks:
            by_mid.setdefault(task["model_id"], task)
        state = self.campaign.supervisor.state
        records = state["tasks"]
        now = time.time()
        self.collect_costs(records)
        active, pending = defaultdict(list), defaultdict(list)
        for row in records.values():
            dev = row["device"]
            if row["status"] not in ("pending", "running") or dev not in DEVICES or not row.get("task_path"):
                continue
            t = frozen_task(row["task_path"])
            key = (t["model_id"], dev, t["batch_size"])
            rate = self.live_rates.get(key) or self.rate(*key)
            if not rate:
                continue
            resume_tokens = (frozen_task(t["stage_a_migration"]["path"])["checkpoint"]["tokens"]
                             if t.get("stage_a_migration") else 0)
            start = max((row.get("heartbeat") or {}).get("tokens", 0), t.get("evaluation_start_tokens", 0), resume_tokens)
            costs = active if row["status"] == "running" else pending
            costs[dev].append(max(0, t["target_tokens"]-start)/rate + self.overhead(dev,t["phase"],t["batch_size"]))
        forecasts = {}
        for mid, task in by_mid.items():
            sides = self.sides[mid]
            if len(sides) != 2:
                continue
            capacity = offers(task)
            forecasts[mid] = {}
            for dev in DEVICES:
                cap = capacity[dev]
                lanes = max(1, cpu_limit if dev == "cpu" else cap["lanes"])
                key = mid, dev, 2
                trained = self.measured.get(key)
                rate = self.live_rates.get(key) or (statistics.median(trained) if trained else sides[dev]["rate"])
                overhead = sides[dev]["overhead"]
                duration = EXPOSURE["a1"]/rate + self.overhead(dev, "a1", 2, overhead)
                finish = completion_times(lanes, active[dev], pending[dev], WAVE_SIZE, duration)
                followup = EXPECTED_FOLLOWUP/rate + (2*self.overhead(dev,"a2",2,overhead)+self.overhead(dev,"a3",2,overhead))/6
                cap["predictable_wait"] = bool(cap.get("predictable_wait") and (active[dev] or pending[dev]))
                forecasts[mid][dev] = dict(**cap, active=len(active[dev]), pending=len(pending[dev]),
                    rate_basis="live" if key in self.live_rates else "completed" if trained else "calibration",
                    tokens_per_second=rate, group_finish_seconds=finish, expected_followup_seconds=followup,
                    expected_seconds=finish+followup)
        selected = best_offer(forecasts, backfill=self.policy == BACKFILL_POLICY)
        state["dispatch_queue"] = dict(policy=self.policy, observed_unix=now, unbound_groups=len(by_mid),
            awaiting_calibration=sorted(set(by_mid)-set(forecasts)), forecasts=forecasts)
        if selected:
            mid, dev = selected
            return by_mid[mid], dev, forecasts[mid]

    def claim(self, first, dev, forecasts, tasks):
        mid = first["model_id"]
        ids = self.groups[mid]
        members = {t["id"]:t for t in tasks if t["id"] in ids}
        rows = self.campaign.supervisor.state["tasks"]
        if (set(members) != set(ids) or not forecasts[dev]["offer"]
                or any(t["device"] != UNBOUND or rows[tid]["status"] != "pending" or rows[tid]["attempts"]
                       for tid,t in members.items())):
            raise ValueError("dispatch requires six untouched registered intents and a live offer")
        intents = {tid:reference(rows[tid]["task_path"]) for tid in ids}
        if any(read_bound(intents[tid]) != members[tid] for tid in ids):
            raise ValueError("dispatch intent differs from its registered bytes")
        decision = dict(policy=self.policy, model_id=mid, batch_size=2, task_ids=ids, device=dev,
            first_task_id=first["id"], claimed_unix=time.time(), forecasts=forecasts, intents=intents,
            calibrations={d:self.sides[mid][d]["evidence"] for d in DEVICES},
            prior_requests={tid:self.outstanding[tid]["task"] for tid in ids if tid in self.outstanding})
        path = freeze_json(self.decision_path(mid), decision)
        self.claimed_intents.update(intents)
        for tid in ids:
            members[tid]["device"] = dev
            self.register(members[tid], rows[tid])
        self.publish_device(mid, path, dev)
        del self.groups[mid]
