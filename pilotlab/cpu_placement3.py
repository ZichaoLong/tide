"""Choose the device for a whole unstarted A1 batch group using measured turnaround."""
from collections import Counter, defaultdict
from functools import lru_cache
import heapq
import math
from pathlib import Path
import statistics

from .gate3 import MI, eligible, read_evidence
from .io import atomic_json
from .prepare_resume3 import reference
from .scheduler3 import freeze_json, read
from .source_migration3 import read_bound

POLICY = "stage-a-measured-turnaround-v1"
WAVE_SIZE = 6
WINDOW_SECONDS = 45
EXPOSURE = {"a1": MI // 4, "a2": 3 * MI // 4, "a3": MI}
# Two of six recipes enter A2, one of six enters A3. Both seeds advance together.
EXPECTED_FOLLOWUP = (2 * EXPOSURE["a2"] + EXPOSURE["a3"]) / 6


@lru_cache(maxsize=4096)
def frozen_task(path):
    return read(path)


def completion_times(lanes, active, pending, group_size, duration):
    clocks = [0.] * max(1, lanes)
    for seconds in [*sorted(active, reverse=True), *pending]:
        heapq.heapreplace(clocks, clocks[0] + seconds)
    finishes = []
    for _ in range(group_size):
        finish = clocks[0] + duration
        heapq.heapreplace(clocks, finish)
        finishes.append(finish)
    return max(finishes)


class RecipePlacement:
    def __init__(self, campaign):
        self.campaign = campaign
        batch = campaign.migration
        self.previous = read_bound(batch["old_scheduler"])
        self.old_order = {tid:i for i, tid in enumerate(self.previous["tasks"])}
        self.outstanding = batch["outstanding"]
        self.reassigned = {}
        self.proposed = {}
        self.costs = defaultdict(list)
        self.measured = defaultdict(list)
        self.observed = set()
        self.collect_costs(self.previous["tasks"])

    def collect_costs(self, records):
        for tid, row in records.items():
            if tid in self.observed or row["status"] != "completed" or not row.get("result_path"):
                continue
            self.observed.add(tid)
            if row["stage"] not in EXPOSURE:
                continue
            result = read(row["result_path"])
            train = result["training"]
            if not train.get("seconds") or not train.get("segment_tokens"):
                continue
            task = frozen_task(row["task_path"])
            device = task["device"]
            key = (device, task["phase"], task["batch_size"])
            wall = max(result["wall_seconds"], row["attempts"][-1].get("worker_wall_seconds") or 0)
            self.costs[key].append(max(0., wall - train["seconds"]))
            if task["source_identity"] == self.campaign.identity:
                self.measured[task["model_id"], device, task["batch_size"]].append(train["tokens_per_second"])

    def overhead(self, device, phase, batch, fallback=0.):
        samples = self.costs.get((device, phase, batch))
        if not samples:
            samples = [x for (d, p, _), values in self.costs.items() if (d, p) == (device, phase) for x in values]
        return statistics.median(samples) if samples else fallback

    def rate(self, model_id, device, batch, default=None):
        samples = self.measured.get((model_id, device, batch))
        if samples:
            return statistics.median(samples)
        profile = self.campaign.resources.get(model_id, {})
        if device == "cpu" and batch == 2:
            return profile.get("cpu_tokens_per_second") or default
        if device != "cpu":
            return profile.get("tokens_per_second_by_batch", {}).get(str(batch)) or default
        return default

    def queue(self, excluded):
        state = self.campaign.supervisor.state.get("tasks", {})
        self.collect_costs(state)
        active, pending = defaultdict(list), defaultdict(list)
        known = dict(self.proposed)
        position = min((self.old_order[tid] for tid in excluded if tid in self.old_order), default=len(self.old_order))
        for tid, item in self.outstanding.items():
            if tid not in state and tid not in known and tid not in excluded:
                if self.old_order[tid] > position and item["disposition"] == "not_started":
                    continue
                task = frozen_task(item["task"]["path"])
                start = read_bound(item["receipt"])["checkpoint"]["tokens"] if item.get("receipt") else task.get("evaluation_start_tokens", 0)
                known[tid] = task, start
        for tid, row in state.items():
            known.pop(tid, None)
            if tid in excluded or row["status"] not in ("running", "pending"):
                continue
            task = frozen_task(row.get("task_path") or str(self.campaign.output / "tasks" / tid / "task.json"))
            hb = row.get("heartbeat") or {}
            start = max(hb.get("tokens") or 0, task.get("evaluation_start_tokens", 0))
            known[tid] = task, start
        for tid, (task, start) in known.items():
            if tid in excluded:
                continue
            device, phase, batch = task["device"], task["phase"], task["batch_size"]
            rate = self.rate(task["model_id"], device, batch)
            if not rate:
                continue  # Unmeasured queued work cannot justify a speed advantage.
            duration = max(0, task["target_tokens"] - start) / rate + self.overhead(device, phase, batch)
            row = state.get(tid, {})
            target = active if row.get("status") == "running" else pending
            target[device].append(duration)
        return active, pending

    def paired_rates(self, model, batch, resource):
        cpu = resource.get("cpu_measurement")
        if not cpu:
            return None
        ct, cr, _ = read_evidence(cpu)
        if (ct["batch_size"] != batch or ct["model_id"] != model["id"]
                or ct["source_identity"] != self.campaign.identity or ct["model"] != model["model"]):
            return None
        for npu in resource["measurements"]:
            nt, nr, _ = read_evidence(npu)
            if nt["batch_size"] != batch:
                continue
            ignored = {"id", "device", "closing_reserve_seconds"}
            if (ct["device"] != "cpu" or nt["device"] != "npu:0"
                    or {k:v for k,v in ct.items() if k not in ignored} != {k:v for k,v in nt.items() if k not in ignored}
                    or eligible(ct, cr) or eligible(nt, nr)):
                return None
            return {t["device"]: (r["training"]["tokens_per_second"], max(0., r["wall_seconds"]-r["training"]["seconds"]))
                    for t, r in ((ct, cr), (nt, nr))}
        return None

    def movable(self, ids):
        for tid in ids:
            if tid in self.campaign.migration["completed"] or tid in self.campaign.supervisor.state.get("tasks", {}):
                return False
            item = self.outstanding.get(tid)
            if item:
                original = frozen_task(item["task"]["path"])
                record = self.previous["tasks"][tid]
                if (item["disposition"] != "not_started" or item.get("receipt") or original.get("resume_checkpoint")
                        or record.get("attempts") or record.get("heartbeat") or record["status"] != "pending"):
                    return False
        return True

    def choose(self, model, batch, tasks, paired):
        campaign = self.campaign
        cfg = campaign.plan["config"]
        ids = {t["id"] for t in tasks}
        active, pending = self.queue(ids)
        state = campaign.supervisor.state
        lanes = {"cpu": min(cfg["max_cpu_workers"], cfg["max_cpu_task_threads"] // tasks[0]["cpu_threads"]),
                 "npu:0": (state.get("actual_npu_worker_concurrency") or
                           self.previous.get("peak_npu_worker_concurrency") or cfg["target_npu_devices"])}
        admission = state.get("cpu_admission", {})
        if admission.get("affinity_idle_cores") is not None:
            usable = admission["cpu_workers"] + int(admission["affinity_idle_cores"] // tasks[0]["cpu_threads"])
            lanes["cpu"] = max(1, min(lanes["cpu"], usable))
        forecasts = {}
        for device, (calibrated_rate, probe_overhead) in paired.items():
            rate = self.rate(model["id"], device, batch, calibrated_rate)
            overhead = self.overhead(device, "a1", batch, probe_overhead)
            duration = EXPOSURE["a1"] / rate + overhead
            finish = completion_times(lanes[device], active[device], pending[device], len(tasks), duration)
            if device == "cpu":
                needed = min(lanes[device], len(active[device])+len(pending[device])+len(tasks))
                waves = math.ceil(max(0, needed - admission.get("limit", WAVE_SIZE)) / WAVE_SIZE)
                finish = max(finish, duration + waves * WINDOW_SECONDS)
            followup = EXPECTED_FOLLOWUP / rate + (
                2 * self.overhead(device, "a2", batch, probe_overhead) + self.overhead(device, "a3", batch, probe_overhead)) / 6
            forecasts[device] = dict(lanes=lanes[device], pending=len(pending[device]), active=len(active[device]),
                tokens_per_second=rate, a1_wall_seconds=duration, group_finish_seconds=finish,
                expected_followup_seconds=followup, expected_seconds=finish+followup)
        device = min(forecasts, key=lambda d: forecasts[d]["expected_seconds"])
        return dict(policy=POLICY, model_id=model["id"], batch_size=batch, device=device, forecasts=forecasts,
                    task_ids=sorted(ids), prior_requests={k:v["task"] for k in ids if (v:=self.outstanding.get(k))})

    def apply(self, model, recipes, seeds):
        resource = self.campaign.resources[model["id"]]
        for batch in resource["batches"]:
            group = [r for r in recipes if r["batch_size"] == batch]
            tasks = [self.campaign.task(model, "a1", seed, recipe=r, target=EXPOSURE["a1"], suffix="-"+r["recipe_id"])
                     for r in group for seed in seeds]
            ids = {t["id"] for t in tasks}
            saved = resource.get("a1_placements", {}).get(str(batch))
            path = self.campaign.output / "decisions" / f'{model["id"]}-a1-batch{batch}-device.json'
            if saved:
                decision = read_bound(saved)
            elif path.exists():
                decision = read(path)
            else:
                if not self.movable(ids):
                    continue
                paired = self.paired_rates(model, batch, resource)
                if not paired:
                    continue
                decision = self.choose(model, batch, tasks, paired)
                freeze_json(path, decision)
            device = decision["device"]
            if (decision["policy"] != POLICY or set(decision["task_ids"]) != ids or decision["batch_size"] != batch
                    or decision["model_id"] != model["id"] or device not in ("cpu", "npu:0")):
                raise ValueError("invalid frozen A1 device decision")
            if not saved:
                resource.setdefault("a1_placements", {})[str(batch)] = reference(path)
                resource["device_by_batch"][str(batch)] = device
                atomic_json(self.campaign.output/"model-resources.json", self.campaign.resources)
            for recipe in group:
                recipe["device"] = device
            for task in tasks:
                tid = task["id"]
                prior = self.outstanding.get(tid)
                # A receipt authorizes exactly this previously untouched input. Later
                # continuations already carry the new device in their frozen task.
                if prior and decision["prior_requests"].get(tid) == prior["task"] and prior["disposition"] == "not_started" and not prior.get("receipt"):
                    self.reassigned[tid] = (prior["task"], device)
                if tid not in self.campaign.migration["completed"]:
                    self.proposed[tid] = ({**task, "device": device}, 0)
        return recipes


def host_cpu_ticks(affinity):
    total = idle = 0
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if not fields or not fields[0][3:].isdigit() or int(fields[0][3:]) not in affinity:
            continue
        ticks = list(map(int, fields[1:9]))
        total += sum(ticks)
        idle += ticks[3]
    return total, idle


class CpuRamp:
    """Admit one six-task wave at a time and compare continuing NPU workers."""
    def __init__(self, config):
        self.affinity = set(config["host_cpu_affinity"])
        self.maximum = config["max_cpu_workers"]
        self.limit = min(WAVE_SIZE, self.maximum)
        self.sample = None
        self.rates = {}
        self.previous_count = 0

    def admission(self, records, cpu_count, now, state):
        if self.sample and now - self.sample[0] < WINDOW_SECONDS:
            return self.limit
        moments = {}
        groups = Counter(r.get("shared_group") for r in records if r.get("device") != "cpu")
        for row in records:
            hb = row.get("heartbeat") or {}
            if row["device"] != "cpu" and hb.get("status") == "training" and now - hb["updated_unix"] < 30:
                key = (hb["task_id"], hb["pid"], row.get("shared_group"), groups[row.get("shared_group")])
                moments[key] = (hb["updated_unix"], hb["tokens"])
        ticks = host_cpu_ticks(self.affinity)
        idle = ratio = None
        comparisons = 0
        if self.sample:
            _, old_ticks, old_moments = self.sample
            elapsed = ticks[0] - old_ticks[0]
            if elapsed > 0:
                idle = len(self.affinity) * (ticks[1] - old_ticks[1]) / elapsed
            rates = {}
            for key, (stamp, tokens) in moments.items():
                if key in old_moments:
                    before, start = old_moments[key]
                    if stamp > before and tokens > start:
                        rates[key] = (tokens-start)/(stamp-before)
            common = [rate/self.rates[key] for key, rate in rates.items() if key in self.rates]
            comparisons = len(common)
            ratio = statistics.median(common) if comparisons >= 3 else None
            self.rates = rates
            if idle is not None and (idle < 4 or ratio is not None and ratio < .85 and cpu_count > self.previous_count):
                self.limit = max(1, min(self.limit, cpu_count) - WAVE_SIZE)
            elif idle is not None and idle >= 4 and cpu_count >= self.limit:
                self.limit = min(self.maximum, self.limit + min(WAVE_SIZE, int(idle // 4)))
        self.sample = now, ticks, moments
        self.previous_count = cpu_count
        state["cpu_admission"] = dict(policy=POLICY, observed_unix=now, cpu_workers=cpu_count, limit=self.limit,
            affinity_idle_cores=idle, comparable_npu_workers=comparisons, median_npu_rate_ratio=ratio)
        return self.limit
