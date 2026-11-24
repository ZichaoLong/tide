"""Serialize manual and automatic placement against the same resource snapshot."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import time

from .runq_devices import admission, covers, external
from .runq_store import device, event, packed, setting, unit, write
from .scheduler import proc_identity
from .source_migration3 import read_bound


def ram_available():
    return int(next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")).split()[1])*1024


def decoded(row, root):
    job = dict(row)
    job.update(data=estimated_spec(json.loads(job["spec"])), permission=json.loads(job["grant"] or "null"),
               unit=unit(root, "job", job["id"], job["generation"]))
    return job


@lru_cache(maxsize=4096)
def trajectory_peak(owner, task_id, backend):
    """Authenticated high water from this same frozen trajectory before its interruption."""
    path = Path(owner)/"tasks"/task_id/"task.json"
    try:
        task = json.loads(path.read_bytes())
        if task.get("id") != task_id or task.get("device") != ("npu:0" if backend=="npu" else "cpu"):
            return None
        receipt = read_bound(task["stage_a_migration"])
        prior, result = read_bound(receipt["original_task"]), read_bound(receipt["result"])
        if (receipt["mode"] != "signal" or receipt["task_id"] != task_id or result["id"] != task_id
                or result["task_sha256"] != receipt["original_task"]["sha256"]
                or result["device"] != task["device"] or result["source_identity"] != task["source_identity"]
                or any(prior.get(k) != task.get(k) for k in ("id","device","model","batch_size","sequence_length",
                                                              "stream_chunks","precision","cpu_threads"))
                or result["training"]["segment_steps"] < 1):
            return None
        high = result["device_memory"]["cpu_rss_peak_bytes"]
        staging = task.get("checkpoint_reservation_bytes",512*2**20)
        if type(high) is int and high>0 and type(staging) is int and staging>0:
            return {"forecast":(3*high+1)//2+256*2**20+staging,"peak":high,"result":receipt["result"]}
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return None


def estimated_spec(spec):
    if len(spec["allowed"]) == 1 and spec.get("owner") and spec.get("task_id"):
        backend = spec["allowed"][0]
        prior = trajectory_peak(spec["owner"],spec["task_id"],backend)
        if prior:
            peak = min(spec["ram"][backend],prior["forecast"])
            return {**spec,"ram":{backend:peak},"cold_ram":spec["ram"],"prior_host_measurement":prior}
    return spec


@lru_cache(maxsize=1024)
def checkpoint_bytes(path, task_id):
    task = json.loads(Path(path).read_bytes())
    if task["id"] != task_id or task["device"] != "cpu":
        raise ValueError("not this CPU trajectory")
    return int(task.get("checkpoint_reservation_bytes",512*2**20))


def active_peak(job, rss):
    spec = job["data"]
    declared = spec["ram"][job["backend"]]
    # Before an actual optimizer update, a new CPU shape keeps its full
    # cold estimate (accelerator tensors + host RSS). Afterwards use this
    # same process's kernel high-water mark, allowing the existing 1.5x
    # host margin and the task's full checkpoint staging estimate.
    if not (rss and job["backend"] == "cpu" and job.get("first_step") and spec.get("owner") and spec.get("task_id")):
        return max(declared,rss)
    try:
        argv = json.loads(job["command"])["argv"]
        path = argv[argv.index("--task")+1]
        staging = checkpoint_bytes(path,spec["task_id"])
        raw = Path(f"/proc/{job['pid']}/status").read_text().splitlines()
        high = int(next(line for line in raw if line.startswith("VmHWM:")).split()[1])*1024
    except (OSError, ValueError, IndexError, KeyError, StopIteration, TypeError):
        return max(declared,rss)
    cold = spec.get("cold_ram",spec["ram"])[job["backend"]]
    high = max(high,rss,spec.get("prior_host_measurement",{}).get("peak",0))
    forecast = max(cold-spec.get("hbm",0), (3*high+1)//2+256*2**20+staging)
    return max(high, min(cold,forecast))


class Placement:
    def __init__(self, db, root, samples, available, now):
        self.db, self.root, self.samples, self.available, self.now = db, root, samples, available, now
        self.cards = samples[-1]["cards"] if samples else {}
        self.limits = setting(db, "limits", {"cards": 8, "cpu": 40})
        ids = self.limits.get("auto_card_ids")
        self.auto_cards = set(map(int, self.cards)) if ids is None else set(ids)
        self.auto_grants = self.limits.get("auto_share_grants", {})
        self.leases = {r["card"]: dict(r) for r in db.execute("SELECT * FROM leases WHERE state IN ('opening','held','closing')")}
        self.by_card = defaultdict(list)
        self.scoped = defaultdict(int)
        self.cpu_auto = 0
        self.bindings = dict(db.execute("SELECT name,backend FROM bindings"))
        self.cooldown = dict(db.execute("SELECT card,MAX(released) FROM leases WHERE state='missed' GROUP BY card"))
        self.owners = {}
        self.active = {}
        for row in db.execute("SELECT * FROM jobs WHERE state IN ('reserved','starting','running')"):
            job = decoded(row, root)
            self.active[job["id"]] = job
            rss = proc_identity(job["pid"]) if job["pid"] else None
            rss = rss["rss_bytes"] if rss and rss["start_ticks"] == job["ticks"] else 0
            self.charge(job, rss)
        self.parked = set()
        self.opening_host = {}
        for lease in self.leases.values():
            if lease["state"] != "opening":
                continue
            row = db.execute("SELECT spec FROM jobs WHERE id=? AND state='queued'",(lease["trigger_job"],)).fetchone()
            if row:
                self.hold_host(lease["trigger_job"],estimated_spec(json.loads(row["spec"])))

    def hold_host(self, ident, spec):
        peak = spec["ram"]["npu"]
        self.opening_host[ident] = peak
        self.available -= peak
        self.scoped[spec.get("owner")] += peak

    def charge(self, job, rss=0):
        backend = job["backend"]
        peak = active_peak(job,rss)
        self.available -= max(0, peak-rss)
        self.scoped[job["data"].get("owner")] += max(peak, rss)
        if backend == "cpu":
            self.cpu_auto += not job["manual"]
        else:
            self.by_card[job["lease"]].append(job)

    def reason(self, job, message):
        if job["reason"] != message:
            self.db.execute("UPDATE jobs SET reason=? WHERE id=?", (message, job["id"]))
            event(self.db,job["id"],"waiting",message)

    def lease_role(self, lease, role):
        lease["role"] = role
        self.db.execute("UPDATE leases SET role=? WHERE id=?", (role, lease["id"]))

    def regular(self):
        return sum(l["role"] == "auto" and l["state"] != "closing" for l in self.leases.values())

    def maintain(self):
        for lease in self.leases.values():
            if lease["role"] == "auto" and self.limits.get("auto_card_ids") is not None and lease["card"] not in self.auto_cards:
                self.lease_role(lease, "drain")
        for lease in sorted(self.leases.values(), key=lambda x: x["id"]):
            current = self.by_card[lease["id"]]
            card = self.cards.get(str(lease["card"]))
            if lease["role"] == "auto" and self.regular() > self.limits["cards"]:
                self.lease_role(lease, "drain")
            if lease["state"] != "held" or any(j["manual"] for j in current):
                continue
            if not current and lease["role"] == "manual":
                trigger = self.db.execute("SELECT state FROM jobs WHERE id=?",(lease["trigger_job"],)).fetchone()
                if trigger and trigger[0] == "queued":
                    continue
            if lease["role"] == "manual":
                self.lease_role(lease, "drain")
            if (lease["role"] == "drain" and lease["card"] in self.auto_cards and card
                    and self.regular() < self.limits["cards"]):
                residents = external(card, {j["unit"] for j in current})
                if not residents or covers(self.auto_grants.get(str(lease["card"])), residents,
                                          lease["card"], self.samples[-1]["boot"], self.now):
                    self.lease_role(lease, "auto")

    def host_reason(self, spec, backend):
        peak = spec["ram"][backend]
        if peak > self.available:
            return "waiting for available host RAM"
        if spec.get("ram_limit") is not None and self.scoped[spec["owner"]] + peak > spec["ram_limit"]:
            return "waiting for campaign RAM"
        return None

    def reserve(self, job, backend, lease=None):
        binding = job["data"].get("binding")
        if binding:
            self.db.execute("INSERT OR IGNORE INTO bindings VALUES(?,?)", (binding, backend))
            self.bindings[binding] = backend
        job.update(backend=backend, lease=lease, generation=job["generation"]+1)
        job["unit"] = unit(self.root, "job", job["id"], job["generation"])
        self.db.execute("""UPDATE jobs SET state='reserved',backend=?,lease=?,generation=?,assigned=?,reason=NULL
                        WHERE id=? AND state='queued'""", (backend, lease, job["generation"], self.now, job["id"]))
        self.charge(job)
        event(self.db, job["id"], "assigned", backend if not lease else f"lease {lease}")

    def npu(self, job):
        pin = int(job["target"][4:]) if job["manual"] else None
        spec = job["data"]
        if any(l["state"] == "opening" and l["trigger_job"] == job["id"] for l in self.leases.values()):
            return True, "acquiring cooperative device lease"
        cards = [pin] if pin is not None else sorted((int(key) for key in self.cards if int(key) in self.auto_cards), key=self.order_card)
        reason = "waiting for an admissible NPU"
        for card in cards:
            lease = self.leases.get(card)
            if card in self.parked or self.now - self.cooldown.get(card, 0) < 5:
                continue
            if lease and lease["state"] != "held":
                if lease["state"] == "opening" and lease["trigger_job"] == job["id"]:
                    return True, "acquiring cooperative device lease"
                continue
            current = self.by_card[lease["id"]] if lease else []
            primaries = [j for j in current if j["manual"] and j["state"] == "running"]
            grant = job["permission"] if job["manual"] else self.auto_grants.get(str(card))
            grants = [grant]
            if pin is None:
                grants.extend(j["permission"] for j in primaries if j["fill"])
                if not lease and self.regular() >= self.limits["cards"]:
                    continue
                if lease and lease["role"] != "auto" and not primaries:
                    continue
            wait = admission(self.samples, card, current, spec["hbm"], grants,
                             opening=not lease, manual=job["manual"], now=self.now)
            if wait:
                reason = wait
                continue
            if lease:
                self.reserve(job, "npu", lease["id"])
                return True, None
            role = "manual" if pin is not None else "auto"
            lease_grant = grant if pin is not None or self.cards[str(card)]["processes"] else None
            ident = self.db.execute("""INSERT INTO leases(card,role,state,trigger_job,grant,created)
                VALUES(?,?,'opening',?,?,?)""", (card, role, job["id"], packed(lease_grant), self.now)).lastrowid
            self.leases[card] = dict(id=ident, state="opening", role=role, trigger_job=job["id"], card=card)
            self.hold_host(job["id"],spec)
            event(self.db, job["id"], "lease_requested", str(card))
            return True, "acquiring cooperative device lease"
        return False, reason

    def order_card(self, card):
        lease = self.leases.get(card)
        # Spread across physical cards before packing additional workers.
        if lease:
            workers = self.by_card[lease["id"]]
            return (0 if not workers and lease["state"] == "held" else 2, len(workers), card)
        return (1, 0, card)

    def run(self):
        self.maintain()
        queued = [decoded(r, self.root) for r in self.db.execute(
            "SELECT * FROM jobs WHERE state='queued' ORDER BY manual DESC, requested,id")]
        wanted = {j["data"]["binding"]: device(j["target"]) for j in queued if j["manual"] and j["data"].get("binding")}
        empty_held = any(l["state"]=="held" and l["role"]=="auto" and not self.by_card[l["id"]] for l in self.leases.values())
        free_card = any(int(k) in self.auto_cards and int(k) not in self.leases and
                        admission(self.samples, int(k), [], 1, [self.auto_grants.get(k)], opening=True, now=self.now) is None
                        for k in self.cards)
        spreading = empty_held or free_card and self.regular()<self.limits["cards"]
        # Fixed trajectories take their own backend first. Under a tight host
        # ceiling, smaller automatic CPU requests can fill otherwise idle cores.
        def priority(job):
            spec = job["data"]
            if job["manual"]:
                return (0,job["requested"],job["id"])
            if job["id"] in self.opening_host or spreading and spec["allowed"] == ["npu"]:
                return (1,job["requested"],job["id"])
            if spec["allowed"] == ["cpu"]:
                return (2,spec["ram"]["cpu"],job["requested"])
            return (3 if len(spec["allowed"])==2 else 4,spec["ram"].get("cpu",0),job["requested"])

        for job in sorted(queued,key=priority):
            spec = job["data"]
            if job["id"] in self.opening_host:
                self.reason(job,"acquiring cooperative device lease")
                continue
            if spec.get("deadline") and self.now >= spec["deadline"]:
                self.db.execute("UPDATE jobs SET state='cancelled',reason='deadline',finished=? WHERE id=?", (self.now,job["id"]))
                event(self.db, job["id"], "expired")
                continue
            owner = spec.get("owner")
            if owner:
                if owner not in self.owners:
                    self.owners[owner] = setting(self.db, "permit."+owner, 0)
                if self.owners[owner] < self.now:
                    self.reason(job, "waiting for campaign controller")
                    continue
            binding = spec.get("binding")
            fixed = self.bindings.get(binding) or wanted.get(binding)
            allowed = [b for b in spec["allowed"] if (not fixed or b == fixed) and
                       (not job["manual"] or device(job["target"]) == b)]
            if len(allowed) == 2 and self.cpu_auto < self.limits["cpu"] and (not spreading or self.regular() >= self.limits["cards"]):
                allowed.sort(key=lambda b: b != "cpu")
            wait = "waiting for compatible backend"
            for backend in allowed:
                wait = self.host_reason(spec, backend)
                if wait:
                    continue
                if backend == "cpu":
                    if job["manual"] or self.cpu_auto < self.limits["cpu"]:
                        self.reserve(job, "cpu")
                        break
                    wait = "automatic CPU workers full"
                else:
                    placed, wait = self.npu(job)
                    if placed:
                        break
            if wait:
                self.reason(job, wait)
            if job["manual"] and job["target"].startswith("npu:") and wait:
                self.parked.add(int(job["target"][4:]))
        for card, lease in self.leases.items():
            if lease["state"] == "held" and not self.by_card[lease["id"]] and self.now - lease["acquired"] > 8:
                self.db.execute("UPDATE leases SET state='closing' WHERE id=?", (lease["id"],))


def place(db, root, samples):
    available = ram_available()
    with write(db):
        Placement(db, root, samples, available, time.time()).run()
