"""Serialize manual and automatic placement against the same resource snapshot."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

from .runq_devices import admission, external
from .runq_store import device, event, packed, setting, unit, write
from .scheduler import proc_identity


def ram_available():
    return int(next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")).split()[1])*1024


def decoded(row, root):
    job = dict(row)
    job.update(data=json.loads(job["spec"]), permission=json.loads(job["grant"] or "null"),
               unit=unit(root, "job", job["id"], job["generation"]))
    return job


class Placement:
    def __init__(self, db, root, samples, available, now):
        self.db, self.root, self.samples, self.available, self.now = db, root, samples, available, now
        self.cards = samples[-1]["cards"] if samples else {}
        self.limits = setting(db, "limits", {"cards": 8, "cpu": 40})
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

    def charge(self, job, rss=0):
        backend = job["backend"]
        peak = job["data"]["ram"][backend]
        self.available -= max(0, peak-rss)
        self.scoped[job["data"].get("owner")] += max(peak, rss)
        if backend == "cpu":
            self.cpu_auto += not job["manual"]
        else:
            self.by_card[job["lease"]].append(job)

    def reason(self, job, message):
        if job["reason"] != message:
            self.db.execute("UPDATE jobs SET reason=? WHERE id=?", (message, job["id"]))

    def lease_role(self, lease, role):
        lease["role"] = role
        self.db.execute("UPDATE leases SET role=? WHERE id=?", (role, lease["id"]))

    def regular(self):
        return sum(l["role"] == "auto" and l["state"] != "closing" for l in self.leases.values())

    def maintain(self):
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
            if (lease["role"] == "drain" and card and not external(card, {j["unit"] for j in current})
                    and self.regular() < self.limits["cards"]):
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
        cards = [pin] if pin is not None else sorted(map(int, self.cards), key=self.order_card)
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
            grants = [job["permission"]] if job["manual"] else [j["permission"] for j in primaries if j["fill"]]
            if pin is None:
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
            ident = self.db.execute("""INSERT INTO leases(card,role,state,trigger_job,grant,created)
                VALUES(?,?,'opening',?,?,?)""", (card, role, job["id"], packed(job["permission"]), self.now)).lastrowid
            self.leases[card] = dict(id=ident, state="opening", role=role, trigger_job=job["id"], card=card)
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
        for job in queued:
            spec = job["data"]
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
            if len(allowed) == 2 and self.cpu_auto < self.limits["cpu"] and self.regular() >= self.limits["cards"]:
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
