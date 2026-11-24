"""Stage A dependency and science checks with independently hosted runq workers."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import subprocess
import time

from .io import atomic_json, sha256
from .runq import start
from .runq_store import command_env, connect, lookup, packed, put, submit, unit, wake, write
from .scheduler3 import Blocked, Supervisor, evaluate_outcome, evaluation_bound, freeze_json, read
from .scheduler_shared3 import SharedSupervisor


def device_cost(db, owner, now, preparation=False):
    """Charge a campaign's dedicated lease once, and union its spans on mixed leases."""
    rows = db.execute("""SELECT l.id,l.acquired,l.released,l.state,l.trigger_job,j.id AS job,j.assigned,
        j.finished,json_extract(j.spec,'$.owner') AS owner,json_extract(j.spec,'$.preparation') AS prep
        FROM leases l JOIN jobs j ON (j.lease=l.id OR j.id=l.trigger_job)
        WHERE l.acquired IS NOT NULL ORDER BY l.id,j.assigned""")
    groups = defaultdict(list)
    for r in rows:
        groups[r["id"]].append(r)
    total = 0.
    for group in groups.values():
        mine = [r for r in group if r["owner"] == owner and (not preparation or r["prep"])]
        if not mine:
            continue
        first = group[0]
        # A lost broker's surviving worker retains the underlying physical fd.
        end = first["released"] or now
        if first["state"] == "lost":
            end = max(end, *(r["finished"] or now for r in group))
        if len(mine) == len(group):
            total += max(0., end-first["acquired"])
            continue
        cursor = first["acquired"]
        intervals = []
        for r in mine:
            begin = first["acquired"] if r["job"] == r["trigger_job"] else r["assigned"]
            if begin is not None:
                intervals.append((begin,min(end,r["finished"] or now)))
        for a,b in sorted(intervals):
            total += max(0.,b-max(cursor,a))
            cursor = max(cursor,b)
    return total


def apply_runq(config, review):
    root = review.get("runq_root")
    if root:
        from .cutover_review3 import PLANNED_CONTROLLER_CONTINUATION
        if review["schema"] != PLANNED_CONTROLLER_CONTINUATION or not Path(root).is_absolute() or review.get("temporary_npu"):
            raise ValueError("runq needs a reviewed controller continuation with an absolute account queue")
        config["runq_root"] = root


class QueueSupervisor(SharedSupervisor):
    def __init__(self, config, output):
        self.root = Path(config["runq_root"])
        self.db = connect(self.root)
        self.owner = str(Path(output).resolve())
        self.registered = {}
        super().__init__(config, output)
        start(self.root)

    def device_seconds(self):
        return device_cost(self.db,self.owner,time.time())

    def preparation_device_seconds(self):
        return device_cost(self.db,self.owner,time.time(),preparation=True)

    def supervision_tick(self):
        put(self.db,"permit."+self.owner,time.time()+60)

    def stage_drained(self):
        return True

    def actual_npu_concurrency(self, records):
        live = self.db.execute("""SELECT j.backend,l.card FROM jobs j LEFT JOIN leases l ON j.lease=l.id
                         WHERE j.state='running' AND json_extract(j.spec,'$.owner')=?""",(self.owner,)).fetchall()
        for backend in ("cpu","npu"):
            n = sum(r["backend"]==backend for r in live)
            self.state["actual_"+backend+"_worker_concurrency"] = n
            key = "peak_"+backend+"_worker_concurrency"
            self.state[key] = max(self.state.get(key,0),n)
        return len({r["card"] for r in live if r["card"] is not None})

    def register_job(self, task, row):
        index = len(row["attempts"])
        key = f'{self.config["run_id"]}/{task["id"]}/{index}'
        if key in self.registered:
            return self.registered[key]
        move = row.get("queue_relocation")
        target = f'npu:{move["to"]}' if move and move["next_attempt"] == index else "auto"
        existing = self.db.execute("SELECT id,spec,target FROM jobs WHERE job_key=?", (key,)).fetchone()
        if existing:
            spec = json.loads(existing["spec"])
            if spec["owner"] != self.owner or spec["task_id"] != task["id"]:
                raise Blocked("queue request identity changed")
            if target != "auto" and existing["target"] != target:
                raise Blocked("relocation queue target changed")
            ident = existing["id"]
        else:
            allowed = ["npu","cpu"] if task["device"] == "unbound" else ["npu" if task["device"]=="npu:0" else "cpu"]
            ram = {b:self.worker_reservations({**task,"device":"npu:0" if b=="npu" else "cpu"})["reserved_rss_bytes"] for b in allowed}
            group = getattr(self, "resource_placement", None)
            members = group.groups.get(task["id"]) if group else None
            spec = dict(owner=self.owner,task_id=task["id"],allowed=allowed,ram=ram,
                        hbm=task.get("reserved_hbm_bytes",8*2**30),deadline=self.config["deadline_unix"],
                        binding=self.owner+"/"+members[0] if members else None,preparation=row.get("preparation",False))
            ident = submit(self.db,key,spec,target)
        self.registered[key] = ident
        return ident

    def publish(self, record):
        attempt = record["attempts"][-1]
        ident = attempt["runq_id"]
        value = packed(attempt["runq_command"])
        with write(self.db):
            job = lookup(self.db,ident)
            if job["command"] and job["command"] != value:
                raise Blocked("queue worker command changed")
            self.db.execute("UPDATE jobs SET command=? WHERE id=? AND command IS NULL",(value,ident))

    def reserve_task(self, task, record):
        move = record.get("queue_relocation")
        if not move:
            return super().reserve_task(task, record)
        from .resume3 import bound
        from .resume_adapter3 import build_relay_receipt
        attempts = record["attempts"]
        if not attempts or len(attempts) != move["next_attempt"]:
            raise Blocked("relocation predecessor changed")
        previous = attempts[-1]
        if (previous.get("runq_id") != move["previous_job"] or previous.get("physical_id") != move["from"]
                or previous.get("status") != "stopped_budget" or previous.get("reason") != "signal"):
            raise Blocked("relocation predecessor changed")
        job = lookup(self.db, record["runq_id"])
        card = self.db.execute("SELECT card FROM leases WHERE id=?", (job["lease"],)).fetchone()
        if not card or card[0] != move["to"] or job["target"] != f'npu:{move["to"]}':
            raise Blocked("relocation reserved on the wrong card")
        task_ref = {"path": record["task_path"], "sha256": sha256(record["task_path"])}
        results = []
        for attempt in attempts:
            path = Path(attempt["path"]) / "result.json"
            results.append({"path": str(path), "sha256": sha256(path)})
        receipt = bound(record["resume_receipt"])
        if receipt != build_relay_receipt(task_ref, results, record.get("evaluation_reserved_tokens", 0)):
            raise Blocked("relocation receipt does not match the interrupted attempts")
        super().reserve_task(task, record, evaluation_increment=receipt["evaluation_additional_reservation_tokens"])

    def start_ready_tasks(self, tasks, slots):
        pending = {t["id"]:t for t in tasks if self.state["tasks"][t["id"]]["status"]=="pending"}
        publish = []
        for tid, task in pending.items():
            record = self.state["tasks"][tid]
            ident = self.register_job(task,record)
            job = lookup(self.db,ident)
            record["runq_id"] = ident
            record["queue_wait"] = job["reason"]
            if job["state"] == "queued":
                continue
            if job["state"] != "reserved":
                raise Blocked(f"unexpected queue state for pending task {tid}: {job['state']}")
            backend = job["backend"]
            if task["device"] == "unbound":
                self.resource_placement.bind(task,"npu:0" if backend=="npu" else "cpu",pending)
            if task["device"] != ("npu:0" if backend=="npu" else "cpu"):
                raise Blocked("queue backend differs from frozen recipe")
            workers,plans,_ = self.worker_plans([task],backend)
            _,_,directory,path,argv,resources = plans[0]
            worker = workers[0]
            directory.mkdir(parents=True,exist_ok=True)
            freeze_json(path,task)
            cmd = dict(argv=argv,cwd=worker["cwd"],env={**command_env(),**worker["env"]},
                       affinity=self.config["host_cpu_affinity"],log=str(directory/"stdout.log"))
            self.reserve_task(task,record)
            record.update(status="running",reason=None,task_path=str(path),**resources)
            record["attempts"].append(dict(path=str(directory),runq_id=ident,runq_command=cmd,command=argv,
                          started_unix=job["assigned"],status="starting",device_seconds=0.))
            publish.append(record)
        if publish:
            self._save()  # Recover the immutable command and its budget before allowing exec.
            for record in publish:
                self.publish(record)
        if pending:
            wake(self.root)

    def _poll(self, task, record):
        attempt = record["attempts"][-1]
        job = lookup(self.db,attempt["runq_id"])
        directory = Path(attempt["path"])
        if not job["command"]:
            self.publish(record)
            wake(self.root)
        if job["pid"]:
            identity = {"pid":job["pid"],"start_ticks":job["ticks"]}
            record["process"] = attempt["process"] = identity
            attempt["unit"] = unit(self.root,"job",job["id"],job["generation"])
        if job["lease"]:
            card = self.db.execute("SELECT card FROM leases WHERE id=?",(job["lease"],)).fetchone()[0]
            attempt["physical_id"] = card
        if job["started"]:
            attempt.update(started_unix=job["started"],status="running")
        q = dict(status=job["state"],return_code=job["rc"],job_id=job["id"],lease_id=job["lease"],
                 unit=attempt.get("unit"),generation=job["generation"],process=attempt.get("process"),
                 started_unix=job["started"],finished_unix=job["finished"],
                 physical_ids=[attempt["physical_id"]] if "physical_id" in attempt else [],
                 accounting_scope="worker occupancy; campaign ledger charges each physical lease once")
        if record.get("queue") != q:
            atomic_json(directory/"queue.json",q)
            record["queue"] = q
        hb = read(directory/"heartbeat.json")
        if hb:
            record["heartbeat"] = hb
        if not job["first_step"] and (directory/"steps.jsonl").exists():
            with (directory/"steps.jsonl").open() as stream:
                line = stream.readline()
            if line.endswith("\n"):
                first = json.loads(line)
                self.db.execute("UPDATE jobs SET first_step=? WHERE id=? AND first_step IS NULL",
                                (first["time_unix"],job["id"]))
        if job["state"] not in ("completed","failed","cancelled"):
            return False
        result_path = directory/"result.json"
        result = read(result_path)
        status,reason = evaluate_outcome(task,result,job["rc"])
        if result and (result.get("task_sha256") != sha256(record["task_path"]) or
                       (task.get("evaluation_budget_key") and result.get("evaluation",{}).get("tokens",0)>evaluation_bound(task))):
            status,reason = "failed","queue_worker_identity_or_evaluation_bound"
            record["observation_error"] = reason
        attempt.update(status=status,return_code=job["rc"],finished_unix=job["finished"],reason=reason)
        record.update(status=status,reason=reason,result_path=str(result_path),result_sha256=sha256(result_path) if result else None)
        self.retry_transient(task,record,result)
        return True

    def verify_completed(self, task, record):
        Supervisor.verify_completed(self,task,record)
        from .resume3 import verify_adapter_result
        result = read(record["result_path"])
        move = record.get("queue_relocation")
        if move:
            from .continuation3 import continuation_origin
            adapter = result.get("execution_adapter", {})
            if (record["attempts"][-1].get("physical_id") != move["to"]
                    or adapter.get("schema") != "pilot-r3-execution-adapter-v1"
                    or adapter.get("receipt") != record["resume_receipt"]
                    or adapter.get("adapter") != self.config["resume_adapter"]):
                raise Blocked("relocated result does not match its physical card and resume receipt")
            # Follow the interrupted attempt through any earlier source migration.
            continuation_origin(task, result)
            return
        verify_adapter_result(self.config,task,record,result)
        if task.get("stage_a_migration"):
            from .source_migration3 import migration_origin
            migration_origin(task,result)

    def stop_all(self):
        put(self.db,"permit."+self.owner,0)
        self.state.update(status="a_blocked_needs_repair",stage_b_gate="closed",blocking_reason=self.stop_reason)
        if self.stop_reason in {"global_deadline","global_device_hour_limit","preparation_device_hour_limit",
                                 "storage_exhausted","storage_reserve_exhausted"}:
            self.state["status"] = "stopped_budget"
            rows = self.db.execute("""SELECT id,generation FROM jobs WHERE state IN ('starting','running')
                                      AND json_extract(spec,'$.owner')=?""",(self.owner,))
            names = [unit(self.root,"job",r["id"],r["generation"]) for r in rows]
            if names:
                subprocess.run(["systemctl","--user","stop",*names],check=False)
        self._save()
