"""Stage A dependency and science checks with independently hosted runq workers."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import subprocess
import time

from .io import atomic_json, sha256
from .hbm_profile3 import GIB, HbmProfile, hbm_shape, oom_kind
from .runq import start
from .runq_store import command_env, connect, event, lookup, packed, put, setting, submit, unit, wake, write
from .scheduler3 import Blocked, Supervisor, evaluate_outcome, evaluation_bound, freeze_json, live, read
from .scheduler_shared3 import SharedSupervisor


def queue_hbm(db, owner, task, allowed, observed=0, shape=None):
    if "npu" not in allowed:
        return task.get("reserved_hbm_bytes", 8 * GIB)
    floors = setting(db, "hbm_floors." + owner, {})
    floor = floors.get(shape or hbm_shape(task), 0) if floors else 0
    declared = task.get("reserved_hbm_bytes", 0)
    return max(declared, observed, floor) or 8 * GIB


def signal_evaluation_coverage(config, task, record):
    """Include the unused part of a carried stage's inherited reservation once."""
    reserved = record.get("evaluation_reserved_tokens", 0)
    increments = config.get("migration_evaluation_reservations", {})
    if task.get("stage_a_migration") and task["id"] in increments:
        from .followup_migration3 import remaining_evaluation
        from .source_migration3 import read_bound
        entry = read_bound(task["stage_a_migration"])
        if entry["task_id"] != task["id"] or entry["evaluation_additional_reservation_tokens"] != increments[task["id"]]:
            raise Blocked("signal resume migration accounting changed")
        reserved += max(0, remaining_evaluation(task, entry) - increments[task["id"]])
    return reserved


def first_update_time(directory):
    path = Path(directory) / "steps.jsonl"
    if path.is_file():
        with path.open() as stream:
            line = stream.readline()
        if line.endswith("\n"):
            return json.loads(line)["time_unix"]
    return None


def device_cost(db, owner, now, preparation=False, *, alongside=None):
    """Charge a campaign's dedicated lease once, and union its spans on mixed leases."""
    rows = db.execute("""SELECT l.id,l.acquired,l.released,l.state,l.trigger_job,j.id AS job,j.assigned,
        j.finished,json_extract(j.spec,'$.owner') AS owner,json_extract(j.spec,'$.preparation') AS prep
        FROM leases l JOIN jobs j ON (j.lease=l.id OR j.id=l.trigger_job)
        WHERE l.acquired IS NOT NULL ORDER BY l.id,j.assigned""")
    groups = defaultdict(list)
    for r in rows:
        groups[r["id"]].append(r)
    def duration(group, mine):
        if not mine:
            return 0.
        first = group[0]
        # A lost broker's surviving worker retains the underlying physical fd.
        end = first["released"] or now
        if first["state"] == "lost":
            end = max(end, *(r["finished"] or now for r in group))
        if len(mine) == len(group):
            return max(0., end-first["acquired"])
        cursor = first["acquired"]
        intervals = []
        for r in mine:
            begin = first["acquired"] if r["job"] == r["trigger_job"] else r["assigned"]
            if begin is not None:
                intervals.append((begin,min(end,r["finished"] or now)))
        total = 0.
        for a,b in sorted(intervals):
            total += max(0.,b-max(cursor,a))
            cursor = max(cursor,b)
        return total

    total = 0.
    for group in groups.values():
        mine = [r for r in group if r["owner"] == owner and (not preparation or r["prep"])]
        if not mine:
            continue
        primary = [r for r in group if r["owner"] == alongside and (not preparation or r["prep"])] if alongside else []
        # The original controller already charges its own spans. Bill only the increment to their union.
        total += duration(group, mine + primary) - duration(group, primary)
    return total


def apply_runq(config, review):
    root = review.get("runq_root")
    if root:
        from .cutover_review3 import PLANNED_CONTROLLER_CONTINUATION
        if review["schema"] != PLANNED_CONTROLLER_CONTINUATION or not Path(root).is_absolute() or review.get("temporary_npu"):
            raise ValueError("runq needs a reviewed controller continuation with an absolute account queue")
        config["runq_root"] = root


class QueueSupervisor(SharedSupervisor):
    def __init__(self, config, output, *, inherited_hbm=None):
        self.root = Path(config["runq_root"])
        self.db = connect(self.root)
        self.owner = str(Path(output).resolve())
        self.registered = {}
        super().__init__(config, output)
        self.hbm_profile = HbmProfile(self.output)
        self.inherited_hbm = inherited_hbm or {}
        self.shape_stats = self.hbm_profile.summaries()
        self.hbm_jobs = {}
        self.last_hbm_sample = None
        self.capacity_retry_jobs = set(setting(self.db, "hbm_retry_jobs." + self.owner, []))
        recovered = False
        for row in self.state["tasks"].values():
            if row["status"] == "failed" and row.get("runq_id") in self.capacity_retry_jobs:
                recovered |= self.retry_capacity(read(row["task_path"]), row, read(row["result_path"]))
        if recovered:
            self._save()
        start(self.root)

    def device_seconds(self):
        return device_cost(self.db,self.owner,time.time())

    def preparation_device_seconds(self):
        return device_cost(self.db,self.owner,time.time(),preparation=True)

    def supervision_tick(self):
        put(self.db,"permit."+self.owner,time.time()+60)
        self.refresh_hbm()

    def measured_hbm(self, shape):
        stats = getattr(self, "shape_stats", {}).get(shape, {})
        measured = stats.get("reservation_bytes", 0)
        return max(measured, getattr(self, "inherited_hbm", {}).get(shape, 0))

    def refresh_hbm(self):
        samples = setting(self.db, "samples", [])
        if not samples or samples[-1]["at"] == self.last_hbm_sample:
            return
        now = time.time()
        samples = [s for s in samples[-2:] if now - s["at"] < 15 and
                   (self.last_hbm_sample is None or s["at"] > self.last_hbm_sample)]
        if not samples:
            return
        rows = self.db.execute("""SELECT j.*,l.card FROM jobs j LEFT JOIN leases l ON j.lease=l.id
            WHERE j.state IN ('queued','reserved','starting','running') AND COALESCE(j.backend,'npu')='npu'
            AND json_extract(j.spec,'$.owner')=?""", (self.owner,)).fetchall()
        active = {}
        with self.hbm_profile.db:
            for job in rows:
                spec = json.loads(job["spec"])
                if "npu" not in spec["allowed"]:
                    continue
                row = self.state["tasks"].get(spec["task_id"], {})
                task = self.hbm_jobs.get(job["id"])
                if task is None:
                    path = row.get("task_path")
                    frozen = read(path) if path else None
                    if not frozen or frozen["id"] != spec["task_id"]:
                        continue
                    task = (hbm_shape(frozen), frozen["model_id"], frozen["batch_size"], frozen.get("reserved_hbm_bytes", 0))
                active[job["id"]] = task
                if not job["pid"] or not row.get("attempts"):
                    continue
                attempt = row["attempts"][-1]
                if attempt.get("runq_id") != job["id"]:
                    continue
                name = unit(self.root, "job", job["id"], job["generation"])
                first_update = job["first_step"] or first_update_time(attempt["path"])
                for sample in samples:
                    card = sample["cards"].get(str(job["card"]), {})
                    for process in card.get("processes", []):
                        if (process["unit"] == name and process["pid"] == job["pid"] and
                                (process["identity"] or {}).get("start_ticks") == job["ticks"]):
                            before = first_update is None or sample["at"] < first_update
                            self.hbm_profile.record_driver(attempt["path"], *task[:3], process["bytes"], before)
        self.hbm_jobs = active
        self.shape_stats = self.hbm_profile.summaries()
        with write(self.db):
            floors = setting(self.db, "hbm_floors." + self.owner, {})
            for ident, (shape, _, _, declared) in active.items():
                job = lookup(self.db, ident)
                if job["state"] not in ("queued", "reserved", "starting", "running") or job["backend"] == "cpu":
                    continue
                spec = json.loads(job["spec"])
                current = spec["hbm"]
                original = json.loads(job["original"])[0]["hbm"]
                # Retain an operator's change when taking ownership of the baseline.
                prior = spec.get("profile_hbm_bytes", original)
                minimum = spec.get("hbm_min_bytes", 0) if prior == current else current
                wanted = max(declared, minimum, floors.get(shape, 0), self.measured_hbm(shape)) or 8 * GIB
                if wanted < current and job["state"] in ("starting", "running") and not job["first_step"]:
                    continue
                if wanted != current:
                    spec.update(hbm=wanted, profile_hbm_bytes=wanted, hbm_min_bytes=minimum)
                    self.db.execute("UPDATE jobs SET spec=? WHERE id=?", (packed(spec), ident))
                    event(self.db, ident, "shape_hbm_reservation", packed(dict(previous=current, current=wanted, shape=shape)))
        self.last_hbm_sample = samples[-1]["at"]
        wake(self.root)

    def close(self):
        self.hbm_profile.db.close()
        super().close()

    def stage_drained(self):
        return True

    def inflight_model_ids(self):
        """Find models whose queue requests still need a controller on handoff."""
        records = self.state["tasks"]
        models = {row["model_id"] for row in records.values()
                  if row["status"] == "running" or (row["status"] == "pending" and row.get("runq_id"))}
        jobs = self.db.execute("""SELECT DISTINCT json_extract(spec, '$.task_id') FROM jobs
            WHERE state IN ('queued','reserved','starting','running') AND json_extract(spec, '$.owner')=?
            AND COALESCE(json_extract(spec, '$.purpose'), '') != 'deferred_source_review'""",
            (self.owner,))
        for (tid,) in jobs:
            if tid in records:
                models.add(records[tid]["model_id"])
                continue
            # SQLite submission may have committed before scheduler.json was saved.
            if not isinstance(tid, str) or not tid or Path(tid).name != tid or tid in (".", ".."):
                raise Blocked(f"invalid queue task id: {tid!r}")
            directory = self.output / "tasks" / tid
            task = read(directory / "task.json")
            if task is None:
                task = read(directory / "intent.json")
            if not task or task.get("id") != tid or not task.get("model_id"):
                raise Blocked(f"cannot identify the frozen model for queue task {tid}")
            models.add(task["model_id"])
        return models

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
            shape = hbm_shape(task)
            hbm = queue_hbm(self.db,self.owner,task,allowed,self.measured_hbm(shape),shape)
            spec = dict(owner=self.owner,task_id=task["id"],allowed=allowed,ram=ram,
                        hbm=hbm,deadline=self.config["deadline_unix"],
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

    def retry_capacity(self, task, row, result):
        """Reopen a reviewed first OOM from the last committed optimizer update."""
        ident = row.get("runq_id")
        if (ident not in self.capacity_retry_jobs or row["status"] != "failed" or row.get("observation_error")
                or len(row["attempts"]) != 1 or row.get("resume_receipt") or row.get("qualified_task")
                or not task or task.get("device") != "npu:0" or task.get("resume_checkpoint") or task.get("stage_a_migration")
                or not result):
            return False
        attempt = row["attempts"][0]
        job = lookup(self.db, ident)
        original = json.loads(job["original"])[0]
        floor = queue_hbm(self.db, self.owner, task, ["npu"], self.measured_hbm(hbm_shape(task)))
        path = Path(attempt["path"]) / "result.json"
        kind = oom_kind(result, path.parent / "stdout.log")
        if (not kind or attempt.get("runq_id") != ident or original.get("owner") != self.owner or original["task_id"] != task["id"]
                or job["state"] != "failed" or job["backend"] != "npu" or job["rc"] != 1 or not job["pid"]
                or original["hbm"] >= floor
                or live({"pid": job["pid"], "start_ticks": job["ticks"]}) or result.get("status") != "failed"
                or result.get("id") != task["id"] or not task.get("source_identity")
                or result.get("source_identity") != task["source_identity"] or result.get("device") != "npu:0"
                or result.get("task_sha256") != sha256(row["task_path"]) or str(path) != row.get("result_path")
                or sha256(path) != row.get("result_sha256")):
            return False
        progress = result.get("training", {})
        if any(type(progress.get(k)) is not int or progress[k] < 0 for k in ("steps", "tokens")):
            return False
        checkpoint = result.get("checkpoint")
        if progress["steps"] or progress["tokens"]:
            if (not checkpoint or checkpoint != read(path.parent / "checkpoint.json", {}).get("current")
                    or not checkpoint.get("path") or not Path(checkpoint["path"]).is_file()
                    or sha256(checkpoint["path"]) != checkpoint.get("sha256")):
                return False
            if any(checkpoint.get(k) != progress[k] for k in ("steps", "tokens")):
                quantum = task["batch_size"] * task["sequence_length"]
                if not (type(checkpoint.get("steps")) is int and 0 < checkpoint["steps"] < progress["steps"]
                        and checkpoint.get("tokens") == checkpoint["steps"] * quantum and progress["tokens"] == progress["steps"] * quantum):
                    return False
            row.update(retry_checkpoint=checkpoint["path"], retry_checkpoint_sha256=checkpoint["sha256"])
        elif checkpoint:
            return False
        row["capacity_retry"] = dict(job=ident, result={"path": str(path), "sha256": row["result_sha256"]},
                                     previous_hbm=original["hbm"], reserved_hbm=floor, checkpoint=checkpoint, oom=kind,
                                     replay_steps=progress["steps"]-(checkpoint or {}).get("steps", 0),
                                     replay_tokens=progress["tokens"]-(checkpoint or {}).get("tokens", 0))
        for key in ("result_path", "result_sha256", "heartbeat", "queue", "process"):
            row.pop(key, None)
        row.update(status="pending", reason=None)
        return True

    def retry_transient(self, task, record, result):
        return super().retry_transient(task, record, result) or self.retry_capacity(task, record, result)

    def reserve_task(self, task, record):
        if record.get("qualified_task"):
            from .source_migration3 import read_bound
            receipt = read_bound(task["stage_a_migration"])
            return super().reserve_task(task, record,
                evaluation_increment=receipt["evaluation_additional_reservation_tokens"])
        move = record.get("queue_relocation")
        resume = record.get("queue_signal_resume") or move
        if not resume:
            return super().reserve_task(task, record)
        from .resume3 import bound
        from .resume_adapter3 import build_relay_receipt
        attempts = record["attempts"]
        if not attempts or len(attempts) != resume["next_attempt"]:
            raise Blocked("signal resume predecessor changed")
        previous = attempts[-1]
        if (previous.get("runq_id") != resume["previous_job"]
                or previous.get("status") != "stopped_budget" or previous.get("reason") != "signal"):
            raise Blocked("signal resume predecessor changed")
        job = lookup(self.db, record["runq_id"])
        if move:
            card = self.db.execute("SELECT card FROM leases WHERE id=?", (job["lease"],)).fetchone()
            if (previous.get("physical_id") != move["from"] or not card or card[0] != move["to"]
                    or job["target"] != f'npu:{move["to"]}'):
                raise Blocked("relocation reserved on the wrong card")
        task_ref = {"path": record["task_path"], "sha256": sha256(record["task_path"])}
        results = []
        for attempt in attempts:
            path = Path(attempt["path"]) / "result.json"
            results.append({"path": str(path), "sha256": sha256(path)})
        receipt = bound(record["resume_receipt"])
        covered = record.get("evaluation_reserved_tokens", 0) if move else signal_evaluation_coverage(self.config, task, record)
        if receipt != build_relay_receipt(task_ref, results, covered):
            raise Blocked("signal resume receipt does not match the interrupted attempts")
        super().reserve_task(task, record, evaluation_increment=receipt["evaluation_additional_reservation_tokens"])

    def start_ready_tasks(self, tasks, slots):
        pending = {t["id"]:t for t in tasks if self.state["tasks"][t["id"]]["status"]=="pending"}
        publish = []
        for tid, task in pending.items():
            record = self.state["tasks"][tid]
            stats = getattr(self, "shape_stats", {}).get(hbm_shape(task), {})
            leader = stats.get("leader")
            if leader and leader != tid and task["device"] in ("npu:0", "unbound"):
                key = f'{self.config["run_id"]}/{tid}/{len(record["attempts"])}'
                if not self.db.execute("SELECT 1 FROM jobs WHERE job_key=?", (key,)).fetchone():
                    record["queue_wait"] = "waiting for measured NPU representative " + leader
                    continue
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
        if not job["first_step"]:
            first = first_update_time(directory)
            if first is not None:
                self.db.execute("UPDATE jobs SET first_step=? WHERE id=? AND first_step IS NULL",
                                (first,job["id"]))
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
        if result and job["backend"] == "npu" and not record.get("observation_error"):
            with self.hbm_profile.db:
                self.hbm_profile.record_result(task, sha256(record["task_path"]), result_path, result,
                                               record["result_sha256"], outcome=status)
            self.shape_stats = self.hbm_profile.summaries()
        self.retry_transient(task,record,result)
        return True

    def verify_completed(self, task, record):
        Supervisor.verify_completed(self,task,record)
        from .resume3 import verify_adapter_result
        result = read(record["result_path"])
        move = record.get("queue_relocation")
        if move or record.get("queue_signal_resume"):
            from .continuation3 import continuation_origin
            adapter = result.get("execution_adapter", {})
            runtime = getattr(self, "worker_runtimes", {}).get(task["id"], {})
            if ((move and record["attempts"][-1].get("physical_id") != move["to"])
                    or adapter.get("schema") != "pilot-r3-execution-adapter-v1"
                    or adapter.get("receipt") != record["resume_receipt"]
                    or adapter.get("adapter") != runtime.get("adapter", self.config["resume_adapter"])):
                raise Blocked("signal resume result does not match its physical card or receipt")
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
