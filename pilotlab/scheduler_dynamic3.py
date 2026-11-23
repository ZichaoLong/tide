"""Durable task admission to a finite NPU owner; no device operations here."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from .io import sha256
from .scheduler3 import Blocked, evaluate_outcome, evaluation_bound, freeze_json, live, read, worker_limit

GIB = 2**30


@dataclass
class OwnerCapacity:
    key: str
    physical_id: int | None
    slots: int
    policy_limit: int
    active: int
    hbm: int
    threads: int
    rss: int
    observed_unix: float
    external: int
    empty_hbm: int
    admissions_left: int

    def fits(self, task, resource):
        return (self.slots > 0 and self.admissions_left > 0 and self.hbm >= task.get("reserved_hbm_bytes", 8*GIB)
                and self.threads >= resource["cpu_threads"] and self.rss >= resource["reserved_rss_bytes"])

    def charge(self, task, resource):
        self.slots -= 1
        self.admissions_left -= 1
        self.active += 1
        self.hbm -= task.get("reserved_hbm_bytes", 8*GIB)
        self.threads -= resource["cpu_threads"]
        self.rss -= resource["reserved_rss_bytes"]


class DynamicAdmission:
    """SharedSupervisor hooks for task completion independent of lease close."""

    def dynamic_enabled(self):
        return bool(self.config.get("dynamic_shared_admission"))

    def available_checkpoint_writes(self):
        limit = self.config.get("max_inflight_checkpoint_bytes")
        if limit is not None and self.config.get("use_live_checkpoint_headroom"):
            free = self.resource_cache["available_storage_bytes"]
            return min(limit, max(0, free - self.config.get("storage_reserve_bytes", 0)))
        return limit

    def start_dynamic_ready_tasks(self, tasks, slots):
        pending = [task for task in tasks if self.state["tasks"][task["id"]]["status"] == "pending"]
        records = [record for record in self.state["tasks"].values() if record["status"] == "running"]
        placement = getattr(self, "dispatch_placement", None)
        if placement:
            placement.observe_rates(records, time.time())
        exclusive_limit = self.config.get("exclusive_max_hbm_bytes")
        if exclusive_limit is not None:
            possible = []
            for task in pending:
                record = self.state["tasks"][task["id"]]
                if task.get("device", "npu:0").startswith("npu") and task.get("reserved_hbm_bytes", 8*GIB) > exclusive_limit:
                    record["reason"] = "reservation_exceeds_exclusive_hbm_limit"
                else:
                    possible.append(task)
            if pending and not possible and not records:
                raise Blocked("reservation_exceeds_exclusive_hbm_limit:" + pending[0]["id"])
            now = time.time()
            pending = [task for task in possible if self.state["tasks"][task["id"]].get("hbm_retry_after_unix", 0) <= now]
            pending.sort(key=lambda task: self.state["tasks"][task["id"]].get("hbm_deferrals", 0))
        slots = min(slots, max(0, worker_limit(self.config, 80)-len(records)))
        groups = self.state.get("shared_groups", {})
        direct = getattr(self, "resource_placement", None)
        owners = {record["shared_group"] for record in records
                  if record.get("device", "npu:0").startswith("npu") and record.get("shared_group")}
        owners.update(key for key, entry in groups.items()
                      if entry.get("dynamic_admission") and not entry.get("release_verified")
                      and not entry.get("launch_aborted_before_spawn"))
        owner_limit = self.config.get("max_npu_devices", 8)
        if type(owner_limit) is not int or not 1 <= owner_limit <= 8:
            raise Blocked("dynamic_owner_limit_must_be_at_most_eight")
        if len(owners) > owner_limit:
            raise Blocked("dynamic_existing_owner_limit_exceeded")
        probe = lambda task: task.get("purpose") == "capacity_calibration" or task.get("phase") == "preflight"
        probes = [task for task in pending if task.get("device", "npu:0").startswith("npu") and probe(task)]
        probe_owners = {record.get("shared_group") for record in records
                        if record.get("device", "npu:0").startswith("npu") and probe(record)}
        if probes and not probe_owners and len(owners) >= owner_limit:
            # A single drain is enough to establish a probe lane. Never signal
            # healthy children, or close one more owner on every poll.
            already_draining = any(groups[key].get("close_requested") == "capacity_probe_lane"
                                   for key in owners if key in groups)
            candidates = [key for key in owners if groups.get(key, {}).get("dynamic_admission")
                          and not groups[key].get("close_requested")]
            if candidates and not already_draining:
                candidate = min(candidates, key=lambda key: (sum(row.get("shared_group") == key for row in records), key))
                self.queue_admission(candidate, close_reason="capacity_probe_lane")
        if slots <= 0:
            return
        usage = {"checkpoint": 0, "rss": 0, "threads": 0, "cpu_threads": 0}
        checkpoint_limit = self.config.get("max_inflight_checkpoint_bytes")
        write_headroom = self.available_checkpoint_writes()

        def charge(task):
            resource = self.worker_reservations(task)
            if checkpoint_limit is not None:
                usage["checkpoint"] += resource["checkpoint_reservation_bytes"]
            usage["rss"] += resource["reserved_rss_bytes"]
            usage["threads"] += resource["cpu_threads"]
            if task.get("device") == "cpu":
                usage["cpu_threads"] += resource["cpu_threads"]

        for record in records:
            charge(record)
        if direct:
            usage["rss"] = max(usage["rss"], (getattr(self, "resource_cache", None) or {}).get("rss_bytes", 0))

        def fits(task):
            resource = self.worker_reservations(task)
            checkpoint = resource["checkpoint_reservation_bytes"]
            if checkpoint_limit is not None and checkpoint > checkpoint_limit:
                raise Blocked("single_task_checkpoint_reservation_exceeds_limit")
            limits = (("rss", "reserved_rss_bytes", self.config.get("max_rss_bytes", 1024*GIB)),)
            host_threads = self.config.get("max_cpu_threads", 320)
            if host_threads is not None:
                limits += (("threads", "cpu_threads", host_threads),)
            if task.get("device") == "cpu":
                limits += (("cpu_threads", "cpu_threads", self.config.get("max_cpu_task_threads", 160)),)
            for key, field, ceiling in limits:
                if resource[field] > ceiling:
                    raise Blocked(f"single_task_{key}_reservation_exceeds_limit")
                if usage[key]+resource[field] > ceiling:
                    return False
            return write_headroom is None or usage["checkpoint"] + checkpoint <= write_headroom

        cpu_count = sum(record.get("device") == "cpu" for record in records)
        cpu_limit = self.config.get("max_cpu_workers", 40)
        cpu_pending = [task for task in pending if task.get("device") == "cpu"]
        if getattr(self, "cpu_ramp", None):
            from collections import Counter
            cpu_limit = min(cpu_limit, self.cpu_ramp.admission(records, cpu_count, time.time(), self.state))
            remaining = Counter(row["model_id"] for row in self.state["tasks"].values()
                                if row["stage"] == "a1" and row["status"] in ("pending", "running"))
            cpu_pending.sort(key=lambda t: (t["phase"] == "a1", remaining[t["model_id"]]))
        elastic = [task for task in pending if task.get("device") == "unbound"] if placement or direct else []

        def drain_cpu(queue):
            nonlocal slots, cpu_count
            for task in queue:
                if not slots or cpu_count >= cpu_limit:
                    break
                if fits(task):
                    self.start_group([task], "cpu")
                    charge(task)
                    slots -= 1
                    cpu_count += 1

        # Exclusive calibrations unlock other work. With no shared queue the
        # established fixed-task ordering is retained.
        if elastic:
            drain_cpu(t for t in cpu_pending if probe(t))
            cpu_pending = [t for t in cpu_pending if not probe(t)]
        else:
            drain_cpu(cpu_pending)
            cpu_pending = []
        probe_limit = self.config.get("dynamic_probe_owners", 1)
        if type(probe_limit) is not int or not 1 <= probe_limit <= 2:
            raise Blocked("dynamic_probe_owner_limit_must_be_one_or_two")
        for task in probes:
            if not slots or len(owners) >= owner_limit or len(probe_owners) >= probe_limit:
                break
            if fits(task):
                key = self.start_group([task], "npu")
                owners.add(key)
                probe_owners.add(key)
                charge(task)
                slots -= 1
        ready = [task for task in pending if task.get("device", "npu:0").startswith("npu") and not probe(task)]
        available = []
        for key in sorted(owners):
            entry = groups.get(key, {})
            if not slots or not (ready or elastic):
                break
            if not entry.get("dynamic_admission") or entry.get("close_requested"):
                continue
            group = self.group_observation(key)
            if group.get("status") != "running" or group.get("allocation_acquired_unix") is None or group.get("capacity_deferral"):
                continue
            manifest = read(entry["manifest_path"])
            limits = manifest["dynamic_admission"]
            admissions_left = min(limits["max_total_workers"]-len(entry["task_ids"]),
                                  limits["max_commands"]-1-len(entry.get("admission_commands", [])))
            if admissions_left <= 0:
                self.queue_admission(key, close_reason="finite_owner_admission_limit")
                continue
            target = group.get("policy", {}).get("admission_limit", 1)
            if type(target) is not int or not 1 <= target <= min(16, manifest["max_parallel"]):
                raise Blocked("dynamic_owner_policy_limit_invalid")
            if target == 1 and group.get("policy", {}).get("unmeasured_exploration_used") is not True:
                # One reserved waiting worker exposes a real backlog to the
                # runner's single finite 1->2 throughput measurement.
                target = min(2, manifest["max_parallel"])
            active = [record for record in self.state["tasks"].values()
                      if record["status"] == "running" and record.get("shared_group") == key]
            device = group.get("resources", {}).get("device") or {}
            reserved = sum(record.get("reserved_hbm_bytes", 8*GIB) for record in active)
            used = device.get("used_bytes", 0)
            baseline = device.get("baseline_used_bytes", 4*GIB)
            hbm = max(used, baseline + reserved)
            if device.get("external_occupants"):
                hbm = max(hbm, used + reserved)
            hbm_limit = min(manifest["max_hbm_bytes"], device.get("total_bytes", manifest["max_hbm_bytes"]))
            threads = sum(self.worker_reservations(record)["cpu_threads"] for record in active)
            rss = sum(self.worker_reservations(record)["reserved_rss_bytes"] for record in active)
            capacity = OwnerCapacity(key, group.get("physical_id"), target-len(active),
                group.get("policy", {}).get("admission_limit", 1), len(active), hbm_limit-hbm,
                manifest["max_cpu_threads"]-threads, manifest["max_rss_bytes"]-rss,
                device.get("observed_unix", 0), len(device.get("external_occupants", [])), hbm_limit-baseline, admissions_left)
            available.append(capacity)

        def append_to(capacity, task):
            nonlocal slots
            self.append_group(capacity.key, [task])
            capacity.charge(task, self.worker_reservations(task))
            charge(task)
            slots -= 1
            if not capacity.admissions_left:
                self.queue_admission(capacity.key, close_reason="finite_owner_admission_limit")

        if direct:
            drain_cpu(cpu_pending)
            pending_by_id = {task["id"]: task for task in pending}

            def held_card(task):
                resource = self.worker_reservations(task)
                return min((cap for cap in available if cap.physical_id is not None
                            and 0 <= time.time() - cap.observed_unix <= 15 and cap.fits(task, resource)),
                           key=lambda cap: (cap.active, -cap.hbm, cap.key), default=None)

            def new_owner(task):
                nonlocal slots
                owners.add(self.start_group([task], "npu"))
                charge(task)
                slots -= 1

            # Open physical leases for fixed-device trajectories first. Their
            # successors and checkpoints keep the backend of the parent.
            for task in ready:
                if not slots:
                    break
                if not fits(task):
                    continue
                if len(owners) < owner_limit:
                    new_owner(task)
                elif (card := held_card(task)) is not None:
                    append_to(card, task)

            npu_workers = sum(r.get("device") == "npu:0" for r in self.state["tasks"].values()
                              if r["status"] == "running")
            npu_limit = owner_limit * min(16, self.config.get("npu_workers_per_device", 16))
            for task in sorted(elastic, key=lambda t: t.get("reserved_hbm_bytes", 8*GIB)):
                if not slots:
                    break
                # Binding one seed also binds its untouched partner. Recheck
                # membership on every iteration rather than splitting a recipe.
                device = task["device"]
                cpu_ok = device != "npu:0" and cpu_count < cpu_limit and fits({**task, "device": "cpu"})
                npu_ok = device != "cpu" and fits({**task, "device": "npu:0"})
                hbm = task.get("reserved_hbm_bytes", 8*GIB)
                npu_ok = npu_ok and (exclusive_limit is None or hbm <= exclusive_limit)
                card = held_card(task) if npu_ok else None
                fresh = npu_ok and len(owners) < owner_limit
                if cpu_ok and (not (card or fresh) or cpu_count*npu_limit <= npu_workers*cpu_limit):
                    if device == "unbound":
                        direct.bind(task, "cpu", pending_by_id)
                    self.start_group([task], "cpu")
                    charge(task)
                    slots -= 1
                    cpu_count += 1
                elif card or fresh:
                    if device == "unbound":
                        direct.bind(task, "npu:0", pending_by_id)
                    if fresh:
                        new_owner(task)
                    else:
                        append_to(card, task)
                    npu_workers += 1
            self.state["dispatch_queue"] = {"policy": direct.policy, "observed_unix": time.time(),
                "cpu_workers": cpu_count, "cpu_limit": cpu_limit, "npu_owners": len(owners),
                "unbound_tasks": sum(t["device"] == "unbound" for t in elastic)}
            return

        def try_dispatch(protect_fixed):
            nonlocal slots, cpu_count
            # Only held owners with a fresh whole-card observation can offer a
            # first-member lane. Reserve fixed work's priority on its own pool,
            # without spending the other pool's shared checkpoint/RSS headroom.
            now = time.time()
            held = [cap for cap in available if cap.physical_id is not None and cap.admissions_left > 0
                    and not groups[cap.key].get("close_requested") and 0 <= now-cap.observed_unix <= 15
                    and self.owner_running(cap.key)]
            cpu_first = protect_fixed and cpu_count < cpu_limit and any(fits(t) for t in cpu_pending)
            npu_first = {cap.key for cap in held if protect_fixed and cap.slots > 0
                         and any(cap.fits(t, self.worker_reservations(t)) and fits(t) for t in ready)}

            def offers(task):
                resource = self.worker_reservations(task)
                cpu_ok = slots > 0 and not cpu_first and cpu_count < cpu_limit and fits({**task, "device":"cpu"})
                cards = [cap for cap in held if cap.key not in npu_first and cap.active < cap.policy_limit
                         and cap.fits(task, resource)]
                card = max(cards, key=lambda cap: (cap.policy_limit-cap.active, cap.hbm), default=None)
                npu_offer = None
                if slots > 0 and card and fits({**task, "device":"npu:0"}):
                    npu_offer = dict(owner=card.key, physical_id=card.physical_id, observed_unix=card.observed_unix,
                        active=card.active, limit=card.policy_limit, available_hbm_bytes=card.hbm,
                        external_occupants=card.external)
                busy = any((cap.active >= cap.policy_limit or cap.key in npu_first) and not cap.external
                           and cap.empty_hbm >= task.get("reserved_hbm_bytes", 8*GIB) for cap in held)
                return {"cpu":dict(offer={"workers":cpu_count,"limit":cpu_limit} if cpu_ok else None, lanes=cpu_limit,
                                   predictable_wait=cpu_first or 0 < cpu_limit <= cpu_count),
                        "npu:0":dict(offer=npu_offer, lanes=sum(cap.policy_limit for cap in held),
                                     held_cards=len(held), predictable_wait=busy)}

            choice = placement.dispatch_rank(elastic, cpu_limit, offers)
            if not choice:
                return False
            task, device, forecasts = choice
            self.reserve_task({**task, "device":device}, dict(self.state["tasks"][task["id"]]))
            placement.claim(task, device, forecasts, elastic)
            if device == "cpu":
                self.start_group([task], "cpu")
                cpu_count += 1
                slots -= 1
                charge(task)
            else:
                cap = next(cap for cap in held if cap.key == forecasts[device]["offer"]["owner"])
                append_to(cap, task)
            return True

        competed = bool(elastic and slots and (cpu_pending or ready))
        claimed = competed and try_dispatch(protect_fixed=True)
        drain_cpu(cpu_pending)
        for capacity in available:
            for task in list(ready):
                if slots <= 0 or capacity.slots <= 0 or capacity.admissions_left <= 0:
                    break
                if capacity.fits(task, self.worker_reservations(task)) and fits(task):
                    append_to(capacity, task)
                    ready.remove(task)
        for task in ready:
            if not slots or len(owners) >= owner_limit:
                break
            if fits(task):
                key = self.start_group([task], "npu")
                owners.add(key)
                charge(task)
                slots -= 1

        if not placement:
            return
        if not elastic:
            self.state["dispatch_queue"] = {"policy": placement.policy, "unbound_groups": 0, "observed_unix": time.time()}
            return
        if not claimed and (not competed or slots):
            try_dispatch(protect_fixed=False)
        # The queue may contain only movable groups. Request a cooperative
        # ticket without attaching any task to a device still in the queue.
        waiting = {t["model_id"] for t in elastic if t["device"] == "unbound"
                   and len(placement.sides[t["model_id"]]) == 2 and fits({**t, "device":"npu:0"})}
        tickets = sum(bool(groups.get(key, {}).get("dispatch_lease")) and not groups[key]["task_ids"]
                      and not groups[key].get("close_requested") for key in owners)
        if slots and len(owners) < owner_limit and tickets < len(waiting):
            self.start_group([], "npu", dispatch_lease=True)

    def dynamic_observation(self, entry, observation):
        from .shared_npu3 import read_admission_chain, worker_spec_digest
        manifest = read(entry["manifest_path"])
        if sha256(entry["manifest_path"]) != entry["manifest_sha256"]:
            raise Blocked("dynamic_manifest_changed")
        chain = read_admission_chain(manifest, entry["manifest_path"], observation)
        committed = entry.get("admission_commands", [])
        if len(chain["commands"]) > len(committed):
            raise Blocked("dynamic_unregistered_admission_command")
        for published, registered in zip(chain["commands"], committed):
            if (published["sequence"] != registered["sequence"]
                    or published["sha256"] != registered["sha256"]
                    or Path(published["path"]).resolve() != Path(registered["inbox_path"]).resolve()):
                raise Blocked("dynamic_admission_ledger_mismatch")
        expected = entry.get("worker_spec_sha256", {})
        if any(expected.get(worker["id"]) != worker_spec_digest(worker)
               for worker in chain["published_workers"]):
            raise Blocked("dynamic_worker_spec_not_registered")
        if set(expected) != set(entry["task_ids"]):
            raise Blocked("dynamic_registered_worker_set_mismatch")
        return chain

    def publish_admissions(self, group_id):
        """Replay publication of committed bytes; never reserve or submit twice."""
        entry = self.state["shared_groups"][group_id]
        if entry.get("admission_aborted"):
            raise Blocked("dynamic_admission_previously_aborted")
        for command in entry.get("admission_commands", []):
            path = Path(command["outbox_path"])
            if sha256(path) != command["sha256"]:
                raise Blocked("dynamic_outbox_changed")
            target = Path(command["inbox_path"])
            if not target.exists():
                freeze_json(target, read(path))
            if sha256(target) != command["sha256"]:
                raise Blocked("dynamic_inbox_changed")

    def queue_admission(self, group_id, *, workers=(), close_reason=None):
        entry = self.state["shared_groups"][group_id]
        if entry.get("close_requested"):
            if workers:
                raise Blocked("dynamic_admission_after_close")
            self.publish_admissions(group_id)
            return
        manifest = read(entry["manifest_path"])
        commands = entry.setdefault("admission_commands", [])
        sequence = len(commands)
        dynamic = manifest["dynamic_admission"]
        if sequence >= dynamic["max_commands"]:
            raise Blocked("dynamic_command_limit")
        value = {"schema": "pilot-r3-shared-admission-v1", "sequence": sequence,
            "previous_sha256": commands[-1]["sha256"] if commands else entry["manifest_sha256"],
            "manifest_sha256": entry["manifest_sha256"], "job_id": entry["job_id"],
            "deadline_unix": manifest["deadline_unix"], "kind": "admit" if workers else "close"}
        if workers:
            value["workers"] = list(workers)
        else:
            value["reason"] = close_reason or "stage_complete"
        outbox = Path(entry["path"])/"outbox"/f"{sequence:08d}.json"
        inbox = Path(dynamic["inbox_dir"])/outbox.name
        freeze_json(outbox, value)
        commands.append({"sequence": sequence, "outbox_path": str(outbox), "inbox_path": str(inbox),
                         "sha256": sha256(outbox), "kind": value["kind"],
                         "worker_ids": [worker["id"] for worker in workers]})
        if not workers:
            entry["close_requested"] = value["reason"]
        # Attempts, budgets, owner, command identity and all member identities
        # reach durable state before the owner can observe this command.
        self._save()
        self.publish_admissions(group_id)

    def append_group(self, group_id, tasks):
        from .shared_npu3 import worker_spec_digest
        entry = self.state["shared_groups"][group_id]
        if not entry.get("dynamic_admission") or entry.get("close_requested") or entry.get("release_verified"):
            raise Blocked("dynamic_owner_not_open")
        if any(task.get("purpose") == "capacity_calibration" or task.get("phase") == "preflight" for task in tasks):
            raise Blocked("dynamic_capacity_probe_not_exclusive")
        observation = self.group_observation(group_id)
        if (observation.get("status") != "running" or observation.get("allocation_acquired_unix") is None
                or observation.get("admission", {}).get("closed") or not self.owner_running(group_id)):
            raise Blocked("dynamic_owner_not_ready_for_admission")
        manifest = read(entry["manifest_path"])
        if len(entry["task_ids"])+len(tasks) > manifest["dynamic_admission"]["max_total_workers"]:
            raise Blocked("dynamic_total_worker_limit")
        workers, plans, originals = self.worker_plans(tasks, "npu")
        try:
            for task, record, *_ in plans:
                self.reserve_task(task, record)
        except BaseException:
            for task, record, *_ in plans:
                record.clear()
                record.update(originals[task["id"]])
            raise
        started = time.time()
        sequence = len(entry.get("admission_commands", []))
        inbox = Path(manifest["dynamic_admission"]["inbox_dir"])/f"{sequence:08d}.json"
        try:
            for worker, (task, record, attempt, task_path, command, resources) in zip(workers, plans):
                attempt.mkdir(parents=True, exist_ok=False)
                freeze_json(task_path, task)
                worker.update(task_path=str(task_path), task_sha256=sha256(task_path))
                record.update(status="running", reason=None, task_path=str(task_path), shared_group=group_id,
                    process=entry.get("process"), purpose=task.get("purpose"),
                    reserved_hbm_bytes=worker["reserved_hbm_bytes"], **resources)
                record["attempts"].append({"path": str(attempt), "shared_group": group_id,
                    "group_path": entry["path"], "command": command, "started_unix": started,
                    "status": "admitting", "device_seconds": 0., "process": entry.get("process"),
                    "admission_sequence": sequence, "admission_inbox_path": str(inbox)})
                entry["task_ids"].append(task["id"])
                entry.setdefault("worker_spec_sha256", {})[task["id"]] = worker_spec_digest(worker)
            self.queue_admission(group_id, workers=workers)
        except BaseException as error:
            published_or_uncertain = inbox.exists()
            entry["admission_aborted"] = {"sequence": sequence, "inbox_path": str(inbox),
                "publication": "published_or_unverified" if published_or_uncertain else "not_published",
                "error": f"{type(error).__name__}: {error}"}
            for task, record, attempt, task_path, command, resources in plans:
                if not published_or_uncertain:
                    # Restore exactly the reservation before this attempted
                    # admission, including any imported or prior-attempt cost.
                    for field in ("preparation_reserved_tokens", "evaluation_reserved_tokens", "evaluation_budget_key"):
                        if field in originals[task["id"]]:
                            record[field] = originals[task["id"]][field]
                        else:
                            record.pop(field, None)
                    if len(record.get("attempts", [])) == len(originals[task["id"]].get("attempts", [])):
                        record.setdefault("attempts", []).append({"path": str(attempt), "shared_group": group_id,
                            "group_path": entry["path"], "command": command, "started_unix": started,
                            "admission_sequence": sequence, "admission_inbox_path": str(inbox)})
                    record.update(status="failed", reason="dynamic_admission_not_published",
                                  observation_error=entry["admission_aborted"]["error"])
                    record["attempts"][-1].update(status="not_started", admission_not_published=True,
                        finished_unix=time.time(), reason=entry["admission_aborted"]["error"])
                else:
                    record["attempts"][-1]["admission_publication"] = "published_or_unverified"
            try:
                self._save()
            except Exception:
                pass
            raise

    def owner_running(self, group_id):
        entry = self.state["shared_groups"][group_id]
        process = self.processes.get(group_id)
        return bool(live(entry.get("process")) or process is not None and process.poll() is None)

    def reap_owner(self, group_id):
        process = self.processes.pop(group_id, None)
        if process is not None:
            self.state["shared_groups"][group_id]["owner_return_code"] = process.wait()
        handle = self.handles.pop(group_id, None)
        if handle is not None:
            handle.close()

    def supervision_tick(self):
        if not self.dynamic_enabled():
            return
        self._group_observation_cache = {}
        for group_id, entry in list(self.state.get("shared_groups", {}).items()):
            if not entry.get("dynamic_admission"):
                continue
            self.publish_admissions(group_id)
            group, running = self.observe_owner(group_id)
            if entry.get("observation_error") or group.get("status") == "failed" or group.get("reason"):
                raise Blocked(f"dynamic_owner_failed:{group_id}:{group.get('reason') or entry.get('observation_error')}")
            if group.get("queue_release_error") or group.get("cleanup_observation_error"):
                raise Blocked(f"dynamic_owner_cleanup_failed:{group_id}")
            if running:
                if entry.get("dispatch_lease") and not entry["task_ids"] and not entry.get("close_requested"):
                    demand = any(t["device"] == "unbound" and t["status"] == "pending"
                                 for t in self.state["tasks"].values())
                    if not demand:
                        self.queue_admission(group_id, close_reason="dispatch_demand_satisfied")
                    elif group.get("allocation_acquired_unix") is None:
                        continue
                if group.get("capacity_deferral"):
                    self.queue_admission(group_id, close_reason="hbm_admission_deferred")
                elif not any(row["status"] == "running" and row.get("shared_group") == group_id
                           for row in self.state["tasks"].values()):
                    idle = entry.setdefault("idle_since_unix", time.time())
                    if time.time()-idle >= self.config.get("dynamic_owner_idle_seconds", 30.):
                        self.queue_admission(group_id, close_reason="idle_without_ready_work")
                else:
                    entry.pop("idle_since_unix", None)
                continue
            reason = self.group_release_error(group_id, group)
            if reason:
                raise Blocked(f"dynamic_owner_release_failed:{group_id}:{reason}")
            entry["release_verified"] = True

    def stage_drained(self):
        if not self.dynamic_enabled():
            return True
        for group_id, entry in list(self.state.get("shared_groups", {}).items()):
            if entry.get("dynamic_admission") and not entry.get("release_verified"):
                self.queue_admission(group_id, close_reason="stage_complete")
        self.supervision_tick()
        drained = all(not entry.get("dynamic_admission") or entry.get("release_verified")
                      for entry in self.state.get("shared_groups", {}).values())
        if drained:
            # A bounded cache saves historical I/O during dispatch. Crossing a
            # stage boundary always reopens all owner/queue/manifest evidence.
            self._group_observation_cache, self._terminal_group_cache = {}, {}
            for group_id, entry in self.state.get("shared_groups", {}).items():
                group = self.group_observation(group_id)
                reason = self.group_release_error(group_id, group,
                    require_completed=bool(entry.get("dynamic_admission")))
                if reason:
                    raise Blocked(f"stage_owner_release_failed:{group_id}:{reason}")
        return drained

    def dynamic_worker_result(self, task, record, group):
        """A task is reusable after its own verified exit and descendant cleanup."""
        from .resume3 import verify_adapter_result
        entry = self.state["shared_groups"][record["shared_group"]]
        worker = group.get("workers", {}).get(task["id"], {})
        if worker.get("status") in {None, "pending", "starting", "running", "exited_waiting_cleanup"}:
            return None
        if (worker.get("cleanup_verified") is not True or worker.get("descendant_cleanup_verified") is not True
                or worker.get("remaining_descendants") != [] or not worker.get("cleanup_verified_unix")
                or not isinstance(worker.get("process"), dict)
                or any(type(worker["process"].get(field)) is not int or worker["process"][field] <= 0
                       for field in ("pid", "start_ticks"))
                or live(worker.get("process"))):
            raise Blocked("dynamic_worker_cleanup_not_verified")
        if worker.get("worker_spec_sha256") != entry["worker_spec_sha256"].get(task["id"]):
            raise Blocked("dynamic_worker_spec_changed")
        path = Path(record["attempts"][-1]["path"])/"result.json"
        if (Path(worker.get("output", "")).resolve() != path.parent.resolve()
                or not path.is_file() or sha256(path) != worker.get("result_sha256")):
            raise Blocked("dynamic_worker_result_binding_changed")
        result = read(path)
        if (not isinstance(result, dict) or result.get("task_sha256") != sha256(record["task_path"])
                or result.get("source_identity") != task.get("source_identity")):
            raise Blocked("dynamic_worker_task_identity_changed")
        status, reason = evaluate_outcome(task, result, worker.get("return_code"))
        clean_exit = worker.get("status") == "completed" and worker.get("return_code") == 0
        clean_rejection = worker.get("status") == "allowed_failure" and worker.get("return_code") == 1
        if (status == "completed" and not clean_exit
                or status == "recipe_rejected" and not (clean_exit or clean_rejection)):
            raise Blocked("dynamic_worker_terminal_exit_invalid")
        if task.get("evaluation_budget_key") and result.get("evaluation", {}).get("tokens", 0) > evaluation_bound(task):
            raise Blocked("dynamic_worker_evaluation_exceeds_declared_bound")
        verify_adapter_result(self.config, task, record, result)
        return status, reason, result, path, worker

    def poll_dynamic_worker(self, task, record):
        group_id = record["shared_group"]
        group, running = self.observe_owner(group_id)
        worker = group.get("workers", {}).get(task["id"], {})
        attempt = record["attempts"][-1]
        if worker.get("process"):
            attempt["worker_process"] = worker["process"]
        attempt["worker_wall_seconds"] = worker.get("wall_seconds", 0.)
        heartbeat = read(Path(attempt["path"])/"heartbeat.json")
        if heartbeat:
            if not isinstance(heartbeat, dict):
                raise Blocked("dynamic_worker_heartbeat_invalid")
            record["heartbeat"] = heartbeat
        record["queue"] = {"status": "running" if group.get("allocation_acquired_unix") is not None
                           and group.get("allocation_released_unix") is None else group.get("status", "queued"),
            "shared_group": group_id, "physical_ids": [group["physical_id"]] if group.get("physical_id") is not None else [],
            "accounting_scope": "physical lease charged once in shared_groups"}
        stopping = getattr(self, "_stopping", False)
        if not stopping and (group.get("status") == "failed" or group.get("reason")):
            raise Blocked(f"dynamic_owner_failed:{group_id}:{group.get('reason')}")
        if stopping:
            release_error = self.group_release_error(group_id, group, require_completed=False)
            if release_error:
                raise Blocked(release_error)
            if worker.get("status") == "not_started" and not worker.get("process") and not worker.get("started_unix"):
                attempt.update(status="not_started", finished_unix=time.time(), device_seconds=0.)
                record.update(status="failed", reason="stopped_before_worker_start", device_seconds=0.)
                return True
        deferred = self.poll_hbm_deferral(task, record, group, worker, running)
        if deferred is not None:
            return deferred
        checked = self.dynamic_worker_result(task, record, group)
        if checked is None:
            if not running:
                raise Blocked("dynamic_owner_exited_before_worker_terminal_evidence")
            return False
        status, reason, result, path, worker = checked
        attempt = record["attempts"][-1]
        attempt.update(status=status, return_code=worker["return_code"], finished_unix=worker.get("finished_unix", time.time()),
            reason=reason, worker_wall_seconds=worker.get("wall_seconds", 0.),
            allocation_unix=group.get("allocation_acquired_unix"), release_unix=None,
            worker_process=worker.get("process"), worker_cleanup_verified_unix=worker["cleanup_verified_unix"])
        record.update(status=status, reason=reason, result_path=str(path), result_sha256=sha256(path), device_seconds=0.)
        self._save()
        return True
