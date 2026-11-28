"""Read the current experimental launch, its scheduler and its host observations."""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import time

from pilotlab.audit3 import cgroup_observation, json_observation, systemd_observation
from pilotlab.report3 import build_report

REPOSITORY = "tide-learning-lab"
LAUNCH = Path(".local/pilot-r3/launch.json")
CLOUD = Path(".local/modelarts-r3/control")


def observe_cloud(root):
    directory = root / CLOUD
    marker = directory / "payload/production.json"
    if not marker.is_file():
        return None
    spec, pointer = json_observation(marker)
    snapshot, sample = json_observation(directory / "status.json")
    unit = "tide-r3-guiyang-controller-20261128.service"
    controller = systemd_observation(unit)
    result = {"kind":"modelarts_snapshot", "sources":[str(marker), str(directory / "status.json")],
              "next_check":"maflow job list --target guiyang --json", "controller":controller,
              "pointer":pointer, "observation":sample, "records":[],
              "summary":"贵阳作业已登记，等待控制器状态快照"}
    if not spec:
        result["summary"] = "贵阳冻结清单无法校验"
        return result
    for name in spec["jobs"]:
        admission, _ = json_observation(directory / "allocations" / name / "submitted.json")
        n = (snapshot or {}).get("nodes", {}).get(name, {})
        platform, q = n.get("platform", {}), n.get("queue") or {}
        active = Counter(t.get("backend") for t in q.get("tasks", []) if t.get("state") == "running")
        result["records"].append({"job":name, "id":(admission or {}).get("id"), "phase":platform.get("phase"),
                                  "secondary_phase":platform.get("secondary_phase"), "runq_counts":q.get("counts", {}),
                                  "running_by_backend":dict(active), "queue_observed_unix":q.get("at")})
    if snapshot:
        ts = snapshot["observed_unix"]
        result.update(snapshot_observed_at=datetime.fromtimestamp(ts, timezone.utc).isoformat(),
                      snapshot_age_seconds=round(time.time()-ts), selected_subjects=snapshot.get("selected_subjects"),
                      phase=snapshot["phase"], counts=snapshot["counts"], canaries=snapshot["canaries"],
                      physical_budget=snapshot["physical_budget"], gate_blockers=snapshot["gate_blockers"],
                      budget_blockers=snapshot["budget_blockers"], errors=snapshot["errors"])
        for field in ("initial_known_tasks", "original_signals", "native_ready_tasks", "pending_candidates",
                      "preparation_budget", "drained", "finished"):
            if field in snapshot:
                result[field] = snapshot[field]
        phases = "、".join(n["secondary_phase"] or n["phase"] or "未知" for n in result["records"])
        result["summary"] = (f'贵阳 {snapshot["phase"]} · 已选 {snapshot.get("selected_subjects")}/344'
                             f' · 作业 {phases} · 首步恢复 {len(snapshot["canaries"])}/2'
                             f' · 平台数据采于 {result["snapshot_observed_at"]}')
    return result


def observe_queue_additions(launch, result):
    from pilotlab.source_migration3 import read_bound

    manifest = launch.get("deferred_source_review")
    if not manifest:
        return
    try:
        path = Path(launch["runq_root"]) / "queue.sqlite3"
        result["sources"].extend([str(path), manifest["submissions"]["path"]])
        groups = {"signal_resumes": read_bound(launch["last_signal_admission"])["new_jobs"],
                  "deferred_source_reviews": {key: item["job"] for key, item in
                       read_bound(manifest["submissions"])["jobs"].items()}}
        stats = {}
        review_workers = Counter()
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("BEGIN")
            for group, jobs in groups.items():
                counts = Counter()
                for key, ident in jobs.items():
                    row = db.execute("""SELECT state, json_extract(spec,'$.owner'), json_extract(spec,'$.task_id'),
                        json_extract(spec,'$.purpose'), backend FROM jobs WHERE id=?""", (ident,)).fetchone()
                    review = group == "deferred_source_reviews"
                    tid = "early-review-" + key.replace("/", "-") if review else key
                    if (not row or row[1:3] != (launch["output"], tid)
                            or review != (row[3] == "deferred_source_review")):
                        raise ValueError(f"入队凭据与作业 {ident} 不一致")
                    counts[row[0]] += 1
                    if review and row[0] == "running":
                        review_workers[row[4]] += 1
                stats[group] = dict(counts)
        result["queue_additions"] = stats
        result["review_workers_by_backend"] = dict(review_workers)
        result["summary"] += (f" · 暂停接续={stats['signal_resumes']}"
                              f" · 早期主体复核作业={stats['deferred_source_reviews']}"
                              f" (CPU {review_workers['cpu']}, NPU {review_workers['npu']})")
    except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as error:
        result["summary"] += f" · 接续与复核队列不可核验 ({error})"


def driver_processes():
    """Return driver PIDs keyed by physical device, using the site's read-only parser."""
    from pilotlab.shared_npu3 import load_queue_helper

    helper = load_queue_helper()
    mapping = helper.parse_mapping(helper.run_npu_smi("info", "-m"))
    raw = helper.run_npu_smi("info")
    if "Process id" not in raw:
        raise ValueError("npu-smi process table unavailable")
    return {card: entry.process_ids for card, entry in helper.parse_status(raw, mapping).items()}


def observe_deferred_followup(root, launch, result):
    link = launch.get("deferred_followup")
    if not link:
        return
    from pilotlab.runq_devices import process_unit
    from pilotlab.runq_store import unit
    from pilotlab.scheduler3 import live
    from pilotlab.source_migration3 import read_bound

    try:
        local = (root / ".local").resolve()
        if any(not Path(link[k]["path"]).resolve().is_relative_to(local) for k in ("plan", "review")):
            raise ValueError("补跑指针位于另一工作树")
        plan = read_bound(link["plan"])
        review = read_bound(link["review"])
        directory = Path(link["output"])
        queue = Path(launch["runq_root"])
        if (not directory.resolve().is_relative_to(local)
                or plan["output"] != str(directory) or plan["review"] != link["review"] or plan["unit"] != link["unit"]
                or plan["config"]["runq_root"] != str(queue) or plan["config"]["predecessor"] != link["predecessor"]
                or link["predecessor"]["owner"] != launch["output"]):
            raise ValueError("补跑计划与启动指针不符")
        state, _ = json_observation(directory / "scheduler.json")
        if not state or state["config"] != plan["config"]:
            raise ValueError("补跑调度记录与冻结配置不符")
        gate = link["predecessor"]
        receipt = read_bound(state["npu_release"]) if state.get("npu_release") else None
        if receipt and receipt["predecessor"] != gate:
            raise ValueError("补跑放行凭据与前置作业不符")
        counts, running, cards = Counter(), Counter(), Counter()
        by_backend = {"cpu": Counter(), "npu": Counter()}
        with closing(sqlite3.connect((queue / "queue.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("BEGIN")
            pred = db.execute("""SELECT job_key,state,started,json_extract(spec,'$.owner'),json_extract(spec,'$.task_id')
                                 FROM jobs WHERE id=?""", (gate["job"],)).fetchone()
            if not pred or (pred[0], pred[3], pred[4]) != (gate["job_key"], gate["owner"], gate["task_id"]):
                raise ValueError("前置作业的队列身份不符")
            rows = db.execute("""SELECT j.id,j.generation,j.state,COALESCE(j.backend,json_extract(j.spec,'$.allowed[0]')),
                             j.pid,j.ticks,l.card FROM jobs j LEFT JOIN leases l ON j.lease=l.id
                             WHERE json_extract(j.spec,'$.owner')=?""", (str(directory),))
            for ident, generation, status, backend, pid, ticks, card in rows:
                counts[status] += 1
                by_backend[backend][status] += 1
                if (status == "running" and live({"pid": pid, "start_ticks": ticks})
                        and process_unit(pid) == unit(queue, "job", ident, generation)):
                    running[backend] += 1
                    if card is not None:
                        cards[card] += 1
        service = systemd_observation(link["unit"])
        props = service["properties"]
        identity = state["controller_identity"]
        controller_live = (props.get("ActiveState") == "active" and props.get("InvocationID") == link["invocation"]
                           and str(identity["pid"]) == props.get("MainPID") and live(identity)
                           and identity["pid"] in cgroup_observation(service)["members"])
        choices, detail = json_observation(directory / "selected.json")
        if choices is None and detail.get("status") != "missing":
            raise ValueError("无法读取补跑选方")
        waiting = sum(r["status"] == "pending" and r["device"] == "npu:0" and not r.get("runq_id")
                      for r in state["tasks"].values())
        result["deferred_followup"] = dict(output=str(directory), unit=link["unit"], controller_live=controller_live,
            status=state["status"], blocking_reason=state.get("blocking_reason"), stage_b_gate=state["stage_b_gate"],
            plan=link["plan"], review=link["review"], dispositions=review["modes"], planned_tasks=review["phase_counts"],
            selected_on_target=len(choices or {}), queue_counts=dict(counts), by_backend=by_backend,
            live_workers=dict(running), npu_workers_by_card=dict(cards), planned_ready_npu_without_queue_job=waiting,
            predecessor=dict(id=gate["job"], state=pred[1], started=pred[2]), npu_release=receipt)
        result["sources"].extend([link["plan"]["path"], link["review"]["path"], str(directory / "scheduler.json"),
                                  "systemctl --user show " + link["unit"]])
        gate_text = "已放行" if receipt else "等待启动"
        result["summary"] += (f" · 早期补跑选方={len(choices or {})}，队列={dict(counts)}"
                              f" (CPU {running['cpu']}，NPU {running['npu']}) · {gate['job']}={pred[1]}，补跑 NPU {gate_text}"
                              f" · 补跑控制器={props.get('ActiveState')}，B={state['stage_b_gate']}")
    except (OSError, KeyError, TypeError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
        result["summary"] += f" · 早期补跑不可核验 ({error})"


def observe(root):
    cloud = observe_cloud(root)
    if cloud is not None:
        return cloud
    path = root / LAUNCH
    launch, pointer = json_observation(path)
    result = {
        "kind": "unavailable", "summary": "本机启动指针不可核验",
        "sources": [str(LAUNCH)], "next_check": "cat docs/STATUS.md", "pointer": pointer,
    }
    if launch is None:
        return result
    try:
        json.dumps(launch, allow_nan=False)
    except ValueError as error:
        result["summary"] = f"启动指针含无效数值 ({error})"
        return result
    observe_round(root, launch, result)
    observe_deferred_followup(root, launch, result)
    _, after = json_observation(path)
    if after.get("sha256") != pointer["sha256"]:
        if result["kind"] == "live":
            result["kind"] = "snapshot_only"
        result["summary"] += " · 采样期间启动指针已更换，请重新查询"
        result["next_check"] = "python scripts/status.py"
    return result


def observe_round(root, launch, result):
    unit, output = launch.get("unit"), launch.get("output")
    if not isinstance(unit, str) or not unit or not isinstance(output, str) or not output:
        result["summary"] = "启动指针缺少有效的 unit 或 output"
        return
    try:
        local = (root / ".local").resolve()
        directory = (root / output).resolve()
    except (OSError, RuntimeError, ValueError) as error:
        result["summary"] = f"轮次路径不可读 ({error})"
        return
    if not directory.is_relative_to(local):
        result["summary"] = "轮次输出位于另一工作树，无法在此检出核验"
        return
    relative = Path(".local") / directory.relative_to(local)
    result["sources"].append(str(relative / "scheduler.json"))
    result["binding"] = {key: launch.get(key) for key in (
        "unit", "foundation_commit", "plan_path", "plan_sha256", "prepared_path", "current_review",
        "output", "log", "budget_ledger", "deadline_unix",
    )}
    for key in ("fresh_shannon", "temporary_card_window", "last_signal_admission", "deferred_source_review"):
        if key in launch:
            result["binding"][key] = launch[key]
    try:
        report = build_report(directory)
        controller = report["lifecycle"]["controller"]
        ledger = report["resources"]["budget_ledger"] or {}
        result["scheduler"] = {
            "status": report["status"], "stage": report["stage"], "updated_unix": report["updated_unix"],
            "blocking_reason": report["blocking_reason"], "task_counts": report["execution"]["task_counts"],
            "verified_completed_tasks": report["execution"]["verified_completed_tasks"],
            "selected_models": report["execution"]["selected_models"],
            "stage_b_gate": report["gates"]["stage_b_recorded"], "controller": controller,
            "budget": {k: ledger.get(k) for k in ("device_seconds", "max_device_seconds", "deadline_unix")},
            "observation_errors": report["observation_errors"],
        }
    except (OSError, IndexError, KeyError, ValueError, TypeError, AttributeError) as error:
        result["summary"] = f"调度记录不可核验 ({error})"
        return
    scheduler_paths = {"scheduler.json", str(directory / "scheduler.json")}
    if report["status"] == "unavailable" or any(e["path"] in scheduler_paths for e in report["observation_errors"]):
        result["summary"] = "本机调度记录缺失或不可读"
        return
    result["kind"] = "snapshot_only"
    result["next_check"] = shlex.join(["python", "-m", "pilotlab.inspect3", "--round", str(relative)])
    service = systemd_observation(unit)
    group = cgroup_observation(service)
    props = service.get("properties", {})
    recorded_service = launch.get("systemd", {})
    result.update(systemd=service, cgroup=group)
    result["sources"].extend([f"systemctl --user show {unit}", "/sys/fs/cgroup", "/proc"])
    result["summary"] = (f"{unit} · scheduler={report['status']} stage={report['stage']} "
                         f"tasks={report['execution']['task_counts']} B={report['gates']['stage_b_recorded']} "
                         f"blocker={report['blocking_reason']} "
                         f"· systemd={props.get('ActiveState', 'unknown')} cgroup={group['status']}")
    observe_queue_additions(launch, result)
    if not isinstance(recorded_service, dict):
        result["summary"] += " · 启动指针中的服务身份无效"
        return
    expected = recorded_service.get("InvocationID")
    live = (service["status"] == "observed" and props.get("LoadState") == "loaded"
            and props.get("ActiveState") == "active" and props.get("ControlGroup")
            and group["status"] == "populated" and controller.get("alive") is True
            and controller.get("same_start_ticks") is True and controller["pid"] in group["members"]
            and str(controller["pid"]) == props.get("MainPID")
            and (not expected or props.get("InvocationID") == expected))
    if not live:
        result["summary"] += " · 未绑定到本次运行的活控制器"
        return
    result["kind"] = "live"
    result["summary"] += f" · MainPID={controller['pid']} cgroup_pids={len(group['members'])}"
    result["sources"].append("npu-smi info -m / npu-smi info")
    try:
        devices = driver_processes()
        members = set(group["members"])
        if launch.get("runq_root"):
            from pilotlab.runq_devices import process_unit
            from pilotlab.runq_store import unit as worker_unit
            from pilotlab.scheduler3 import live
            state, _ = json_observation(directory / "scheduler.json")
            workers = {"cpu": [], "npu": []}
            for row in (state or {}).get("tasks", {}).values():
                if row.get("status") != "running" or not row.get("attempts"):
                    continue
                attempt = row["attempts"][-1]
                identity = attempt.get("process")
                if not live(identity):
                    continue
                name = process_unit(identity["pid"])
                q = row.get("queue") or {}
                if (q.get("job_id") != attempt.get("runq_id") or not isinstance(q.get("generation"), int)
                        or name != worker_unit(launch["runq_root"],"job",attempt["runq_id"],q["generation"])
                        or name != attempt.get("unit")):
                    continue
                workers["cpu" if row["device"] == "cpu" else "npu"].append(identity["pid"])
                members.add(identity["pid"])
            result["worker_units"] = workers
            result["summary"] += f" · Stage A CPU workers={len(workers['cpu'])} NPU workers={len(workers['npu'])}"
        by_card = {}
        for card, pids in devices.items():
            owned = members.intersection(pids)
            if owned:
                by_card[card] = sorted(owned)
        result["driver"] = {"status": "observed", "service_pids_by_physical_card": by_card}
        result["summary"] += f" · driver_cards={len(by_card)} driver_pids={len(set().union(*by_card.values()))}"
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, ImportError) as error:
        result["driver"] = {"status": "unavailable", "reason": str(error)}
        result["summary"] += " · NPU 驱动不可核验"
