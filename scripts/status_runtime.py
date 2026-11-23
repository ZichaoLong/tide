"""Read the current experimental launch, its scheduler and its host observations."""
import json
from pathlib import Path
import shlex
import subprocess

from pilotlab.audit3 import cgroup_observation, json_observation, systemd_observation
from pilotlab.report3 import build_report

REPOSITORY = "tide-learning-lab"
LAUNCH = Path(".local/pilot-r3/launch.json")


def driver_processes():
    """Return driver PIDs keyed by physical device, using the site's read-only parser."""
    from pilotlab.shared_npu3 import load_queue_helper

    helper = load_queue_helper()
    mapping = helper.parse_mapping(helper.run_npu_smi("info", "-m"))
    raw = helper.run_npu_smi("info")
    if "Process id" not in raw:
        raise ValueError("npu-smi process table unavailable")
    return {card: entry.process_ids for card, entry in helper.parse_status(raw, mapping).items()}


def observe(root):
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
