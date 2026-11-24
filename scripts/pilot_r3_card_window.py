"""Temporarily refill extra physical cards in a Stage A project queue."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from pilotlab.audit3 import cgroup_observation, systemd_observation
from pilotlab.io import atomic_json, sha256
from pilotlab.runq_devices import observe, process_unit
from pilotlab.runq_store import connect, lookup, put, setting, unit, wake, write
from pilotlab.scheduler import proc_identity


ACTIVE = "('reserved','starting','running')"


def read(path):
    return json.loads(Path(path).read_bytes())


def jobs_on_cards(db, plan, since):
    cards = plan["extra_cards"]
    marks = ",".join("?" for _ in cards)
    return [dict(row) for row in db.execute(f"""
        SELECT j.id, j.generation, j.lease, l.card, j.pid, j.ticks FROM jobs j
        JOIN leases l ON l.id=j.lease WHERE l.card IN ({marks}) AND j.state IN {ACTIVE}
        AND json_extract(j.spec,'$.owner')=? AND j.manual=0 AND j.assigned>=?
        ORDER BY j.id""", (*cards, plan["owner"], since))]


def arm_timer(path, plan, early_deadline):
    # The callback reads the exact committed deadline. Arm first so a failed
    # timer installation cannot leave the queue expanded.
    date = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(early_deadline))
    name = plan["release_unit"]
    log = path.parent / "release.log"
    subprocess.run(["systemd-run", "--user", "--unit=" + name, "--on-calendar=" + date,
                    "--timer-property=AccuracySec=1ms", "--slice=background.slice",
                    "--property=Restart=on-failure", "--property=RestartSec=15s", "--property=RestartPreventExitStatus=3",
                    "--property=StandardOutput=append:" + str(log),
                    "--property=StandardError=append:" + str(log),
                    "--working-directory=" + plan["source"], "/usr/bin/env",
                    "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=" + plan["source"],
                    sys.executable, str(Path(__file__).resolve()), "release", str(path)], check=True)
    subprocess.run(["systemctl", "--user", "is-active", "--quiet", name + ".timer"], check=True)


def activate(db, path, plan, key):
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("use the system Python with Linux pidfd support to arm the release")
    with write(db):
        previous = setting(db, key)
        if previous:
            return previous
        limits = setting(db, "limits")
        base, extra = set(plan["base_cards"]), set(plan["extra_cards"])
        if (not extra or base & extra or set(limits["auto_card_ids"]) != base
                or limits["cards"] != len(base) or plan["seconds"] <= 0):
            raise ValueError("automatic device policy changed before activation")
        marks = ",".join("?" for _ in extra)
        if db.execute(f"SELECT 1 FROM leases WHERE card IN ({marks}) AND state IN ('opening','held','closing')",
                      tuple(extra)).fetchone():
            raise ValueError("temporary card already has a project lease")
        arm_timer(path, plan, time.time() + plan["seconds"])
        start = time.time()
        window = {"started": start, "deadline": start + plan["seconds"], "before": limits}
        put(db, key, window)
        put(db, "limits", {**limits, "cards": len(base | extra), "auto_card_ids": sorted(base | extra)})
    wake(Path(plan["queue"]))
    atomic_json(path.parent / "activation.json", window)
    return window


def close_refill(db, plan, key):
    with write(db):
        window = setting(db, key)
        if "targets" not in window:
            limits = setting(db, "limits")
            extra = set(plan["extra_cards"])
            ids = limits.get("auto_card_ids")
            if ids is None:
                ids = [int(k) for k in setting(db, "samples")[-1]["cards"]]
            keep = sorted(set(ids) - extra)
            grants = {k: v for k, v in limits.get("auto_share_grants", {}).items() if int(k) not in extra}
            put(db, "limits", {**limits, "cards": min(limits["cards"], len(keep)),
                               "auto_card_ids": keep, "auto_share_grants": grants})
            window.update(closed=time.time(), targets=jobs_on_cards(db, plan, window["started"]))
            put(db, key, window)
    wake(Path(plan["queue"]))
    return window


def same_placement(job, target):
    return (job["generation"], job["lease"]) == (target["generation"], target["lease"])


def signal_ready(job, expected_unit):
    """Send TERM to the workload after its checkpoint handler is installed."""
    pid = job["pid"]
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return False
    try:
        identity = proc_identity(pid)
        if (not identity or identity["start_ticks"] != job["ticks"] or process_unit(pid) != expected_unit):
            return False
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        caught = int(next(line.split()[1] for line in status if line.startswith("SigCgt:")), 16)
        if not (caught & (1 << (signal.SIGTERM - 1))):
            return False
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False
    finally:
        os.close(fd)


def verify_result(job):
    command = json.loads(job["command"])
    argv = command["argv"]
    task_path = Path(argv[argv.index("--task") + 1])
    output = Path(argv[argv.index("--output") + 1])
    result_path = output / "result.json"
    task, result = read(task_path), read(result_path)
    if (result["id"] != task["id"] or result["task_sha256"] != sha256(task_path)
            or result["source_identity"] != task["source_identity"]):
        raise ValueError("terminal receipt differs from the dispatched task")
    receipt = {"task": str(task_path), "task_sha256": result["task_sha256"],
               "result": str(result_path), "result_sha256": sha256(result_path),
               "status": result["status"], "reason": result.get("stop_reason"), "return_code": job["rc"]}
    if result["status"] == "budget_exhausted" and result["stop_reason"] == "signal":
        cp = result["checkpoint"]
        if (job["rc"] != 2 or read(output / "checkpoint.json")["current"] != cp
                or not cp.get("state_sha256") or Path(cp["path"]).stat().st_size != cp["size_bytes"]
                or sha256(cp["path"]) != cp["sha256"]
                or any(cp[k] != result["training"][k] for k in ("steps", "tokens"))):
            raise ValueError("signal interruption lacks its verified final checkpoint")
        receipt["checkpoint"] = cp
    return receipt


DECODE = """
import json, sys
from pathlib import Path
from pilotlab.worker3 import _read_checkpoint, validate_checkpoint
errors = {}
for ident, receipt in json.load(sys.stdin).items():
    try:
        cp = receipt['checkpoint']
        task = json.loads(Path(receipt['task']).read_bytes())
        payload = _read_checkpoint(Path(cp['path']), cp['sha256'])
        validate_checkpoint(payload, task, task['device'])
        if any(payload[k] != cp[k] for k in ('steps', 'tokens', 'state_sha256')):
            raise ValueError('checkpoint payload differs from receipt')
        del payload
    except Exception as error:
        errors[ident] = str(error)
print(json.dumps(errors))
"""


def release(db, path, plan, key):
    window = setting(db, key)
    if window is None:
        return  # Timer was armed but the activation transaction never committed.
    time.sleep(max(0, window["deadline"] - time.time()))
    window = close_refill(db, plan, key)
    directory = path.parent
    atomic_json(directory / "cutoff.json", window)
    root = Path(plan["queue"])
    signalled = read(directory / "signals.json") if (directory / "signals.json").exists() else {}
    until = time.monotonic() + 900
    while True:
        pending = []
        for target in window["targets"]:
            job = lookup(db, target["id"])
            if not same_placement(job, target) or job["state"] not in ("reserved", "starting", "running"):
                continue
            pending.append(job["id"])
            name = unit(root, "job", job["id"], job["generation"])
            if job["state"] == "running" and str(job["id"]) not in signalled and signal_ready(job, name):
                signalled[str(job["id"])] = {"pid": job["pid"], "ticks": job["ticks"], "at": time.time()}
                atomic_json(directory / "signals.json", signalled)
        leases = [dict(row) for row in db.execute(
            "SELECT id,card,state FROM leases WHERE created>=? AND role IN ('auto','drain') "
            "AND state IN ('opening','held','closing')", (window["started"],)) if row["card"] in plan["extra_cards"]]
        if not pending and not leases:
            break
        if time.monotonic() >= until:
            raise RuntimeError(f"release still in progress: jobs={pending}, leases={leases}")
        time.sleep(2)
    sample = observe()
    receipts, issues = {}, {}
    decoders = defaultdict(dict)
    for target in window["targets"]:
        job = lookup(db, target["id"])
        name = unit(root, "job", target["id"], target["generation"])
        if (not cgroup_observation(systemd_observation(name))["released"]
                or any(p["unit"] == name for card in sample["cards"].values() for p in card["processes"])):
            raise RuntimeError("worker is still releasing its unit or driver allocation: " + name)
        if same_placement(job, target) and job["started"]:
            try:
                receipt = verify_result(job)
                receipts[job["id"]] = receipt
                if receipt.get("checkpoint"):
                    command = json.loads(job["command"])
                    decoder = (command["argv"][0], command["cwd"], command["env"]["PYTHONPATH"])
                    decoders[decoder][job["id"]] = receipt
                if not (receipt.get("checkpoint") or (receipt["status"] == "completed" and job["rc"] == 0)):
                    issues[job["id"]] = "independent terminal outcome: " + receipt["status"]
            except (OSError, ValueError, KeyError) as error:
                issues[job["id"]] = str(error)
        else:
            receipts[job["id"]] = {"status": "left placement before execution"}
    report = {"finished": time.time(), "cutoff": window["closed"], "targets": window["targets"],
              "limits": setting(db, "limits"), "driver_sample": sample, "receipts": receipts, "issues": issues}
    atomic_json(directory / "device-release.json", report)
    for (python, source, pythonpath), group in decoders.items():
        env = {**os.environ, **plan["decode_env"], "PYTHONPATH": pythonpath, "ASCEND_RT_VISIBLE_DEVICES": "",
               "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        result = subprocess.run([python, "-c", DECODE], cwd=source, env=env, input=json.dumps(group),
                                capture_output=True, text=True, check=True)
        errors = json.loads(result.stdout.strip().splitlines()[-1])
        issues.update(errors)
        for ident in group:
            receipts[ident]["checkpoint_decoded"] = str(ident) not in errors
    report["verified"] = time.time()
    atomic_json(directory / "release.json", report)
    if issues:
        raise SystemExit(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("activate", "release"))
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    path = args.plan.resolve()
    plan = read(path)
    root = Path(plan["queue"])
    if not (root / "queue.sqlite3").is_file():
        raise ValueError("the specified project queue does not exist")
    db = connect(root)
    key = "card_window." + sha256(path)
    if args.action == "activate":
        print(json.dumps(activate(db, path, plan, key)))
    else:
        release(db, path, plan, key)


if __name__ == "__main__":
    main()
