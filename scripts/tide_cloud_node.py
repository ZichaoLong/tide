"""Run one independently allocated ModelArts node and its frozen TIDE tasks."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

from tide_batch import export
from tide_portable import ident, read, relative, require, sha, write


ROOT = Path(__file__).resolve().parent


def observation(path, value):
    # OBS mounts publish on close. The reader retries incomplete observations.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def deliver(path, obs, name):
    for delay in (0, 1, 3, 8, 15):
        time.sleep(delay)
        try:
            return export(path, obs, name)
        except (OSError, ValueError):
            if delay == 15:
                raise


def hydrate(area, obs, descriptor):
    base = relative(ROOT / "bases", descriptor["source"])
    supplied = descriptor.get("base_manifest")
    manifest = supplied if supplied is not None else read(base / "base.json")
    require(manifest["identity"] == descriptor["source"], "frozen base identity differs")
    target = area / "frozen"
    if (target / "freeze.json").exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    files, paths = dict(manifest["files"]), dict(manifest["paths"])
    cache = area.parents[1] / "objects"
    cache.mkdir(parents=True, exist_ok=True)

    def blob(record):
        h = record["sha256"]
        require(len(h) == 64 and all(c in "0123456789abcdef" for c in h), "invalid object digest")
        dest = cache / h
        with (cache / (h + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not dest.exists():
                src = obs / "inputs/objects" / h
                with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temp:
                    tmp = Path(temp.name)
                    try:
                        with src.open("rb") as stream:
                            shutil.copyfileobj(stream, temp, 2**20)
                        temp.flush()
                        require(tmp.stat().st_size == record["size"] and sha(tmp) == h,
                                "cloud input body differs: " + h)
                        os.replace(tmp, dest)
                    finally:
                        tmp.unlink(missing_ok=True)
            require(dest.stat().st_size == record["size"], "cached object length differs")
        return dest

    def install(name, src):
        dest = relative(target, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            try:
                os.link(src, dest)
            except OSError:
                shutil.copyfile(src, dest)

    for name, record in files.items():
        local = relative(base, name)
        install(name, local if local.is_file() else blob(record))
    for old, item in descriptor["references"].items():
        name = item["name"]
        if "bytes" in item:
            dest = relative(target, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(base64.b64decode(item["bytes"], validate=True))
            require(sha(dest) == item["sha256"], "bound auxiliary evidence differs")
        else:
            install(name, blob(item))
        paths[old] = name
        files[name] = {k: item[k] for k in ("sha256", "size")}
    key = ident(descriptor["task"])
    taskfile = relative(target, "tasks/" + key + ".json")
    taskfile.parent.mkdir(parents=True, exist_ok=True)
    taskfile.write_bytes(base64.b64decode(descriptor["task_bytes"], validate=True))
    require(sha(taskfile) == descriptor["fingerprint"], "frozen task bytes changed in transport")
    paths[descriptor["original_path"]] = "tasks/" + key + ".json"
    files[paths[descriptor["original_path"]]] = {"sha256": descriptor["fingerprint"], "size": taskfile.stat().st_size}
    if descriptor.get("resume"):
        install("runtime-endpoint.pt", blob(descriptor["resume"]))
    write(target / "freeze.json", {"schema": "tide-portable/v1", "files": files, "paths": paths,
                                   "roots": manifest["roots"], "tasks": {key: {
                                       "path": "tasks/" + key + ".json", "fingerprint": descriptor["fingerprint"]}}})
    return target


def run_one(obs, shared, envelope, checksum):
    require(sha(envelope) == checksum, "queue descriptor changed")
    d = read(envelope)
    key, attempt = ident(d["task"]), ident(d["attempt"])
    area = shared / "attempts" / attempt
    area.mkdir(parents=True, exist_ok=True)
    prefix = "products/attempts/" + attempt + "/"
    with (area / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        terminal = area / "receipt.json"
        if terminal.exists():
            receipt = read(terminal)
            require(receipt["descriptor_sha256"] == checksum, "attempt belongs to a different descriptor")
            deliver(terminal, obs, "products/receipts/" + attempt + ".json")
            return int(receipt["status"] == "failed")
        output = area / "output"
        rec = {"schema": "tide-cloud-result/v1", "task": key, "attempt": attempt,
               "fingerprint": d["fingerprint"], "descriptor_sha256": checksum,
               "job": os.environ["MAFLOW_RUN"], "device": os.environ["RUNQ_DEVICE"],
               "board": os.environ.get("RUNQ_PHYSICAL_DEVICE"), "chip": os.environ.get("RUNQ_PHYSICAL_CHIP"),
               "runq": os.environ["MAFLOW_RUNQ_COMMIT"], "started": time.time(), "artifacts": {},
               "status": "failed"}
        child = None
        try:
            frozen = hydrate(area, obs, d)
            plan = area / "runtime-plan.json"
            plan.write_bytes(base64.b64decode(d["runtime_plan_bytes"], validate=True))
            require(sha(plan) == d["runtime_plan_sha256"], "runtime plan differs")
            deadline = min(d["deadline_unix"], read(Path(os.environ["MAFLOW_WORK"]) / "queue.json")["deadline"])
            cmd = [sys.executable, "-B", "-u", str(ROOT / "tide_portable.py"), "worker", "--frozen",
                   str(frozen), "--task", key, "--fingerprint", d["fingerprint"], "--output", str(output),
                   "--device", rec["device"], "--deadline-unix", str(deadline),
                   "--runtime-plan", str(plan), "--runtime-plan-sha", d["runtime_plan_sha256"]]
            if d.get("resume"):
                cmd += ["--resume", str(frozen / "runtime-endpoint.pt"), "--resume-sha", d["resume"]["sha256"]]
            if d["canary"]:
                cmd += ["--check-first-save"]
            marker = output / "runtime-first-update.json"
            published = False
            if not (output / "project-verdict.json").is_file():
                with (area / "worker.log").open("ab") as log:
                    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                    def forward(signum, frame):
                        if child.poll() is None:
                            child.send_signal(signal.SIGTERM)
                    signal.signal(signal.SIGTERM, forward)
                    signal.signal(signal.SIGINT, forward)
                    last_progress = 0.
                    closing_sent = False
                    while child.poll() is None:
                        if d["canary"] and marker.is_file() and not published:
                            try:
                                export(marker, obs, "products/first-update/" + attempt + ".json")
                                published = True
                            except (OSError, ValueError):
                                pass
                        if time.time() >= deadline - 360 and not closing_sent:
                            forward(signal.SIGTERM, None)
                            closing_sent = True
                        if time.time() - last_progress >= 60:
                            heart = output / "heartbeat.json"
                            try:
                                if heart.is_file():
                                    sample = read(heart)
                                    observation(obs / "products/progress" / (attempt + ".json"),
                                                {"attempt": attempt, "at": time.time(), "heartbeat": sample})
                                    trained = sample["tokens"] - d["start_tokens"]
                                    if trained > 0 and sample.get("seconds", 0) > 0:
                                        remaining = math.ceil((sample["target_tokens"] - sample["tokens"] +
                                                               d["closing_tokens"]) * sample["seconds"] / trained)
                                        subprocess.run([os.environ["MAFLOW_RUNQ"], "request",
                                                        os.environ["RUNQ_JOB_ID"], "--remaining-seconds",
                                                        str(max(1, remaining + d["closing_seconds"]))],
                                                       capture_output=True, timeout=12)
                            except (OSError, ValueError, subprocess.TimeoutExpired):
                                pass
                            last_progress = time.time()
                        time.sleep(1)
                rec["exit"] = child.returncode
            if d["canary"] and marker.is_file() and not published:
                deliver(marker, obs, "products/first-update/" + attempt + ".json")
            if (output / "project-verdict.json").is_file():
                v = read(output / "project-verdict.json")
                require(v["fingerprint"] == d["fingerprint"] and v["task"] == key, "wrong worker verdict")
                rec["status"] = v["status"]
            for name in ("result.json", "project-verdict.json", "runtime-first-update.json", "steps.jsonl",
                         "checkpoint.json", "health.jsonl", "work.jsonl", "validation.jsonl",
                         "startup-memory.json", "failure.json"):
                if (output / name).is_file():
                    rec["artifacts"][name] = deliver(output / name, obs, prefix + name)
            if (output / "result.json").exists():
                result = read(output / "result.json")
                cp = result.get("checkpoint") or {}
                if cp:
                    path = Path(cp["path"]).resolve()
                    require(path.is_relative_to(output.resolve()) and sha(path) == cp["sha256"],
                            "worker published an invalid checkpoint")
                    rec["artifacts"]["checkpoint.pt"] = deliver(path, obs, prefix + "checkpoint.pt")
                capture = result.get("failure_capture") or {}
                if capture.get("path") and Path(capture["path"]).is_file():
                    path = Path(capture["path"]).resolve()
                    require(path.is_relative_to(output.resolve()), "failure witness outside this attempt")
                    rec["artifacts"]["failure.pt"] = deliver(path, obs, prefix + "failure.pt")
        except Exception:
            rec.update(status="failed", error=traceback.format_exc())
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
        if (area / "worker.log").is_file():
            rec["artifacts"]["worker.log"] = deliver(area / "worker.log", obs, prefix + "worker.log")
        rec["finished"] = time.time()
        write(terminal, rec)
        deliver(terminal, obs, "products/receipts/" + attempt + ".json")
        return int(rec["status"] == "failed")


def pending_dispatches(obs, enrolled):
    pending = []
    try:
        requests = list((obs / "inputs/dispatch").iterdir())
    except FileNotFoundError:
        return pending
    except OSError as exc:
        print("dispatch listing unavailable", exc, file=sys.stderr, flush=True)
        return pending
    for request in requests:
        if request.suffix != ".json" or request.name in enrolled:
            continue
        try:
            body = request.read_bytes()
            d = json.loads(body)
            require(isinstance(d, dict) and ident(d["attempt"]) == request.stem and d["job"] == os.environ["MAFLOW_RUN"],
                    "dispatch identity differs")
            priority = d.get("priority")
            pending.append((priority if type(priority) is int else 0, request, d, hashlib.sha256(body).hexdigest()))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print("dispatch not readable", request.name, exc, file=sys.stderr, flush=True)
    return sorted(pending, key=lambda x: (-x[0], x[1].name))


def enroll(request, d, checksum, call, obs, shared, release):
    try:
        require(d["backend"] in ("npu", "cpu") and type(d["priority"]) is int
                and all(type(d[k]) is int and d[k] > 0 for k in ("threads", "ram_mib")), "invalid worker resources")
        opts = ["--worker", 0, "--key", d["attempt"], "--cwd", ROOT, "--backend", d["backend"],
                "--threads", d["threads"], "--ram-mib", d["ram_mib"], "--priority", d["priority"]]
        if d.get("estimated_seconds") is not None:
            require(type(d["estimated_seconds"]) is int and d["estimated_seconds"] > 0, "invalid runtime estimate")
            opts.extend(["--estimated-seconds", d["estimated_seconds"]])
        if d["backend"] == "npu":
            require(type(d["hbm_mib"]) is int and d["hbm_mib"] > 0 and type(d["reserve_chip"]) is bool,
                    "invalid NPU resources")
            opts.extend(["--hbm-mib", d["hbm_mib"]])
            if d["reserve_chip"]:
                opts.append("--reserve-chip")
    except (ValueError, KeyError, TypeError) as exc:
        receipt = {"schema": "tide-cloud-result/v1", "task": d.get("task"), "attempt": request.stem,
                   "fingerprint": d.get("fingerprint"), "descriptor_sha256": checksum, "job": d["job"],
                   "runq": release["runq_commit"], "status": "failed", "error": repr(exc), "artifacts": {}}
        local = shared / "rejections" / request.name
        # Once sealed, retries after an OBS outage publish the same bytes.
        if local.is_file():
            require(read(local)["descriptor_sha256"] == checksum, "conflicting prior rejection")
        else:
            write(local, {**receipt, "finished": time.time()})
        export(local, obs, "products/receipts/" + request.name)
        return
    result = call("submit", *opts, "--", sys.executable, "-B", "-u", str(ROOT / "tide_cloud_node.py"),
                  "worker", str(request), checksum)
    require(isinstance(result, dict) and result.get("key") == d["attempt"] and result.get("worker") == 0,
            "runq has not acknowledged this attempt")


def serve(obs, shared):
    exe = os.environ["MAFLOW_RUNQ"]
    release = read(ROOT / "production.json")
    require(release["runq_commit"] == os.environ["MAFLOW_RUNQ_COMMIT"]
            and release["runq_sha256"] == os.environ["MAFLOW_RUNQ_SHA256"], "unexpected runq release")
    work = Path(os.environ["MAFLOW_WORK"])
    deadline = min(release["deadline_unix"], read(work / "queue.json")["deadline"])
    queue = shared / "runq"
    queue.mkdir(mode=0o700, parents=True, exist_ok=True)
    write(Path(os.environ["MAFLOW_RUNQ_QUEUE_FILE"]), {"mode": "project", "shared": str(queue), "deadline": deadline})
    ctl = [exe, "--json", "cluster"]
    def call(action, *args):
        cmd = [*ctl, action, "--shared", str(queue), *map(str, args)]
        return json.loads(subprocess.check_output(cmd, text=True, timeout=30))
    with (shared / "runq-agent.log").open("ab") as log:
        agent = subprocess.Popen([exe, "--json", "--root", str(work / "tide-runq-state"), "cluster", "serve",
                                  "--shared", str(queue), "--deadline-unix", str(deadline)], stdout=log,
                                 stderr=subprocess.STDOUT)
    enrolled = set()
    try:
        last_heartbeat = 0.
        while time.time() < deadline - 90 and agent.poll() is None:
            try:
                if (obs / "control" / "drain-tide.json").exists():
                    break
                registered = (queue / ".runq-cluster" / "allocation.json").exists()
            except OSError as exc:
                print("control mount unavailable", exc, file=sys.stderr, flush=True)
                registered = False
            if not registered:
                time.sleep(1)
                continue
            pending = pending_dispatches(obs, enrolled)
            for _, request, d, checksum in pending:
                try:
                    enroll(request, d, checksum, call, obs, shared, release)
                    enrolled.add(request.name)
                except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                    print("retrying dispatch", request.name, exc, file=sys.stderr, flush=True)
            if time.time() - last_heartbeat >= 15 or pending:
                try:
                    state = call("status")
                    counts = {}
                    for task in state["tasks"]:
                        counts[task["state"]] = counts.get(task["state"], 0) + 1
                    observation(obs / "products/tide-status.json", {"at": time.time(), "counts": counts,
                                "workers": state["workers"], "tasks": state["tasks"], "deadline": deadline,
                                "job": os.environ["MAFLOW_RUN"], "runq": os.environ["MAFLOW_RUNQ_COMMIT"]})
                    last_heartbeat = time.time()
                except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                    print("queue heartbeat unavailable", exc, file=sys.stderr, flush=True)
            time.sleep(3)
        while agent.poll() is None and time.time() < deadline:
            try:
                call("finish")
                break
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                print("retrying graceful drain", exc, file=sys.stderr, flush=True)
                time.sleep(1)
        code = agent.wait(timeout=max(15, deadline - time.time()))
        receipt = shared / "queue-receipt.json"
        write(receipt, {"job": os.environ["MAFLOW_RUN"], "at": time.time(), "exit": code,
                        "runq": os.environ["MAFLOW_RUNQ_COMMIT"], "queue": call("status")})
        deliver(receipt, obs, "products/queue-receipt.json")
        return code
    finally:
        if agent.poll() is None:
            agent.terminate()
            agent.wait(timeout=30)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("serve", "worker"))
    cli.add_argument("envelope", nargs="?", type=Path)
    cli.add_argument("checksum", nargs="?")
    args = cli.parse_args()
    obs = Path(os.environ["MAFLOW_OBS"])
    shared = Path(os.environ["MAFLOW_SHARED"]) / "tide-r3"
    shared.mkdir(mode=0o700, parents=True, exist_ok=True)
    sys.exit(serve(obs, shared) if args.action == "serve" else run_one(obs, shared, args.envelope, args.checksum))
