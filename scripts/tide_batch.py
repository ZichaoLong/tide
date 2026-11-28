"""Run one maflow attempt through the per-instance runq queues, or reconcile its products."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

from tide_portable import Frozen, ident, read, relative, require, sha, write


def export(path, obs, name):
    destination = relative(obs, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            md5.update(block)
    expected = digest.hexdigest()
    if not destination.exists():
        shutil.copyfile(path, destination)
    require(sha(destination) == expected, "published artifact differs: " + name)
    return {"path": name, "size": destination.stat().st_size, "sha256": expected,
            "content_md5": md5.hexdigest()}


def run_local(root, attempt):
    """The account queue already owns the CPU/device and declared group resources."""
    require(
        os.environ.get("RUNQ_JOB_ID") and os.environ.get("RUNQ_DEVICE"),
        "local attempts require an assigned runq slot",
    )
    failed = False
    for spec in attempt["tasks"]:
        require(
            spec["resources"]["backend"] == os.environ["RUNQ_DEVICE"].split(":")[0],
            "task backend differs from group",
        )
        failed |= bool(task_main(root, spec["id"]))
    return int(failed)


def task_main(root, key):
    attempt = read(root / "maflow-attempt.json")
    spec = next(t for t in attempt["tasks"] if t["id"] == key)
    attempt_id = ident(attempt["attempt"])
    obs = Path(os.environ["MAFLOW_OBS"]).resolve()
    base = Path(os.environ["MAFLOW_SHARED"]).resolve() / "tide" / attempt_id
    directory = base / "work" / ident(key)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    products = "products/tide/" + key + "/" + attempt_id + "/"
    verdict_path = directory / "attempt-result.json"
    with (directory / "attempt.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if verdict_path.exists():
            rec = read(verdict_path)
            require(
                (rec["attempt"], rec["fingerprint"], rec["plan_sha256"])
                == (attempt_id, spec["fingerprint"], attempt["plan_sha256"]),
                "attempt replay conflict",
            )
            export(verdict_path, obs, "products/tasks/" + key + ".json")
            return int(rec["status"] == "failed")
        rec = {
            "schema": "maflow-task-result/v1",
            "batch": attempt["batch"],
            "project": attempt["project"],
            "task": key,
            "attempt": attempt_id,
            "plan_sha256": attempt["plan_sha256"],
            "fingerprint": spec["fingerprint"],
            "status": "failed",
            "artifacts": {},
            "started": time.time(),
            "worker": int(os.environ.get("VC_TASK_INDEX", "0")),
            "queue_id": os.environ.get("RUNQ_JOB_ID"),
            "physical_board": os.environ.get("RUNQ_PHYSICAL_DEVICE"),
            "physical_chip": os.environ.get("RUNQ_PHYSICAL_CHIP"),
            "work": str(directory),
            "resume_from": attempt.get("resume", {}).get(key),
        }
        try:
            output = directory / "worker"
            cmd = [
                sys.executable,
                "-B",
                str(root / "tide_portable.py"),
                "worker",
                "--frozen",
                str(root / "frozen"),
                "--task",
                key,
                "--fingerprint",
                spec["fingerprint"],
                "--output",
                str(output),
                "--device",
                os.environ["RUNQ_DEVICE"],
                "--deadline-unix",
                str(time.time() + spec.get("worker_seconds", 500)),
            ]
            source = rec["resume_from"]
            if source:
                incoming = relative(obs, source["path"])
                local = directory / "resume.pt"
                shutil.copyfile(incoming, local)
                require(
                    local.stat().st_size == source["size"] and sha(local) == source["sha256"],
                    "resume transfer differs",
                )
                cmd += ["--resume", str(local), "--resume-sha", source["sha256"]]
            elif spec.get("interrupt_after_step"):
                cmd += ["--stop-after-step", str(spec["interrupt_after_step"])]
            if (output / "project-verdict.json").exists():
                # A prior wrapper can exit between a durable verdict and the final export.
                previous = read(output / "project-verdict.json")
                require(previous["fingerprint"] == spec["fingerprint"], "previous local verdict conflicts")
                rec["worker_exit"] = int(previous["status"] == "failed")
            else:
                with (directory / "worker.log").open("ab") as log:
                    worker = subprocess.run(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=spec.get("worker_seconds", 500) + 35,
                    )
                rec["worker_exit"] = worker.returncode
            # Transport immutable evidence before publishing the task receipt.
            for kind, name in (
                ("result", "result.json"),
                ("verdict", "project-verdict.json"),
                ("steps", "steps.jsonl"),
                ("checkpoint_index", "checkpoint.json"),
                ("stop_boundary", "intentional-stop.json"),
                ("health", "health.jsonl"),
                ("work_observations", "work.jsonl"),
            ):
                if (output / name).is_file():
                    rec["artifacts"][kind] = export(output / name, obs, products + name)
            rec["artifacts"]["log"] = export(directory / "worker.log", obs, products + "worker.log")
            verdict = read(output / "project-verdict.json")
            require(
                verdict["fingerprint"] == spec["fingerprint"] and verdict["task"] == key,
                "project verdict identity",
            )
            checkpoint = verdict.get("checkpoint")
            if checkpoint:
                source_path = Path(checkpoint["path"]).resolve()
                require(
                    source_path.parent == (output / "checkpoints").resolve(), "unexpected checkpoint location"
                )
                artifact = export(source_path, obs, products + "checkpoint.pt")
                require(artifact["sha256"] == checkpoint["sha256"], "worker checkpoint differs")
                rec["artifacts"]["checkpoint"] = {
                    **artifact,
                    "step": checkpoint["steps"],
                    "tokens": checkpoint["tokens"],
                    "state_sha256": checkpoint["state_sha256"],
                    "original_path": str(source_path),
                }
            if source:
                require(
                    verdict.get("resume", {}).get("from_steps") == source["step"]
                    and verdict["resume"]["sha256"] == source["sha256"],
                    "step or chosen generation differs",
                )
            require(
                rec["worker_exit"] == 0 and verdict["status"] in {"interrupted", "completed"},
                "project verdict failed",
            )
            rec.update(
                status=verdict["status"],
                steps=verdict["steps"],
                tokens=verdict["tokens"],
                worker_pid=verdict["worker_pid"],
                worker_ticks=verdict["ticks"],
            )
        except Exception:
            rec["error"] = traceback.format_exc()
        rec["finished"] = time.time()
        write(verdict_path, rec)
        export(verdict_path, obs, "products/tasks/" + key + ".json")
        return int(rec["status"] == "failed")


def cluster(root):
    a = read(root / "maflow-attempt.json")
    require(a["schema"] == "maflow-attempt/v1" and a["tasks"], "invalid maflow attempt")
    if a.get("site") == "local":
        return run_local(root, a)
    rank = int(os.environ.get("VC_TASK_INDEX", "0"))
    count = int(os.environ.get("VC_WORKER_NUM", "1"))
    attempt = ident(a["attempt"])
    area = Path(os.environ["MAFLOW_SHARED"]).resolve() / "tide" / attempt
    area.mkdir(mode=0o700, parents=True, exist_ok=True)
    queue = area / "queue"
    release = read(root / "release.json")
    duration = release["resume_run_seconds"] if a["number"] > 1 else release["run_seconds"]
    budget = min(duration - 25, int(os.environ.get("TIDE_QUEUE_SECONDS", str(duration - 25))))
    require(budget > 0, "project queue budget must fit the job")
    deadline = time.time() + budget
    obs = Path(os.environ["MAFLOW_OBS"]).resolve()
    runtime = release["runq"]
    require(
        (os.environ["MAFLOW_RUNQ_COMMIT"], os.environ["MAFLOW_RUNQ_SHA256"])
        == (runtime["commit"], runtime["sha256"]),
        "cloud runq does not match the frozen project release",
    )
    executable = Path(os.environ["MAFLOW_RUNQ"])
    require(executable.is_absolute() and os.access(executable, os.X_OK), "cloud runq is not executable")
    ctl = [str(executable), "--json"]
    write(Path(os.environ["MAFLOW_RUNQ_QUEUE_FILE"]), {"mode": "project", "shared": str(queue), "deadline": deadline})
    ent = area / f"node-{rank}.json"
    write(
        ent,
        {
            "attempt": attempt,
            "rank": rank,
            "root": str(root),
            "python": sys.executable,
            "runq": runtime,
            "pid": os.getpid(),
            "ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
            "at": time.time(),
        },
    )

    def call(op, *args):
        p = subprocess.run(
            [*ctl, "cluster", op, "--shared", str(queue), *map(str, args)],
            capture_output=True,
            text=True,
            timeout=25,
        )
        require(p.returncode == 0, f"runq {op}: " + (p.stderr or p.stdout)[-1000:])
        return json.loads(p.stdout)

    local = Path(tempfile.mkdtemp(prefix="tide-q-" + str(rank) + "-"))
    log = area / f"agent-{rank}.log"
    with log.open("wb") as stream:
        svc = subprocess.Popen(
            [
                *ctl,
                "--root",
                str(local),
                "cluster",
                "serve",
                "--shared",
                str(queue),
                "--deadline-unix",
                str(deadline + 15),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    try:
        if rank == 0:
            status = {}
            while time.time() < deadline:
                status = call("status")
                if len([w for w in status["workers"] if w["state"] == "serving"]) == count:
                    break
                time.sleep(0.4)
            require(len(status.get("workers", [])) == count, "missing worker instances")
            keys = []
            for i, spec in enumerate(a["tasks"]):
                worker = int(spec.get("worker", i % count)) if count > 1 else 0
                where = read(area / f"node-{worker}.json")
                key = ident(spec["id"])
                keys.append(key)
                resources = spec["resources"]
                argv = [
                    "--worker",
                    worker,
                    "--key",
                    key,
                    "--cwd",
                    where["root"],
                    "--backend",
                    resources["backend"],
                    "--ram-mib",
                    resources["ram_mib"],
                    "--threads",
                    resources["threads"],
                    "--timeout",
                    max(1, int(deadline - time.time())),
                ]
                if resources["backend"] == "npu":
                    argv.extend(["--hbm-mib", resources["hbm_mib"]])
                call(
                    "submit",
                    *argv,
                    "--",
                    where["python"],
                    "-B",
                    str(Path(where["root"]) / "tide_batch.py"),
                    "task",
                    key,
                )
            while time.time() < deadline:
                status = call("status")
                states = {j["key"]: j["state"] for j in status["tasks"]}
                if all(
                    states.get(k) in {"completed", "cancelled", "blocked", "failed", "unconfirmed"}
                    for k in keys
                ):
                    break
                time.sleep(1.0)
            write(area / "queue-result.json", status)
            export(area / "queue-result.json", obs, "products/tide-queue.json")
            call("finish")
            require(
                all(states.get(k) == "completed" for k in keys),
                "queue tasks incomplete or failed: " + str(states),
            )
        rc = svc.wait(timeout=max(5, deadline + 30 - time.time()))
        require(rc == 0, "queue agent exited with " + str(rc))
        stopped = read(queue / ".runq-cluster" / "workers" / f"{rank}.json")
        require(stopped["state"] == "stopped", "queue did not finish")
        node = {
            **read(ent),
            "queue_instance": stopped["instance"],
            "queue_counts": stopped["counts"],
            "queue_stopped_at": time.time(),
        }
        write(area / f"node-result-{rank}.json", node)
        export(area / f"node-result-{rank}.json", obs, f"products/tide-node-{rank}.json")
        shutil.rmtree(local)
        wait_seconds = int(a["tasks"][0].get("inspection_seconds", 0)) if a["number"] == 1 else 0
        wait_until = time.monotonic() + min(300, wait_seconds, max(0, deadline - time.time()))
        while time.monotonic() < wait_until:
            mark = obs / "control/client-confirmed.json"
            if mark.is_file() and read(mark).get("attempt") == attempt:
                break
            time.sleep(1)
        return 0
    finally:
        if svc.poll() is None:
            svc.terminate()
            try:
                svc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                svc.kill()
                svc.wait()


def aggregate(frozen_dir, roots):
    frozen = Frozen(frozen_dir)
    project_root = Path(__file__).resolve().parents[1]
    source = project_root if (project_root/"pilotlab/gate3.py").is_file() else frozen.directory/"source"
    sys.path.insert(0, str(source))
    try:
        from pilotlab.gate3 import read_evidence
    finally:
        sys.path.pop(0)
    expected = frozen.manifest["tasks"]
    receipts, evidence, locations, problems = {}, {}, {}, []
    for root in map(Path, roots):
        for f in sorted((root / "products/tasks").glob("*.json")):
            try:
                rec = read(f)
                key = ident(rec["task"])
                aid = ident(rec["attempt"])
                require(
                    rec["schema"] == "maflow-task-result/v1"
                    and key == f.stem
                    and key in expected
                    and rec["fingerprint"] == expected[key]["fingerprint"],
                    "unknown task or changed fingerprint",
                )
                artifacts = rec["artifacts"]
                for name, artifact in artifacts.items():
                    path = relative(root, artifact["path"])
                    require(
                        path.stat().st_size == artifact["size"] and sha(path) == artifact["sha256"],
                        "changed artifact: " + name,
                    )
                require(rec["status"] in {"completed", "interrupted", "failed"}, "invalid state")
                if rec["status"] != "failed":
                    verdict = read(relative(root, artifacts["verdict"]["path"]))
                    result = read(relative(root, artifacts["result"]["path"]))
                    require(
                        verdict["status"] == rec["status"]
                        and result["task_sha256"] == rec["fingerprint"]
                        and result["id"] == key
                        and result["source_identity"]
                        == read(relative(frozen.directory, expected[key]["path"]))["source_identity"]
                        and result["checkpoint"]["sha256"] == artifacts["checkpoint"]["sha256"],
                        "worker evidence mismatch",
                    )
                    if rec["status"] == "completed":
                        require(
                            result["status"] == "completed"
                            and result["health"]["eligible"]
                            and result["recovery"]["checkpoint_load"]["status"] == "exact",
                            "project completion failed",
                        )
                    if rec.get("resume_from"):
                        parent = rec["resume_from"]
                        resumed = result.get("resume") or {}
                        require(
                            resumed.get("sha256") == parent["sha256"]
                            and resumed.get("from_steps") == parent["step"]
                            and not resumed.get("fallback_used")
                            and resumed.get("checkpoint_load", {}).get("status") == "exact"
                            and result["training"]["steps"] > parent["step"],
                            "result does not restore the declared checkpoint",
                        )
                instance = (key, aid)
                old = receipts.get(instance)
                require(old is None or old == rec, "conflicting receipt for one attempt")
                receipts[instance] = rec
                if rec["status"] in ("completed", "interrupted"):
                    locations.setdefault(
                        instance,
                        {
                            "receipt": {"path": str(f.resolve()), "sha256": sha(f)},
                            **{
                                role: {
                                    "path": str(relative(root, artifacts[role]["path"])),
                                    "sha256": artifacts[role]["sha256"],
                                }
                                for role in ("checkpoint", "result")
                            },
                        },
                    )
                if rec["status"] == "completed":
                    evidence.setdefault(
                        instance,
                        {
                            "task_path": str(relative(frozen.directory, expected[key]["path"])),
                            "task_sha256": rec["fingerprint"],
                            "result_path": str(relative(root, artifacts["result"]["path"])),
                            "result_sha256": artifacts["result"]["sha256"],
                            "checkpoint_path": str(relative(root, artifacts["checkpoint"]["path"])),
                            "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
                        },
                    )
            except (OSError, ValueError, KeyError, TypeError) as e:
                problems.append({"file": str(f), "error": str(e)})
    ids = {(v["batch"], v["project"], v["plan_sha256"]) for v in receipts.values()}
    if len(ids) > 1:
        problems.append({"error": "different batch, project or frozen plan"})
    summary = {}
    for key in sorted(expected):
        selected = [v for (k, _), v in sorted(receipts.items()) if k == key]
        successful = [r for r in selected if r["status"] == "completed"]
        if len(successful) > 1:
            problems.append({"task": key, "error": "multiple completed attempts"})
        parents = {}
        for rec in selected:
            parent = rec.get("resume_from")
            if parent:
                hits = [
                    p
                    for p in selected
                    if p["attempt"] != rec["attempt"]
                    and p["status"] == "interrupted"
                    and p["artifacts"]["checkpoint"]["sha256"] == parent["sha256"]
                    and p["artifacts"]["checkpoint"]["step"] == parent["step"]
                    and (
                        not parent.get("parent", parent.get("attempt"))
                        or p["attempt"] == parent.get("parent", parent.get("attempt"))
                    )
                ]
                if len(hits) != 1:
                    problems.append(
                        {"task": key, "error": "missing or ambiguous predecessor", "attempt": rec["attempt"]}
                    )
                else:
                    parents[rec["attempt"]] = hits[0]["attempt"]
        summary[key] = {
            "status": "completed" if len(successful) == 1 else "missing" if not selected else "incomplete",
            "attempts": [{"attempt": r["attempt"], "status": r["status"]} for r in selected],
        }
        if len(successful) == 1:
            aid = successful[0]["attempt"]
            item = evidence[(key, aid)]
            original_task = read(item["task_path"])
            if parents or original_task.get("resume_checkpoint"):
                chain = []
                seen = set()
                while aid not in seen:
                    seen.add(aid)
                    chain.append(locations[(key, aid)])
                    if aid not in parents:
                        break
                    aid = parents[aid]
                else:
                    problems.append({"task": key, "error": "cyclic portable attempt ancestry"})
                item["portable_attempts"] = chain[::-1]
                if original_task.get("resume_checkpoint"):
                    old = original_task["resume_checkpoint"]
                    item["initial_checkpoint"] = {
                        "original": old,
                        "path": str(frozen.path(old)),
                        "sha256": original_task["resume_checkpoint_sha256"],
                    }
            summary[key]["evidence"] = item
            try:
                read_evidence(item, checkpoint=True)
            except (OSError, ValueError, TypeError, KeyError) as error:
                problems.append({"task":key,"error":"project evidence check: "+str(error)})
    return {
        "schema": "tide-aggregate/v1",
        "tasks": summary,
        "problems": problems,
        "missing": [k for k, r in summary.items() if r["status"] == "missing"],
        "complete": not problems and all(r["status"] == "completed" for r in summary.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("cluster")
    sub.add_parser("task").add_argument("task_id")
    merge = sub.add_parser("aggregate")
    merge.add_argument("--frozen", type=Path, required=True)
    merge.add_argument("--attempt-root", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "aggregate":
        result = aggregate(args.frozen, args.attempt_root)
        if args.output:
            write(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(int(not result["complete"]))
    root = Path(__file__).resolve().parent
    sys.exit(task_main(root, args.task_id) if args.command == "task" else cluster(root))
