"""Freeze or start the reviewed Stage A follow-up alongside its original controller."""
from __future__ import annotations

import argparse
from contextlib import closing
import fcntl
import json
from pathlib import Path
import sqlite3
import subprocess
import time

from pilotlab.audit3 import systemd_observation
from pilotlab.hbm_profile3 import HbmProfile
from pilotlab.io import atomic_json, sha256
from pilotlab.prepare_campaign3 import budget_remaining, storage_observation
from pilotlab.prepare_resume3 import reference
from pilotlab.runq import dispatcher_unit
from pilotlab.runq_store import lookup, setting
from pilotlab.scheduler3 import freeze_json, read
from pilotlab.source_migration3 import read_bound


def copy_once(path, dest):
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write(path.read_bytes())
        dest.chmod(0o444)
    if sha256(dest) != sha256(path):
        raise ValueError(f"frozen source changed: {dest}")
    return sha256(dest)


def prepare(args):
    root, dst = Path(__file__).resolve().parents[1], args.directory.resolve()
    launch = read(args.launch)
    review_ref = reference(dst / "review.json")
    review = read_bound(review_ref)
    primary = read_bound(review["primary_plan"])
    config = primary["config"]
    if (review["primary_plan"] != {"path": launch["plan_path"], "sha256": launch["plan_sha256"]}
            or config["runq_root"] != launch["runq_root"] or review["stage_b_gate"] != "closed"):
        raise ValueError("primary launch differs from the review")
    plan_path = dst / "plan.json"
    if plan_path.exists():
        existing = read(plan_path)
        if (existing["review"] != review_ref or existing["config"]["predecessor"]["job"] != args.predecessor
                or existing["software_input"] != reference(args.software_env)
                or any(sha256(p) != h for p, h in existing["controller_files"].items())):
            raise ValueError("destination contains another frozen follow-up")
        return reference(plan_path)
    original = read_bound(review["original_inputs"]["original_plan"])
    subjects = [m for m in original["subjects"] if m["id"] in review["subjects"]]
    for source in review["sources"].values():
        identity = read_bound(source["identity"])
        if (sha256(Path(config["data"]) / "manifest.json") != identity["data_manifest_sha256"]
                or any(sha256(p) != h for p, h in identity["source_files"].items())):
            raise ValueError("frozen worker or data identity changed")
    controller = dst / "source"
    files = sorted((root / "pilotlab").glob("*.py"))
    storage = storage_observation(dst, reserve=0, snapshot_bytes=sum(p.stat().st_size for p in files))
    budget = budget_remaining(launch["budget_ledger"], time.time())
    if config["deadline_unix"] != budget["deadline_unix"]:
        raise ValueError("primary ledger and deadline disagree")
    software = read(args.software_env)
    if software.get("ASCEND_ENV_KIND") != "dev-workspace" or software.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
        raise ValueError("expected the source probes' validated Ascend environment")
    frozen = {str(controller / "pilotlab" / p.name): copy_once(p, controller / "pilotlab" / p.name) for p in files}
    path = Path(config["runq_root"]) / "queue.sqlite3"
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        job = lookup(db, args.predecessor)
        spec = json.loads(job["spec"])
        if (spec["owner"] != launch["output"] or spec["allowed"] != ["npu"]
                or job["job_key"].split("/")[:2] != [config["run_id"], spec["task_id"]]
                or spec["deadline"] != config["deadline_unix"]):
            raise ValueError("requested predecessor is not a primary NPU task")
        gate = dict(job=job["id"], job_key=job["job_key"], owner=spec["owner"], task_id=spec["task_id"],
                    minimum_generation=max(1, job["generation"]))
        limits = setting(db, "limits")
        floors = setting(db, "hbm_floors." + spec["owner"], {})
    profile = HbmProfile(launch["output"], readonly=True)
    try:
        snapshot = profile.snapshot()
    finally:
        profile.db.close()
    inherited = read_bound(review["original_inputs"]["r07_model_resources"])
    initial = {mid: inherited[mid] for mid in review["subjects"] if mid in inherited}
    ref_arm = review["sources"]["reference"]
    shared = {k: config[k] for k in ("python", "data", "budget_ledger", "deadline_unix", "runq_root",
              "host_cpu_affinity", "max_cpu_task_threads", "max_workers", "shared_execution",
              "dynamic_shared_admission", "resource_dispatch", "npu_workers_per_device")}
    shared.update(source=str(controller), foundation=ref_arm["foundation"], worker_source=ref_arm["source"],
        worker_foundation=ref_arm["foundation"], resume_adapter=ref_arm["adapter"], predecessor=gate,
        run_id=dst.name, max_npu_devices=limits["cards"], max_cpu_workers=limits["cpu"],
        storage_reserve_bytes=0, prior_evaluation_reservations={m: r["evaluation"]["previous"]
        for m, r in review["subjects"].items()})
    env = {**software, "PYTHONPATH": str(controller) + ":" + ref_arm["foundation"] + "/src"}
    # The source probe environment was saved with the software variables only.
    if any(not (k.startswith("ASCEND_") or k in {"LD_LIBRARY_PATH", "PATH", "PYTHONPATH", "MKL_NUM_THREADS",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED",
            "TOOLCHAIN_HOME", "TORCH_DEVICE_BACKEND_AUTOLOAD"}) or "VISIBLE_DEVICES" in k for k in env):
        raise ValueError("unexpected software environment")
    coverage = sum(max(r.get("checkpoint_bytes_by_batch", {}).values(), default=r["checkpoint_bytes"])
                   for mid, r in initial.items() if review["subjects"][mid]["mode"] != "carry_recipe")
    plan = dict(schema="pilot-r3-deferred-stage-a-followup-v1", review=review_ref, output=str(dst / "round"),
                controller_source=str(controller), controller_files=frozen, stage_b_gate="closed", subjects=subjects,
                source_identity=read_bound(ref_arm["identity"]), config=shared, protocol=original["protocol"],
                software_input=reference(args.software_env),
                environment=reference(freeze_json(dst / "software-environment.json", env)),
                hbm_profile=reference(freeze_json(dst / "hbm-profile.json", snapshot)),
                inherited_hbm_floors=reference(freeze_json(dst / "inherited-hbm-floors.json", floors)),
                initial_resources=reference(freeze_json(dst / "initial-model-resources.json", initial)),
                unit="tide-pilot-r3-" + dst.name + ".service", pipeline_stage_a=True,
                pipeline_model_limit=len(subjects), retire_consumed_a_parents=False,
                capacity_observation=dict(storage=storage, budget=budget, queue_limits=limits,
                    sum_largest_checkpoint_known_rerun_models_bytes=coverage,
                    upper_estimate_at_two_generations_per_12_a1_and_3_init=30 * coverage,
                    capacity_still_to_measure=sum(row["old_gap"] == "incomplete_capacity_profile"
                                                  for row in review["subjects"].values())))
    for key in ("cpu_recipe_calibration", "cpu_min_exclusive_rate_ratio", "profile_private_checkpoint_target_bytes",
                "profile_private_parameter_factor"):
        plan[key] = original[key]
    return reference(freeze_json(plan_path, plan))


def submit(args):
    dst = args.directory.resolve()
    plan_ref = reference(dst / "plan.json")
    plan = read_bound(plan_ref)
    env = read_bound(plan["environment"])
    config = plan["config"]
    unit = plan["unit"]
    pointer = args.launch.resolve()
    with (pointer.parent / "launch-submit.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        launch = read(pointer)
        prior = launch.get("deferred_followup")
        if prior:
            if prior["plan"] != plan_ref or prior["unit"] != unit:
                raise ValueError("a different deferred follow-up already owns the launch pointer")
            return prior
        review = read_bound(plan["review"])
        if (review["primary_plan"] != {"path": launch["plan_path"], "sha256": launch["plan_sha256"]}
                or config["predecessor"]["owner"] != launch["output"] or plan["output"] == launch["output"]
                or any(sha256(p) != h for p, h in plan["controller_files"].items())):
            raise ValueError("source or primary controller changed before submission")
        budget = budget_remaining(config["budget_ledger"], time.time())
        storage = storage_observation(plan["output"], reserve=0, snapshot_bytes=0)
        queue = Path(config["runq_root"]) / "queue.sqlite3"
        original = {u: systemd_observation(u)["properties"]["InvocationID"]
                    for u in (launch["unit"], dispatcher_unit(Path(config["runq_root"])))}
        with closing(sqlite3.connect(queue.as_uri() + "?mode=ro", uri=True)) as db:
            before = db.execute("""SELECT id,pid,ticks,generation FROM jobs WHERE state='running'
                AND json_extract(spec,'$.owner')=? ORDER BY id""", (launch["output"],)).fetchall()
        argv = ["systemd-run", "--user", "--unit=" + unit, "--service-type=exec", "--slice=background.slice",
                "--working-directory=" + plan["controller_source"], "--property=KillMode=control-group",
                "--property=RuntimeMaxSec=" + str(int(budget["remaining_wall_seconds"])),
                "--property=TimeoutStopSec=120", "--property=MemoryAccounting=yes",
                "--property=StandardOutput=append:" + str(dst / "campaign.log"),
                "--property=StandardError=append:" + str(dst / "campaign.log"),
                *("--setenv=" + k + "=" + v for k, v in env.items()),
                config["python"], "-u", "-m", "pilotlab.deferred_campaign3",
                "--plan", plan_ref["path"], "--sha256", plan_ref["sha256"]]
        service = systemd_observation(unit)["properties"]
        if service.get("ActiveState") not in ("active", "activating"):
            subprocess.run(argv, check=True)
            service = systemd_observation(unit)["properties"]
        # Covers a session interruption between systemd accepting the unit and publishing its receipt.
        pid = int(service["MainPID"])
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
        if service.get("ActiveState") != "active" or command != argv[-8:]:
            raise ValueError("live service is not the frozen follow-up controller")
        if any(systemd_observation(u)["properties"].get("InvocationID") != invocation for u, invocation in original.items()):
            raise ValueError("primary controller or dispatcher changed during live admission")
        with closing(sqlite3.connect(queue.as_uri() + "?mode=ro", uri=True)) as db:
            still_running, ended = [], []
            for job, *identity in before:
                state, *now = db.execute("SELECT state,pid,ticks,generation FROM jobs WHERE id=?", (job,)).fetchone()
                if state == "running" and now == identity:
                    still_running.append(job)
                elif state in ("completed", "failed", "cancelled"):
                    ended.append(dict(job=job, state=state))
                else:
                    raise ValueError("a primary worker changed identity during admission")
        value = dict(unit=unit, invocation=service["InvocationID"], plan=plan_ref, output=plan["output"],
                     review=plan["review"], log=str(dst / "campaign.log"), predecessor=config["predecessor"],
                     submitted_unix=time.time(), storage=storage, budget=budget, original_invocations=original,
                     primary_workers_same_pid=still_running, primary_workers_ended=ended)
        atomic_json(pointer, {**launch, "deferred_followup": value})
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--software-env", type=Path, required=True)
    prep.add_argument("--predecessor", type=int, required=True)
    commands.add_parser("submit")
    args = parser.parse_args()
    result = prepare(args) if args.action == "prepare" else submit(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
