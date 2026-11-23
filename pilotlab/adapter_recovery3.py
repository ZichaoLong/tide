"""Reconstruct an unpublished adapter receipt after a verified zero-origin rejection."""
import json
from pathlib import Path

from .gate3 import eligible
from .prepare_resume3 import reference
from .resume_adapter3 import _result, checked, finish_result_metadata, inherited_selection
from .shared_npu3 import worker_spec_digest
from .source_migration3 import migration_inputs, prior_evaluation_tokens, read_bound, verify_stop_journal


VERIFIER_CONTINUATION = "pilot-r3-zero-origin-verifier-continuation-v1"
REFUSAL = "resume_adapter_refused: ValueError: migration did not restore the declared state and exposure"


def zero_origin_checkpoint(task, result):
    """Independently decode the exact empty private state rejected in this chain."""
    from . import worker3

    for _ in range(64):
        receipt = read_bound(task["stage_a_migration"])
        parent = read_bound(receipt["original_task"])
        cp = receipt["checkpoint"]
        if result["resume"]["state_reused"] is False:
            payload = worker3._read_checkpoint(Path(cp["path"]), cp["sha256"])
            worker3.validate_checkpoint(payload, parent, parent["device"])
            if (cp["steps"] != 0 or cp["tokens"] != 0 or payload["private_state"] is not None
                    or payload["task_id"] != parent["id"]
                    or any(payload[k] != cp[k] for k in ("steps", "tokens", "state_sha256"))):
                raise ValueError("rejected checkpoint has no verified empty zero origin")
            return cp
        if receipt["mode"] != "signal" or not parent.get("stage_a_migration"):
            break
        task, result = parent, read_bound(receipt["result"])
    raise ValueError("failed adapter chain lacks the reviewed zero-origin case")


def recover_completion(audit_ref, scheduler_ref, tid):
    """Rebuild only the missing metadata. The original process exit and result stay bound."""
    audit, state = read_bound(audit_ref), read_bound(scheduler_ref)
    rec = state["tasks"][tid]
    gid = rec["shared_group"]
    reason = f"worker_process_failed:{tid}:1"
    if (audit["schema"] != "pilot-r3-terminal-audit-v1" or audit["status"] != "passed" or audit["findings"]
            or any(audit["execution"]["scheduler"][k] != v for k,v in scheduler_ref.items())
            or state["blocking_reason"] != f"Blocked: dynamic_owner_failed:{gid}:{reason}"
            or state["remaining_descendants"] != [] or state["stage_b_gate"] != "closed"
            or not all(audit["lifecycle"][k]["released"] is True for k in ("cgroup","processes","queue"))):
        raise ValueError("adapter recovery requires the audited terminal release and specific worker exit")
    owner = audit["shared_groups"]["groups"][gid]
    group, entry = owner["group"], owner["entry"]
    worker = group["workers"][tid]
    attempt = rec["attempts"][-1]
    directory = Path(attempt["path"])
    task_ref = reference(rec["task_path"])
    raw_ref = reference(directory/"result.json")
    task = read_bound(task_ref)
    result = _result(raw_ref, task, task_ref["sha256"])
    manifest = read_bound({"path":entry["manifest_path"], "sha256":entry["manifest_sha256"]})
    spec, = (w for w in manifest["workers"] if w["id"] == tid)
    cfg = state["config"]
    receipt_ref = task["stage_a_migration"]
    receipt = read_bound(receipt_ref)
    adapter = cfg["resume_adapter"]
    worker_source = cfg.get("worker_source", cfg["source"])
    foundation = cfg.get("worker_foundation", cfg["foundation"])
    executable = reference(Path(worker_source)/"pilotlab/worker3.py")
    expected = [cfg["python"], "-u", adapter["path"], "--task", task_ref["path"], "--output", str(directory),
                "--data", cfg["data"], "--device", task["device"], "--deadline-unix", str(cfg["deadline_unix"]),
                "--resume-checkpoint", receipt["checkpoint"]["path"], "--receipt", receipt_ref["path"],
                "--receipt-sha256", receipt_ref["sha256"], "--adapter-sha256", adapter["sha256"]]
    if (not owner["binding_verified"] or owner["owner_alive"] or group["reason"] != reason
            or entry["owner_return_code"] != 1 or group["remaining_descendants"] != []
            or not group["queue_released_unix"] or not group["allocation_released_unix"]
            or spec["argv"] != expected or attempt["command"] != expected or spec["cwd"] != worker_source
            or spec["env"]["PYTHONPATH"] != worker_source + ":" + foundation + "/src"
            or worker_spec_digest(spec) != worker["worker_spec_sha256"] or spec["task_sha256"] != task_ref["sha256"]
            or task["id"] != tid or task["phase"] != "a1" or rec["reason"] != "target_reached"
            or any(row.get("status") != "failed" for row in (rec, attempt, worker, group))
            or worker["return_code"] != 1 or attempt["return_code"] != 1
            or not worker["descendant_cleanup_verified"] or worker["remaining_descendants"] != []
            or worker["process"] != attempt["worker_process"] or worker["output"] != str(directory)
            or any(row["result_path"] != raw_ref["path"] or row["result_sha256"] != raw_ref["sha256"]
                   for row in (worker,rec)) or worker["result_status"] != "completed"
            or "execution_adapter" in result or result.get("failure_class") or result["stop_reason"] != "target_reached"):
        raise ValueError("original command, owner, worker result or nonzero exit differs from the reviewed failure")
    checked({"path": str(directory/"task.json"), "sha256": task_ref["sha256"]}, "worker request")
    files = {k:reference(directory/name) for k,name in (("log","stdout.log"), ("heartbeat","heartbeat.json"),
             ("checkpoint_index","checkpoint.json"), ("steps","steps.jsonl"))}
    if Path(worker["log"]) != Path(files["log"]["path"]) or Path(worker["log"]).read_text().splitlines()[-1] != REFUSAL:
        raise ValueError("adapter did not report the reviewed ancestry rejection")
    cp = result["checkpoint"]
    recovery = result["recovery"]["checkpoint_load"]
    hb = read_bound(files["heartbeat"])
    final = result["validation_final"]
    train = result["training"]
    if (hb != rec["heartbeat"] or hb["task_id"] != tid or hb["phase"] != task["phase"] or hb["pid"] != worker["pid"]
            or hb["status"] != "completed" or hb["stop_reason"] != "target_reached"
            or not result["finished_unix"] <= hb["updated_unix"] <= worker["finished_unix"]
            or eligible(task,result) or result["causality"]["within_tolerance"] is not True
            or read_bound(files["checkpoint_index"])["current"] != cp or Path(cp["path"]).parent != directory/"checkpoints"
            or recovery["status"] != "exact" or recovery["state_sha256"] != cp["state_sha256"]
            or any(row[k] != cp[k] for row in (hb,train,recovery) for k in ("tokens","steps"))
            or final["status"] != "completed" or final["split"] != "selection" or final["tokens"] != task["eval_tokens"]
            or final["requested_tokens"] != final["tokens"] or result["evaluation"]["tokens"] != final["tokens"]):
        raise ValueError("final checkpoint, exposure, selection or heartbeat is incomplete")
    checked(cp, "final worker checkpoint")
    steps = [json.loads(line) for line in Path(files["steps"]["path"]).read_text().splitlines()]
    verify_stop_journal(steps,hb,cp,task,attempt)
    if (len(steps) != train["segment_steps"] or steps[0]["step"] != train["from_steps"] + 1
            or any(s["tokens"] != s["step"] * task["sequence_length"] * task["batch_size"] for s in steps)):
        raise ValueError("journal does not cover every update from the loaded checkpoint")
    _, previous = migration_inputs(task,receipt)
    origin = receipt["initial_evaluation"]
    if (receipt["mode"] != "signal" or origin["mode"] != "inherit"
            or result["validation_initial"] != inherited_selection(origin["observation"],origin["result"],tid,origin["exposure"])
            or receipt["prior_observed_evaluation_tokens"] != prior_evaluation_tokens(receipt)):
        raise ValueError("initial selection or cumulative evaluation exposure changed")
    finish_result_metadata(result,task,receipt,previous,inherited=True,task_ref=task_ref,receipt_ref=receipt_ref,
                           worker_ref=executable,adapter_ref=adapter)
    result["post_run_verification"] = {"schema":VERIFIER_CONTINUATION, "original_result":raw_ref,
        "original_return_code":attempt["return_code"], "terminal_audit":audit_ref, "scheduler":scheduler_ref,
        "group":gid, "worker_pid":worker["pid"], "evidence":files, "zero_origin":zero_origin_checkpoint(task,result),
        "verifier_sha256":{name:reference(Path(__file__).with_name(name))["sha256"] for name in
                           ("adapter_recovery3.py","continuation3.py","source_migration3.py","resume_adapter3.py")}}
    return result
