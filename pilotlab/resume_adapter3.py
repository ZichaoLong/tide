"""Resume a bound, signal-interrupted stage-A request using its frozen core.

Run this file directly with PYTHONPATH naming the original worker snapshot.
Completed initial evaluation stays at its original exposure. Training, state
loading, causality checks and every subsequent evaluation use that worker.
The adapter has a separate identity; it is not part of the checkpoint's source
identity and does not establish numerical trajectory equivalence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import sys


SCHEMA = "pilot-r3-task-resume-receipt-v1"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def checked(reference, label):
    if not isinstance(reference, dict) or not reference.get("path") or not reference.get("sha256"):
        raise ValueError(f"missing {label} reference")
    path = Path(reference["path"]).resolve()
    if file_hash(path) != reference["sha256"]:
        raise ValueError(f"{label} SHA256 mismatch")
    return path


def _validate_legacy_receipt(args, worker):
    """Reject changed or unrelated evidence before device initialization."""
    if file_hash(__file__) != args.adapter_sha256:
        raise ValueError("adapter SHA256 mismatch")
    receipt_path = checked({"path": str(args.receipt), "sha256": args.receipt_sha256}, "receipt")
    receipt = read_json(receipt_path)
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported resume receipt schema")
    task_path = checked(receipt.get("original_task"), "original task")
    result_path = checked(receipt.get("result"), "original result")
    checkpoint_path = checked(receipt.get("checkpoint"), "checkpoint")
    if file_hash(args.task) != receipt["original_task"]["sha256"]:
        raise ValueError("CLI task differs from the original immutable task")
    if Path(args.resume_checkpoint).resolve() != checkpoint_path:
        raise ValueError("CLI checkpoint differs from the bound checkpoint")
    output = Path(args.output).resolve()
    if output == result_path.parent or output.is_relative_to(result_path.parent):
        raise ValueError("resume output would modify the original attempt")
    if (output / "result.json").exists() or (output / "checkpoint.json").exists():
        raise ValueError("resume requires a new attempt output")
    task, previous = read_json(task_path), read_json(result_path)
    if (task.get("phase") != "a1" or task.get("resume_checkpoint")
            or task.get("allow_confirmation") or task.get("evaluation_start_tokens", 0) != 0):
        raise ValueError("adapter only accepts an original A1 request from zero exposure")
    if (receipt.get("task_id") != task.get("id") or previous.get("id") != task.get("id")
            or previous.get("model_id") != task.get("model_id") or previous.get("phase") != "a1"
            or previous.get("task_sha256") != receipt["original_task"]["sha256"]):
        raise ValueError("original task/result identity mismatch")
    identity = task.get("source_identity")
    if not identity or previous.get("source_identity") != identity or receipt.get("source_identity") != identity:
        raise ValueError("resume source identity mismatch")
    if (previous.get("schema") != "pilot-r3-worker-result-v1"
            or previous.get("status") != "budget_exhausted" or previous.get("stop_reason") != "signal"
            or previous.get("failure_class") or previous.get("resume")):
        raise ValueError("only an original signal-interrupted worker result is admissible")
    if previous.get("causality", {}).get("within_tolerance") is not True:
        raise ValueError("original causality check was not qualified")
    permitted = {"declared_training_exposure_not_reached", "evaluation_incomplete", "checkpoint_load_not_verified"}
    if set(previous.get("health", {}).get("blocking_reasons", [])) - permitted:
        raise ValueError("original result contains a non-interruption health failure")
    worker.validate_task(task)
    interval = task.get("eval_interval_tokens", 1048576)
    if interval > 0 and task["target_tokens"] > interval:
        raise ValueError("adapter does not accept an A1 request with intermediate evaluations")
    if float(task.get("closing_reserve_seconds", 120)) <= 0:
        raise ValueError("a positive closing reserve is required to identify the initial evaluation")
    training = previous.get("training", {})
    if training.get("from_tokens") != 0 or training.get("from_steps") != 0:
        raise ValueError("original A1 attempt did not start from zero")
    checkpoint = receipt["checkpoint"]
    original_checkpoint = previous.get("checkpoint") or {}
    if Path(original_checkpoint.get("path", "")).resolve() != checkpoint_path:
        raise ValueError("result checkpoint path mismatch")
    for key in ("sha256", "tokens", "steps"):
        if checkpoint.get(key) != original_checkpoint.get(key):
            raise ValueError(f"result checkpoint {key} mismatch")
    if (type(checkpoint.get("tokens")) is not int or not 0 < checkpoint["tokens"] <= task["target_tokens"]
            or checkpoint["tokens"] != training.get("tokens") or checkpoint["steps"] != training.get("steps")):
        raise ValueError("checkpoint and interrupted training exposure disagree")
    initial = previous.get("validation_initial", {})
    count = task.get("eval_tokens", 16384)
    nll = initial.get("nll")
    if (receipt.get("original_initial_evaluation") != initial or initial.get("status") != "completed"
            or initial.get("split") != "selection" or count <= 0
            or initial.get("tokens") != count or initial.get("requested_tokens") != count
            or isinstance(nll, bool) or not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0):
        raise ValueError("original initial evaluation is incomplete or unbound")
    final = previous.get("validation_final", {})
    spent_final = final.get("tokens")
    if (final.get("status") != "budget_exhausted" or final.get("split") != "selection"
            or final.get("requested_tokens") != count or type(spent_final) is not int
            or not 0 <= spent_final < count or spent_final % task["sequence_length"]
            or previous.get("evaluation", {}).get("tokens") != count + spent_final):
        raise ValueError("original evaluation accounting is not a bounded initial/partial-final pair")
    additional = receipt.get("evaluation_additional_reservation_tokens")
    if type(additional) is not int or additional != 32 + spent_final:
        raise ValueError("additional reservation must cover the repeated causality probe and partial final evaluation")
    worker.validate_source_identity(task, Path(args.data))
    payload = worker._read_checkpoint(checkpoint_path, checkpoint["sha256"])
    worker.validate_checkpoint(payload, task, args.device)
    if (payload.get("task_id") != task["id"] or payload.get("tokens") != checkpoint["tokens"]
            or payload.get("steps") != checkpoint["steps"]
            or payload.get("state_sha256") != original_checkpoint.get("state_sha256")
            or payload.get("initialization") != previous.get("initialization")):
        raise ValueError("checkpoint payload differs from the bound task/result")
    return receipt, task, previous


RELAY_SCHEMA = "pilot-r3-task-resume-receipt-v2"
RELAY_PHASES = {"a1", "a2", "a3", "init", "preflight"}


def _result(reference, task, task_sha):
    value = read_json(checked(reference, "prior result"))
    if (value.get("schema") != "pilot-r3-worker-result-v1" or value.get("id") != task["id"]
            or value.get("phase") != task["phase"] or value.get("model_id") != task["model_id"]
            or value.get("task_sha256") != task_sha or value.get("source_identity") != task["source_identity"]):
        raise ValueError("prior result/task/source identity mismatch")
    return value


def initial_origin(reference, task, task_sha):
    """Resolve inherited evaluations to bytes observed at the original exposure."""
    seen = set()
    for _ in range(64):
        path = checked(reference, "initial evaluation result")
        if path in seen:
            raise ValueError("cyclic initial evaluation provenance")
        seen.add(path)
        value = _result(reference, task, task_sha)
        initial = value.get("validation_initial", {})
        parent = initial.get("inherited_from")
        if parent:
            if parent.get("field") != "validation_initial" or parent.get("task_id") != task["id"]:
                raise ValueError("inherited initial evaluation belongs to another field/task")
            mark = value.get("execution_adapter") or {}
            if mark.get("schema") == "pilot-r3-source-migration-adapter-v1":
                # A source transition changes the task hash. Its frozen receipt
                # binds both owners and the original, measured stage-start NLL.
                ref = task.get("stage_a_migration")
                if not ref or ref != mark.get("receipt") or mark.get("new_task", {}).get("sha256") != task_sha:
                    raise ValueError("inherited evaluation lacks its source transition")
                receipt = read_json(checked(ref, "initial evaluation migration"))
                origin = receipt["initial_evaluation"]
                measured = {key:parent[key] for key in ("path","sha256")}
                if (origin.get("mode") != "inherit" or origin.get("result") != measured
                        or origin["observation"].get("nll") != initial.get("nll")
                        or origin["exposure"] != task.get("evaluation_start_tokens", 0)
                        or receipt["original_task"] != mark.get("original_task")
                        or receipt["result"] != mark.get("original_result")):
                    raise ValueError("source transition changed the original initial observation")
                original = read_json(checked(receipt["original_task"], "initial evaluation original task"))
                if receipt["mode"] == "checkpoint_only":
                    if "initial_replay" in receipt:
                        if measured != receipt["initial_replay"]:
                            raise ValueError("inherited evaluation differs from its bound stage-start replay")
                        task, task_sha, reference = original, receipt["original_task"]["sha256"], measured
                        continue
                    receipt = read_json(checked(original["stage_a_migration"], "interrupted source migration"))
                    original = read_json(checked(receipt["original_task"], "interrupted original task"))
                task, task_sha, reference = original, receipt["original_task"]["sha256"], receipt["result"]
                continue
            reference = {key: parent[key] for key in ("path", "sha256")}
            continue
        count = task.get("eval_tokens", 16384)
        nll = initial.get("nll")
        if initial.get("status") != "completed":
            return None
        exposure = value.get("training", {}).get("from_tokens")
        if (count <= 0 or initial.get("tokens") != count or initial.get("requested_tokens") != count
                or initial.get("split") != "selection" or exposure != task.get("evaluation_start_tokens", 0)
                or isinstance(nll, bool) or not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0):
            raise ValueError("initial evaluation is not bound to the original exposure")
        return {"mode": "inherit", "result": reference, "field": "validation_initial",
                "exposure": exposure, "observation": initial}
    raise ValueError("initial evaluation provenance too deep")


def build_relay_receipt(task_reference, result_references, reserved_tokens):
    """Pure evidence check and conservative *incremental* evaluation bound."""
    task = read_json(checked(task_reference, "original task"))
    if task.get("phase") not in RELAY_PHASES or float(task.get("closing_reserve_seconds", 120)) <= 0:
        raise ValueError("unsupported relay phase or evaluation boundary")
    if type(reserved_tokens) is not int or reserved_tokens < 0 or not result_references:
        raise ValueError("invalid prior reservation or missing attempt evidence")
    seen, values = set(), []
    permitted = {"declared_training_exposure_not_reached", "evaluation_incomplete", "checkpoint_load_not_verified",
                 "causality_not_verified"}
    for reference in result_references:
        path = checked(reference, "prior result")
        if path in seen:
            raise ValueError("duplicate prior attempt in evaluation accounting")
        seen.add(path)
        value = _result(reference, task, task_reference["sha256"])
        if (value.get("status") != "budget_exhausted" or value.get("stop_reason") != "signal"
                or value.get("failure_class") or set(value.get("health", {}).get("blocking_reasons", [])) - permitted):
            raise ValueError("relay accepts only signal interruptions without scientific failures")
        causality = value.get("causality")
        if causality and causality.get("within_tolerance") is not True:
            raise ValueError("a failed causality check is not a signal-only interruption")
        if not causality and value.get("training", {}).get("segment_tokens", 0):
            raise ValueError("trained interrupted result lacks its causality qualification")
        spent = value.get("evaluation", {}).get("tokens")
        if type(spent) is not int or spent < 0:
            raise ValueError("missing observed evaluation exposure")
        values.append(value)
    previous = values[-1]
    checkpoint = previous.get("checkpoint")
    checked(checkpoint, "checkpoint")
    origin = task.get("evaluation_start_tokens", 0)
    tokens = checkpoint.get("tokens")
    if (type(tokens) is not int or not origin <= tokens <= task["target_tokens"]
            or tokens != previous.get("training", {}).get("tokens")
            or checkpoint.get("steps") != previous.get("training", {}).get("steps")):
        raise ValueError("checkpoint and interrupted exposure disagree")
    initial = None
    for reference in reversed(result_references):
        initial = initial_origin(reference, task, task_reference["sha256"])
        if initial is not None:
            break
    if initial is None:
        if tokens != origin:
            raise ValueError("cannot recreate original initial evaluation at a later checkpoint")
        initial = {"mode": "rerun", "exposure": origin}
    count, interval = task.get("eval_tokens", 16384), task.get("eval_interval_tokens", 1048576)
    intermediate = max(0, (task["target_tokens"]-1)//interval - tokens//interval) if interval > 0 else 0
    future = 32 + count * (1 + intermediate + (initial["mode"] == "rerun"))
    if task.get("allow_confirmation"):
        future += task.get("confirmation_tokens", 131072)
    prior = sum(value["evaluation"]["tokens"] for value in values)
    # Existing reservations remain charged. Only work beyond their remaining
    # coverage is added; repeated probes and partial final evaluations count.
    required = prior + 32 * len(values) + future
    return {"schema": RELAY_SCHEMA, "task_id": task["id"], "original_task": task_reference,
            "result": result_references[-1], "prior_results": result_references,
            "checkpoint": checkpoint, "source_identity": task["source_identity"],
            "initial_evaluation": initial, "prior_evaluation_reserved_tokens": reserved_tokens,
            "prior_observed_evaluation_tokens": prior, "prior_causality_upper_bound_tokens": 32*len(values),
            "new_attempt_evaluation_upper_bound_tokens": future,
            "evaluation_additional_reservation_tokens": max(0, required-reserved_tokens),
            "accounting": "retain all prior reservations; add only observed prior work plus remaining work exceeding them"}


def validate_receipt(args, worker):
    receipt = read_json(checked({"path": str(args.receipt), "sha256": args.receipt_sha256}, "receipt"))
    if receipt.get("schema") == "pilot-r3-stage-a-source-migration-v1":
        from pilotlab.source_migration3 import migration_inputs
        task = read_json(args.task)
        if (file_hash(__file__) != args.adapter_sha256 or task.get("stage_a_migration") !=
                {"path":str(Path(args.receipt).resolve()), "sha256":args.receipt_sha256}
                or Path(args.resume_checkpoint).resolve() != Path(receipt["checkpoint"]["path"]).resolve()
                or Path(args.output).resolve() == Path(receipt["result"]["path"]).resolve().parent):
            raise ValueError("migration command differs from its frozen task")
        _, previous = migration_inputs(task, receipt)
        worker.validate_task(task)
        worker.validate_source_identity(task, Path(args.data))
        return receipt, task, previous
    if receipt.get("schema") != RELAY_SCHEMA:
        return _validate_legacy_receipt(args, worker)
    if file_hash(__file__) != args.adapter_sha256:
        raise ValueError("adapter SHA256 mismatch")
    expected = build_relay_receipt(receipt["original_task"], receipt["prior_results"],
                                   receipt["prior_evaluation_reserved_tokens"])
    if receipt != expected:
        raise ValueError("relay receipt does not match recomputed evidence and incremental budget")
    task = read_json(checked(receipt["original_task"], "original task"))
    previous = read_json(checked(receipt["result"], "original result"))
    checkpoint = checked(receipt["checkpoint"], "checkpoint")
    if file_hash(args.task) != receipt["original_task"]["sha256"] or Path(args.resume_checkpoint).resolve() != checkpoint:
        raise ValueError("CLI task/checkpoint differs from bound relay")
    output = Path(args.output).resolve()
    for reference in receipt["prior_results"]:
        if output == Path(reference["path"]).parent or output.is_relative_to(Path(reference["path"]).parent):
            raise ValueError("relay output would modify a prior attempt")
    if (output/"result.json").exists() or (output/"checkpoint.json").exists():
        raise ValueError("relay requires a new attempt output")
    worker.validate_task(task)
    worker.validate_source_identity(task, Path(args.data))
    payload = worker._read_checkpoint(checkpoint, receipt["checkpoint"]["sha256"])
    worker.validate_checkpoint(payload, task, args.device)
    if (payload.get("task_id") != task["id"] or payload.get("initialization") != previous.get("initialization")
            or any(payload.get(key) != receipt["checkpoint"].get(key) for key in ("tokens", "steps", "state_sha256"))):
        raise ValueError("relay checkpoint payload differs from the original task/result")
    return receipt, task, previous


def inherited_selection(observation, origin_ref, task_id, exposure):
    initial = copy.deepcopy(observation)
    initial.update(tokens=0, seconds=0.0, original_observed_tokens=observation["tokens"],
        original_observed_seconds=observation.get("seconds"), exposure=exposure,
        inherited_from={**origin_ref, "field": "validation_initial", "task_id": task_id, "exposure": exposure},
        execution="inherited completed original evaluation; no evaluation at the resume checkpoint")
    return initial


def execute(args):
    # Deliberately import from the frozen worker PYTHONPATH, not this script's
    # controller snapshot.  validate_source_identity checks its actual path.
    from pilotlab import worker3 as worker

    receipt, task, previous = validate_receipt(args, worker)
    checkpoint = receipt["checkpoint"]
    migrating = receipt.get("schema") == "pilot-r3-stage-a-source-migration-v1"
    source_task = read_json(checked(receipt["original_task"], "original task")) if migrating else task
    original_evaluate, original_load = worker.evaluate, worker.load_checkpoint
    initial_deadline = args.deadline_unix - float(task.get("closing_reserve_seconds", 120))
    inherited = False
    initial_seen = False
    loaded = False

    def load_bound(reference, active_task, device, expected_sha=None):
        nonlocal loaded
        if Path(reference).resolve() != Path(checkpoint["path"]).resolve() or active_task != task:
            raise ValueError("worker attempted to load an unbound resume state")
        payload, recovery = original_load(reference, source_task, device, checkpoint["sha256"])
        if (recovery.get("fallback_used") or Path(recovery["path"]).resolve() != Path(checkpoint["path"]).resolve()
                or recovery["sha256"] != checkpoint["sha256"] or payload.get("task_id") != source_task["id"]
                or payload.get("tokens") != checkpoint["tokens"] or payload.get("steps") != checkpoint["steps"]):
            raise ValueError("bound resume forbids fallback or changed checkpoint exposure")
        initialization = previous.get("initialization")
        if migrating and receipt["mode"] == "checkpoint_only":
            from pilotlab.source_migration3 import read_bound
            initialization = read_bound(read_bound(source_task["stage_a_migration"])["result"])["initialization"]
        if migrating and (payload.get("state_sha256") != checkpoint["state_sha256"]
                          or payload.get("initialization") != initialization):
            raise ValueError("migration checkpoint state or initialization changed")
        loaded = True
        return payload, recovery

    def evaluate(model, store, active_task, device, deadline, *, split="selection", tokens=None):
        nonlocal inherited, initial_seen
        if deadline == initial_deadline and split == "selection" and tokens is None and not initial_seen:
            if not loaded or active_task != task:
                raise ValueError("unexpected repeated or unbound initial evaluation")
            initial_seen = True
            origin = receipt.get("initial_evaluation")
            if origin and origin["mode"] == "rerun":
                observed = original_evaluate(model, store, active_task, device, deadline, split=split, tokens=tokens)
                return dict(observed, exposure=origin["exposure"], execution="recomputed at unchanged original initial exposure")
            inherited = True
            return inherited_selection(origin["observation"] if origin else previous["validation_initial"],
                origin["result"] if origin else receipt["result"], task["id"], origin["exposure"] if origin else 0)
        return original_evaluate(model, store, active_task, device, deadline, split=split, tokens=tokens)

    worker.load_checkpoint, worker.evaluate = load_bound, evaluate
    try:
        result = worker.execute(args)
    finally:
        worker.load_checkpoint, worker.evaluate = original_load, original_evaluate
    if result.get("status") == "completed" and not initial_seen:
        raise ValueError("completed resume did not process the bound initial evaluation")
    worker_path = Path(worker.__file__).resolve()
    finish_result_metadata(result, task, receipt, previous, inherited=inherited,
        task_ref={"path": str(Path(args.task).resolve()), "sha256": file_hash(args.task)},
        receipt_ref={"path": str(Path(args.receipt).resolve()), "sha256": args.receipt_sha256},
        adapter_ref={"path": str(Path(__file__).resolve()), "sha256": args.adapter_sha256},
        worker_ref={"path": str(worker_path), "sha256": file_hash(worker_path)})
    worker.atomic_json(Path(args.output) / "result.json", result)
    return result


def finish_result_metadata(result, task, receipt, previous, *, inherited, task_ref, receipt_ref, adapter_ref, worker_ref):
    """Bind the executed adapter, evaluation accounting and verified checkpoint ancestry."""
    result["execution_adapter"] = {
        "schema": "pilot-r3-execution-adapter-v1", "adapter": adapter_ref,
        "receipt": receipt_ref, "worker": worker_ref,
        "source_identity_scope": "original frozen training core and checkpoint contract; execution adapter is identified separately",
        "initial_evaluation_inherited": inherited, "original_task": receipt["original_task"],
        "original_result": receipt["result"], "checkpoint": receipt["checkpoint"],
        "evaluation_additional_reservation_tokens": receipt["evaluation_additional_reservation_tokens"],
        "numerical_scope": "state loading is checked by the frozen worker; numerical trajectory thresholds and verdicts are unchanged"}
    if "evaluation" in result:
        prior_tokens = (receipt["prior_observed_evaluation_tokens"] if "prior_observed_evaluation_tokens" in receipt
                        else previous["evaluation"]["tokens"])
        result["evaluation"].update(prior_attempt_tokens=prior_tokens,
            cumulative_task_observed_tokens=prior_tokens + result["evaluation"]["tokens"],
            scope="new selection loss tokens in this attempt; inherited initial evaluation charged to the original attempt; excludes causality probes")
    if receipt["schema"] == "pilot-r3-stage-a-source-migration-v1":
        from pilotlab.source_migration3 import ADAPTER, migration_origin
        result["execution_adapter"].update(schema=ADAPTER, migration_mode=receipt["mode"],
            new_task=task_ref,
            source_identity_scope="old state and stage-start evidence retained under their source; updates use the new frozen source")
        if receipt["mode"] == "backward_range_repair":
            from pilotlab.source_migration3 import prior_evaluation_tokens
            prior = prior_evaluation_tokens(receipt)
            result["evaluation"].update(prior_attempt_tokens=prior,
                cumulative_task_observed_tokens=prior + result["evaluation"]["tokens"])
        if receipt["mode"] == "stage_parent":
            result["evaluation"].pop("prior_attempt_tokens", None)
            result["evaluation"]["cumulative_task_observed_tokens"] = result["evaluation"]["tokens"]
            result["evaluation"]["scope"] = "new stage evaluations from the migrated parent checkpoint"
        if result.get("status") == "completed":
            migration_origin(task, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "output", "data", "resume-checkpoint", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("receipt-sha256", "adapter-sha256", "device"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        from pilotlab import worker3 as worker
        signal.signal(signal.SIGTERM, worker.stop_requested)
        signal.signal(signal.SIGINT, worker.stop_requested)
        result = execute(args)
    except Exception as error:
        print(f"resume_adapter_refused: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    return 0 if result["status"] == "completed" else 2 if result["status"] == "budget_exhausted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
