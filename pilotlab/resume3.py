"""Bound adoption of audited round-three attempts; no mutation of old records."""
from pathlib import Path

from .io import sha256
from .scheduler3 import Blocked, evaluate_outcome, read


def bound(reference):
    path = Path(reference["path"])
    if sha256(path) != reference["sha256"]:
        raise Blocked(f"resume_reference_changed:{path}")
    return read(path)


def imported_entry(config, task_id, _seen=None, *, read_bound=bound):
    seen = set() if _seen is None else set(_seen)
    path = str(Path(config["resume_import"]["path"]).resolve())
    if path in seen or len(seen) >= 64:
        raise Blocked("cyclic_or_deep_resume_import")
    seen.add(path)
    manifest = read_bound(config["resume_import"])
    if manifest.get("schema") != "pilot-r3-resume-import-v1":
        raise Blocked("unsupported_resume_import")
    # The supplementary review retains the failed historical audit and all
    # conservative charges. It does not turn old stopped workers into passes.
    read_bound(manifest["reconciliation"])
    row = manifest["tasks"][task_id]
    task = read_bound(row["task"])
    if task["id"] != task_id or task.get("source_identity") != manifest["worker_source_identity"]:
        raise Blocked("imported_task_identity_mismatch")
    result = read_bound(row["result"]) if row.get("result") else None
    if result and (result.get("task_sha256") != row["task"]["sha256"]
                   or result.get("source_identity") != task["source_identity"]):
        raise Blocked("imported_result_identity_mismatch")
    inherited = row.get("inherited_from")
    if inherited:
        if inherited.get("import") not in manifest.get("prior_imports", []):
            raise Blocked("unbound_parent_resume_import")
        ancestor, original, value = imported_entry({"resume_import": inherited["import"]}, task_id, seen, read_bound=read_bound)
        if (inherited.get("task_id") != task_id or ancestor["disposition"] != "completed"
                or row["disposition"] != "completed" or original != task or value != result
                or ancestor["result"] != row["result"]):
            raise Blocked("recursive_imported_completion_changed")
    return row, task, result


def imported_attempt(config, task_id, attempt_path, *, read_bound=bound):
    """Return the exact evidence for one historical attempt, never the latest result."""
    row, task, _ = imported_entry(config, task_id, read_bound=read_bound)
    candidates = row.get("attempt_evidence")
    if candidates is None:
        candidates = [{key: row.get(key) for key in
                       ("attempt_path", "task", "result", "worker_return_code", "disposition", "group")}]
    found = [entry for entry in candidates if entry.get("attempt_path")
             and Path(entry["attempt_path"]).resolve() == Path(attempt_path).resolve()]
    if len(found) != 1:
        raise Blocked("historical_attempt_evidence_missing_or_ambiguous")
    entry = found[0]
    original = read_bound(entry["task"])
    result = read_bound(entry["result"]) if entry.get("result") else None
    if original != task or (result and (result.get("id") != task_id
            or result.get("task_sha256") != entry["task"]["sha256"]
            or result.get("source_identity") != task.get("source_identity"))):
        raise Blocked("historical_attempt_identity_changed")
    if entry.get("group"):
        read_bound(entry["group"])
    return entry, original, result


def retirement_references(manifest, _seen=None):
    """Verified, deduplicated retirement records from the full import ancestry."""
    seen = set() if _seen is None else set(_seen)
    references = list(manifest.get("prior_retirements", []))
    if manifest.get("prior_retirement"):
        references.append(manifest["prior_retirement"])
    for parent in manifest.get("prior_imports", []):
        path = str(Path(parent["path"]).resolve())
        if path in seen or len(seen) >= 64:
            raise Blocked("cyclic_or_deep_retirement_import")
        references += retirement_references(bound(parent), seen | {path})
    unique = {}
    for reference in references:
        value = bound(reference)
        if value.get("schema") != "pilot-r3-retirement-v1":
            raise Blocked("invalid_prior_retirement")
        path = str(Path(reference["path"]).resolve())
        if path in unique and unique[path] != reference:
            raise Blocked("conflicting_retirement_digests")
        unique[path] = reference
    return list(unique.values())


def verify_imported_completion(config, task, record):
    row, original, result = imported_entry(config, task["id"])
    if (row["disposition"] != "completed" or original != task
            or record.get("result_path") != row["result"]["path"]
            or record.get("result_sha256") != row["result"]["sha256"]
            or evaluate_outcome(task, result, row["worker_return_code"])[0] != "completed"):
        raise Blocked("imported_completion_not_verified")
    return result


def resume_reservation(config, task, record):
    """Return the additional reservation for the one audited resumed attempt."""
    row, original, result = imported_entry(config, task["id"])
    if (original != task or record.get("attempts") or row["disposition"] not in {"resume", "not_started", "retry"}
            or record.get("evaluation_reserved_tokens", 0) != row["evaluation_reserved_tokens"]):
        raise Blocked("resume_reservation_contract_changed")
    extra = row["additional_evaluation_reserved_tokens"]
    if type(extra) is not int or extra < 0:
        raise Blocked("invalid_resume_reservation")
    if row["disposition"] == "resume":
        receipt = bound(row["resume_receipt"])
        if (receipt["evaluation_additional_reservation_tokens"] != extra
                or receipt["original_task"] != row["task"] or receipt["result"] != row["result"]
                or receipt["checkpoint"] != row["checkpoint"]):
            raise Blocked("resume_receipt_budget_binding_mismatch")
    elif row.get("relay_version") == 2:
        from .scheduler3 import evaluation_bound
        expected = (evaluation_bound(task) if row["disposition"] == "retry" else
                    max(0, evaluation_bound(task)-row["evaluation_reserved_tokens"]))
        if extra != expected or row["disposition"] == "not_started" and result is not None:
            raise Blocked("relay_retry_or_unstarted_reservation_mismatch")
    elif extra or result is not None:
        raise Blocked("unstarted_import_has_unexpected_cost_or_result")
    return extra


def verify_adapter_result(config, task, record, result):
    if not record.get("resume_receipt"):
        return
    row, original, _ = imported_entry(config, task["id"])
    receipt = bound(row["resume_receipt"])
    adapter = result.get("execution_adapter", {})
    if (original != task or adapter.get("receipt") != row["resume_receipt"]
            or adapter.get("adapter") != config["resume_adapter"]
            or result.get("resume", {}).get("sha256") != receipt["checkpoint"]["sha256"]
            or result.get("resume", {}).get("from_tokens") != receipt["checkpoint"]["tokens"]):
        raise Blocked("resume_execution_adapter_binding_mismatch")
