"""Run untouched members of the original catalog on a repaired source, A only.

Old exposed subjects remain bound to their original records. This entry never
imports their scientific verdicts or checkpoints, and cannot release stage B.
The original full-catalog audit is retained even when this finite subset ends.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import signal

from .io import atomic_json, sha256
from .round3 import Campaign
from .scheduler3 import Blocked, freeze_json, live, read


def bound(ref):
    if sha256(ref["path"]) != ref["sha256"]:
        raise ValueError("stage-A revision input changed: " + ref["path"])
    return read(ref["path"])


def foundation_files(identity):
    return {path.split("/foundation/")[1]:digest for path,digest in identity["source_files"].items()
            if "/foundation/" in path}


def exposed_models(plan, state, selected):
    def dispatched(records):
        return {row["model_id"] for row in records.values()
                if row.get("attempts") or row.get("prior_attempts") or row.get("result_path") or row.get("imported_completion")}
    exposed = dispatched(state["tasks"]) | set(selected)
    exposed.update(key for key,value in plan["config"].get("prior_evaluation_reservations", {}).items() if value > 0)
    if plan.get("stage_a_transfer"):
        exposed.update(bound(plan["stage_a_transfer"])["model_ids"])
    ancestry = plan.get("qualification_transfer", {}).get("old_evaluation_accounting")
    if ancestry:
        exposed.update(dispatched(bound(ancestry)["tasks"]))
    return exposed


def verify_controller_relay(plan):
    """Bind a controller-only continuation to the very same screened A experiment."""
    worker = bound(plan["worker_prepared"])
    original = bound(worker["campaign_plan"])
    if original.get("resume_import"):
        raise ValueError("expected the initial screened worker plan")
    scope = verify_scope(original)
    imported = bound(plan["resume_import"])
    prior = bound(imported["old_prepared"])
    if (prior.get("worker_prepared", imported["old_prepared"]) != plan["worker_prepared"]
            or imported["worker_source_identity"] != worker["source_identity"]
            or original["source_identity"] != worker["source_identity"]):
        raise ValueError("relay worker provenance differs from the screened experiment")
    runtime = {"config", "worker_prepared", "resume_import", "controller_source_identity",
               "resource_policy", "storage_observation", "budget_remaining_at_freeze"}
    if {k:v for k,v in original.items() if k not in runtime} != {k:v for k,v in plan.items() if k not in runtime}:
        raise ValueError("controller relay changed the experiment or its A-only scope")
    config = plan["config"]
    scheduling = {"source", "foundation", "worker_source", "worker_foundation", "run_id",
                  "resume_adapter", "resume_import", "max_workers", "max_cpu_threads", "host_cpu_affinity",
                  "dynamic_probe_owners", "dynamic_owner_max_total_workers", "dynamic_owner_max_commands"}
    if ({k:v for k,v in config.items() if k not in scheduling}
            != {k:v for k,v in original["config"].items() if k not in scheduling}
            or config.get("worker_source") != worker["source"]
            or config.get("worker_foundation") != worker["foundation"]
            or config.get("resume_import") != plan["resume_import"]):
        raise ValueError("controller relay changed the worker, runtime contract or budget")
    return scope


def verify_scope(plan):
    if plan.get("stage_a_migration"):
        from .stage_a_continued3 import verify_scope as verify_migration
        return verify_migration(plan)
    if plan.get("controller_source_identity") and plan.get("resume_import") and plan.get("worker_prepared"):
        return verify_controller_relay(plan)
    scope = plan["stage_a_revision"]
    if scope.get("policy") != "unstarted_original_subjects_only_no_stage_b":
        raise ValueError("unsupported source-revision scope")
    old = bound(scope["original_plan"])
    state = bound(scope["original_scheduler"])
    selected = bound(scope["original_selected"])
    audit = bound(scope["original_audit"])
    if (state.get("status") != "a_blocked_needs_repair" or state.get("stage_b_gate") != "closed"
            or audit.get("status") != "passed" or audit.get("findings")
            or audit["execution"]["scheduler"]["sha256"] != scope["original_scheduler"]["sha256"]
            or any(audit["lifecycle"][key].get("released") is not True for key in ("processes", "cgroup", "queue"))):
        raise ValueError("original campaign lacks bound terminal release")
    if any(live(value) for value in [state.get("controller_identity"), *state.get("all_seen_processes", {}).values()]):
        raise ValueError("original campaign still owns live processes")
    catalog = {row["id"]:row for row in old["subjects"]}
    if len(catalog) != 344 or len(old["subjects"]) != 344:
        raise ValueError("expected the original finite 344-subject catalog")
    exposed = exposed_models(old, state, selected)
    deferred = sorted(set(catalog) & exposed)
    expected = set(catalog) - exposed
    actual = [row["id"] for row in plan["subjects"]]
    if (len(actual) != len(set(actual)) or set(actual) != expected or scope["deferred_subject_ids"] != deferred
            or any(row != catalog[row["id"]] for row in plan["subjects"])):
        raise ValueError("revision must contain exactly unchanged, previously unstarted subjects")
    if any(key in plan for key in ("stage_a_transfer", "qualification_transfer", "resume_import", "worker_prepared")):
        raise ValueError("this revision cannot transfer old qualification or training state")
    config = plan["config"]
    if any(key in config for key in ("worker_source", "worker_foundation", "resume_import", "resume_adapter")):
        raise ValueError("revision must execute the new frozen training source")
    for key in ("budget_ledger", "deadline_unix", "python", "data"):
        if config[key] != old["config"][key]:
            raise ValueError("revision changed original budget/deadline/environment/data")
    screen = bound(scope["mechanism_screen"])
    screen_plan = bound(screen["plan"])
    if screen.get("status") != "passed_mechanism_screen" or screen.get("remaining_descendants"):
        raise ValueError("repaired-source mechanism screen has not completed")
    if foundation_files(plan["source_identity"]) != foundation_files(screen_plan["source_identity"]):
        raise ValueError("foundation differs from the screened source")
    if set(screen["result_files"]) != {task["id"] for task in screen_plan["tasks"]}:
        raise ValueError("mechanism screen does not cover its finite task table")
    for ref in screen["result_files"].values():
        result = bound(ref)
        if (result.get("status") != "completed" or result.get("health", {}).get("eligible") is not True
                or not result["recovery"]["trajectory"]["within_tolerance"]
                or any(not row["original_threshold_gate"]["within_tolerance"] for row in result["fp64"])):
            raise ValueError("screen contains an unresolved repair qualification failure")
    return scope


def run_stage_a(campaign, scope):
    """Use the existing full tuning/continuation/control pipeline, stop before B."""
    from .round3 import audit_stage_a
    if campaign.supervisor.state.get("status") == "completed":
        # Campaign's completed-only branch verifies artifacts without running.
        return campaign.run()
    for subject in campaign.subjects:
        subject["effective_key"] = subject["model_key"]
    campaign.train_models(campaign.subjects)
    controls = campaign.controls()
    campaign.models += controls
    campaign.profile_models(controls)
    if not campaign.forecast():
        raise Blocked("resolved_controls_exceed_remaining_storage_or_device_budget")
    campaign.train_models(controls)
    manifest = campaign.manifest()
    freeze_json(campaign.output / "model-manifest.json", manifest)
    report = audit_stage_a(manifest, campaign.selected, campaign.matches, remaining_resources_ok=campaign.forecast())
    path = freeze_json(campaign.output / "stage-a-report.json", report)
    # Preserve the full 344-model gate. Only its expected missing-catalog
    # finding may remain when the independently bounded subset completes.
    allowed = "subject_catalog_not_344_unique_effective_models"
    if any(reason != allowed for reason in report["blocking_reasons"]) and not (
            campaign.plan.get("stage_a_migration") and campaign.review_source_chains(report)):
        raise Blocked(f'stage_a_subset_audit_failed:{report["blocking_reasons"][:3]}')
    summary = {"schema":"pilot-r3-final-summary-v1", "status":"completed",
        "scope":"stage_a_subset_only", "subject_count":len(campaign.subjects), "control_count":len(controls),
        "stage_a_report":str(path), "stage_a_report_sha256":sha256(path),
        "global_stage_a_status":report["status"], "global_stage_a_blockers":report["blocking_reasons"],
        "deferred_subject_ids":scope["deferred_subject_ids"], "stage_b_gate":"closed", "formal_trajectories":0}
    atomic_json(campaign.output / "report.json", summary)
    campaign.supervisor.state["stage_b_gate"] = "closed"
    campaign.supervisor.state["terminal_report_sha256"] = sha256(campaign.output / "report.json")
    campaign.supervisor.finish("completed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(argv)
    plan = read(args.plan)
    scope = verify_scope(plan)
    if plan.get("stage_a_migration"):
        from .stage_a_continued3 import MigrationCampaign
        campaign = MigrationCampaign(plan, args.output)
    else:
        campaign = Campaign(plan,args.output)
    signal.signal(signal.SIGTERM,campaign.supervisor.request_stop)
    signal.signal(signal.SIGINT,campaign.supervisor.request_stop)
    try:
        run_stage_a(campaign,scope)
        return 0
    except Exception as error:
        import traceback
        traceback.print_exc()
        campaign.supervisor.stop_reason = campaign.supervisor.stop_reason or f"{type(error).__name__}: {error}"
        campaign.supervisor.stop_all()
        atomic_json(campaign.output / "report.json", {"schema":"pilot-r3-final-summary-v1",
            "status":campaign.supervisor.state["status"], "scope":"stage_a_subset_only",
            "blocking_reason":str(error), "stage_b_gate":"closed", "counts":campaign.supervisor.state["counts"],
            "deferred_subject_ids":scope["deferred_subject_ids"], "traceback":traceback.format_exc()})
        return 1
    finally:
        campaign.supervisor.close()


if __name__ == "__main__":
    raise SystemExit(main())
