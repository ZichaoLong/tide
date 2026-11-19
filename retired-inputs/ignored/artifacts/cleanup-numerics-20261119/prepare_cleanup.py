"""Prepare or revalidate one fixed cleanup manifest; never remove any file.

The parent agent owns live-process checks, review and any later execution.
Only small JSON records are hashed. Tensor payloads and profiler traces are
identified by prior recorded identities plus exact current lstat guards.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / "cleanup-plan.json"
PRIOR = "artifacts/disk-cleanup-20261119/remaining-priority-candidates.json"
PRIOR_SHA256 = "0d89563072ef528511bb4cc9240af1610d4c9975159e48679f25cf5bd5ce6b92"
EXPECTED_TENSOR_COUNT = 70
EXPECTED_TENSOR_ALLOCATED = 31_598_239_744
EXPECTED_PROFILER_ALLOCATED = 5_706_231_808
PROFILER_JOBS = (
    "agg-cost-half-fp32-v1",
    "agg-cost-source-fp32-v1",
    "training-cost-candidate-trace",
    "agg-cost-doubling-fp32-v1",
    "memory-cost-shared-v1",
    "memory-cost-x-v1",
)
FAILED_PROFILER = "artifacts/ffn-local-npu-v1/source/export_only_prof_dir"
BAD_COMPARISON_FIELDS = (
    "metadata_failures", "metadata_mismatches", "dtype_differences",
    "nonfinite", "support_differences", "support_mismatches",
    "threshold_failures", "numerical_failures", "work_differences",
)
GROUP_REASONS = {
    "training_restore": "The named restore-input bundle has a successful fresh-process local restoration and an actual checked optimizer update; cross-path failures and other anchors remain retained.",
    "reception_restore": "The named reception candidate's own fresh-process restoration passed at its original threshold; cross-candidate finite deviations are not reclassified.",
    "fp64_remaining": "This bounded CPU FP64 diagnostic's overall result and all three local restorations passed; retain the original complete small result and source/input identities.",
    "continuous": "The completed continuous run contains successful checkpoint data, next-forward and actual paired-update checks; this is retirement of its saved payload, not a new qualification claim.",
    "installed": "Only the named installed-wheel cases with successful fresh-process local recovery and actual fourth update are selected; placed BF16 failures and the original enclosing failed classification remain retained.",
    "profiler_vendor": "Intermediate profiler files from six completed_finite cost observations; retain costs.json, trace identities, summaries, launch/environment/log/status records and all unlisted files.",
    "profiler_export": "Chrome exports from the same six completed_finite observations; retain their recorded SHA256/size, event/category summaries and original timing boundary. Fine event ordering, shapes and device tracks will no longer be online after retirement.",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def checked_path(name, *, regular=True):
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and rel.parts and ".." not in rel.parts,
            f"not a normalized repository-relative path: {name}")
    require(str(rel) == name and rel.parts[0] == "artifacts",
            f"path outside the artifact scope: {name}")
    current = ROOT
    for index, part in enumerate(rel.parts):
        current = current / part
        info = current.lstat()
        require(not stat.S_ISLNK(info.st_mode), f"symlink path component: {current}")
        if index < len(rel.parts) - 1:
            require(stat.S_ISDIR(info.st_mode), f"non-directory path component: {current}")
    if regular:
        require(stat.S_ISREG(info.st_mode), f"not a regular file: {current}")
    return current, info


def guard(name):
    _, info = checked_path(name)
    require(info.st_nlink == 1, f"hard-linked candidate needs separate accounting: {name}")
    return dict(bytes=info.st_size, allocated_bytes=info.st_blocks * 512,
                device=info.st_dev, inode=info.st_ino, mtime_ns=info.st_mtime_ns,
                ctime_ns=info.st_ctime_ns, mode=info.st_mode, links=info.st_nlink)


def same_guard(name, expected):
    actual = guard(name)
    require(actual == expected, f"file identity/stat changed: {name}")


def read_small_record(name):
    path, info = checked_path(name)
    require(info.st_size <= 16 * 1024 * 1024, f"not a small JSON record: {name}")
    data = path.read_bytes()
    value = json.loads(data)
    require(isinstance(value, dict), f"expected JSON object: {name}")
    return value, dict(path=name, sha256=sha_bytes(data), bytes=len(data),
                       state=value.get("state"), stat_guard=guard(name))


def clean_comparison(value, label, *, exact=False):
    comparison = value.get("comparison", value)
    require(isinstance(comparison, dict), f"missing comparison: {label}")
    if exact:
        require(comparison.get("exact") is True, f"not exact: {label}")
    for field in BAD_COMPARISON_FIELDS:
        require(not comparison.get(field), f"retained comparison failed ({field}): {label}")
    require(comparison.get("finite", True) is True, f"nonfinite comparison: {label}")
    require(comparison.get("within_threshold", True) is True, f"outside threshold: {label}")
    return {key: comparison[key] for key in
            ("exact", "finite", "within_threshold", "atol", "rtol", "tolerance")
            if key in comparison}


def local_restore(value, label):
    require(value.get("state") == "within_threshold", f"local restore not qualified: {label}")
    require(value.get("actual_update") is True and value.get("config_exact") is True,
            f"missing actual update or exact configuration: {label}")
    require(value.get("source_exact") is True, f"restored source identity changed: {label}")
    checks = value["checks"]
    summary = {key: clean_comparison(checks[key], label + ":" + key,
                                    exact=key in ("saved_data", "next_forward"))
               for key in ("saved_data", "next_forward", "actual_update")}
    update = value["optimizer_update"]
    active = update["active_original_parameters"]
    require(update.get("checked") is True and active, f"unchecked optimizer update: {label}")
    require(all(update["after"][key] == update["before"][key] + 1 for key in active),
            f"optimizer step did not advance: {label}")
    return dict(state=value["state"], actual_update=True, source=value.get("source"),
                source_exact=True, bundle_sha256=value.get("bundle_sha256"),
                harness_sha256=value.get("harness_sha256"),
                package_sha256=value.get("package_identity", {}).get("sha256"),
                optimizer_active_parameters=len(active), checks=summary,
                original_tolerance=value.get("tolerance"))


def installed_restore(value, label):
    require(value.get("state") == "within_threshold" and value.get("fresh_process") is True,
            f"installed restoration not successful/fresh: {label}")
    require(value.get("config_exact") is True and value.get("none_gradients_preserved") is True,
            f"installed configuration/gradient connectivity failed: {label}")
    checks = {key: clean_comparison(value[key], label + ":" + key,
                                   exact=key in ("checkpoint_data", "next_forward"))
              for key in ("checkpoint_data", "next_forward", "gradients", "updated")}
    actual = value["actual_fourth_update"]
    require(actual.get("input_gradient_finite") is True and actual.get("changed_state_tensors", 0) > 0,
            f"actual fourth update missing: {label}")
    require(value["installation"].get("state") == "passed", f"installation check failed: {label}")
    return dict(state=value["state"], fresh_process=True, checks=checks,
                actual_fourth_update={key: actual.get(key) for key in
                    ("changed_state_tensors", "input_gradient_finite", "optimizer_step_counts")},
                wheel_sha256=value["installation"].get("wheel_sha256"))


def continuous_restore(value, label):
    require(value.get("state") == "completed", f"continuous run incomplete: {label}")
    checkpoint = value["checkpoint"]
    require(checkpoint.get("state") == "within_threshold", f"checkpoint recovery failed: {label}")
    checks = {}
    for key in ("saved_data", "next_forward", "next_state", "next_flush_result",
                "input_gradients", "parameter_gradients", "updated_parameters",
                "updated_optimizer", "updated_state"):
        if checkpoint.get(key) is not None:
            checks[key] = clean_comparison(checkpoint[key], label + ":" + key,
                                           exact=key in ("saved_data", "next_forward", "next_state", "next_flush_result"))
    require(all(key in checks for key in ("saved_data", "next_forward", "updated_parameters", "updated_optimizer", "updated_state")),
            f"incomplete paired update checks: {label}")
    return dict(state=value["state"], source=value.get("source"),
                checkpoint_state=checkpoint["state"], checks=checks,
                boundary=checkpoint.get("boundary"), updates=value.get("arguments", {}).get("updates"))


def job_root(path):
    current = (ROOT / path).parent
    while current != ROOT:
        if (current / "launch.json").is_file() and (current / "status.json").is_file():
            return relative(current)
        current = current.parent
    raise RuntimeError(f"no original launch/status pair for {path}")


def enumerate_regular(directory):
    path, info = checked_path(directory, regular=False)
    require(stat.S_ISDIR(info.st_mode), f"not a directory: {directory}")
    output = []
    for parent, directories, names in os.walk(path, followlinks=False):
        for name in directories:
            child = Path(parent) / name
            require(not child.is_symlink(), f"symlink directory: {child}")
        for name in names:
            child = Path(parent) / name
            item = relative(child)
            checked_path(item)
            output.append(item)
    return sorted(output)


def atomic_write(value):
    require(not PLAN.exists(), f"plan exists; validate it instead of replacing: {PLAN}")
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".cleanup-plan.", dir=HERE)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, PLAN)
        descriptor = os.open(HERE, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only our own incomplete staged plan.
    return sha_bytes(data)


def preserve_small_records(records):
    """Save original JSON bytes only under this cleanup artifact directory."""
    copies = []
    for row in sorted(records, key=lambda item: item["path"]):
        source, _ = checked_path(row["path"])
        data = source.read_bytes()
        require(sha_bytes(data) == row["sha256"], f"record changed before preservation: {row['path']}")
        target = HERE / "preserved-records" / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(not target.is_symlink() and target.read_bytes() == data,
                    f"existing preserved record differs: {target}")
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=".record.", dir=target.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
                descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)  # Only our own incomplete staged copy.
        copies.append(dict(original_path=row["path"], preserved_path=relative(target),
                           sha256=row["sha256"], bytes=len(data)))
    return copies


def prepare():
    prior, prior_identity = read_small_record(PRIOR)
    require(prior_identity["sha256"] == PRIOR_SHA256, "prior 70-file recommendation identity changed")
    require(prior["count"] == EXPECTED_TENSOR_COUNT and len(prior["files"]) == EXPECTED_TENSOR_COUNT,
            "prior candidate count changed")
    records, values, jobs, files = {}, {}, {}, []

    def record(name):
        if name not in records:
            values[name], records[name] = read_small_record(name)
        return values[name]

    for previous in prior["checked_result_records"]:
        value = record(previous["path"])
        require(records[previous["path"]]["sha256"] == previous["sha256"],
                f"prior recovery evidence changed: {previous['path']}")
        require(value.get("state") == previous["state"], f"prior recovery state changed: {previous['path']}")

    def original_job(name):
        if name in jobs:
            return jobs[name]
        launch = record(name + "/launch.json")
        terminal = record(name + "/status.json")
        require(terminal.get("state") in ("passed", "failed", "terminated"), f"nonterminal original job: {name}")
        require(isinstance(terminal.get("exit_code"), int), f"missing original exit code: {name}")
        source = launch.get("source")
        commit = None
        if isinstance(source, str) and re.fullmatch(r"[0-9a-f]{8,40}", source):
            commit = subprocess.check_output(["git", "rev-parse", "--verify", source + "^{commit}"], cwd=ROOT, text=True).strip()
        jobs[name] = dict(source=source, resolved_source_commit=commit,
                          source_path=launch.get("source_path"), unit=launch.get("unit", launch.get("service")),
                          original_terminal={key: terminal.get(key) for key in
                            ("state", "source", "exit_code", "started", "finished", "elapsed", "descendant_supervision")},
                          launch_record=name + "/launch.json", terminal_record=name + "/status.json",
                          qualification_note="Original terminal and numerical classifications are preserved; local recovery eligibility does not turn an enclosing failed run into passed.")
        for optional in ("environment.json", "terminal-audit.json", "postflight.json"):
            if (ROOT / name / optional).is_file():
                record(name + "/" + optional)
        return jobs[name]

    for previous in prior["files"]:
        name, group = previous["path"], previous["group"]
        require(group in GROUP_REASONS and not group.startswith("profiler"), f"unknown tensor group: {group}")
        require(name.endswith(".pt") and "placed-bf16" not in Path(name).parts[-2:],
                f"unexpected/protected tensor path: {name}")
        current = guard(name)
        for key in ("bytes", "allocated_bytes", "device", "inode", "mtime_ns"):
            require(current[key] == previous[key], f"prior tensor stat changed ({key}): {name}")
        parent = str(PurePosixPath(name).parent)
        evidence = []
        if group in ("training_restore", "reception_restore"):
            evidence = [parent + "/restore.json"]
            eligibility = local_restore(record(evidence[0]), evidence[0])
        elif group == "installed":
            evidence = [parent + "/resume.json"]
            eligibility = installed_restore(record(evidence[0]), evidence[0])
        elif group == "continuous":
            evidence = [parent + "/continuous.json"]
            eligibility = continuous_restore(record(evidence[0]), evidence[0])
        elif group == "fp64_remaining":
            result_root = "artifacts/training-paths-prefix-fp64/results"
            evidence = [result_root + "/result.json"] + [result_root + "/" + variant + "/restore.json" for variant in ("baseline", "prefix", "reception")]
            whole = record(evidence[0])
            require(whole.get("state") == "within_threshold" and whole["result"].get("state") == "within_threshold", "FP64 source diagnostic did not pass")
            require(all(c.get("state") == "within_threshold" for c in whole["result"]["candidates"]), "FP64 source candidate failed")
            eligibility = dict(state="within_threshold", bounded_cpu_fp64=True,
                               restorations={path: local_restore(record(path), path) for path in evidence[1:]})
        original = job_root(name)
        original_job(original)
        files.append(dict(path=name, kind="tensor", group=group, stat_guard=current,
                          original_job=original, eligibility=eligibility, retained_evidence=evidence,
                          prior_recommendation_stat_verified=True,
                          payload_hash_policy="No tensor was read or rehashed; recorded original bundle/input hashes remain in retained evidence."))

    require(sum(row["stat_guard"]["allocated_bytes"] for row in files) == EXPECTED_TENSOR_ALLOCATED,
            "tensor allocated-byte total changed")

    profiler_evidence = {}
    for job in PROFILER_JOBS:
        base = "artifacts/" + job
        original = original_job(base)
        require(original["original_terminal"]["state"] == "passed" and original["original_terminal"]["exit_code"] == 0,
                f"profiler job not successful: {job}")
        costs = record(base + "/costs.json")
        require(costs.get("state") == "completed_finite", f"profiler cost run not completed_finite: {job}")
        exports, traces = {}, []
        for observation in costs["observations"]:
            trace = observation.get("diagnostics", {}).get("trace")
            if not trace:
                continue
            require(trace.get("state") == "completed_finite", f"incomplete profiler observation: {job}")
            identity = trace["trace"]
            export = relative(Path(identity["path"]))
            require(Path(export).parent == Path(base) and export.endswith("-trace.json"), f"unexpected trace export path: {export}")
            require(export not in exports, f"duplicate trace export: {export}")
            summary = trace["summary"]
            require(summary.get("complete_event_count", 0) > 0 and summary.get("categories") and summary.get("top_by_category"), f"missing trace summary: {export}")
            exports[export] = identity
            traces.append(dict(recorded_export=identity, boundary=trace.get("boundary"), summary=summary))
        on_disk_exports = {relative(path) for path in (ROOT / base).glob("*-trace.json")}
        require(set(exports) == on_disk_exports and exports, f"cost record/export listing mismatch: {job}")
        profiler_evidence[job] = dict(costs_record=base + "/costs.json", traces=traces,
                                     preserve_reason="Keep complete costs.json plus launch, environment, logs and terminal records; all nonlisted files are excluded from cleanup.")
        vendor = base + "/source/export_only_prof_dir"
        for name in enumerate_regular(vendor):
            require(not name.endswith((".pt", ".f32")), f"unexpected protected tensor in profiler tree: {name}")
            files.append(dict(path=name, kind="profiler", group="profiler_vendor", original_job=base,
                              stat_guard=guard(name), retained_evidence=[base + "/costs.json"],
                              payload_hash_policy="Profiler payload not rehashed; exact lstat guard and original export identities retained."))
        for name, identity in sorted(exports.items()):
            current = guard(name)
            require(current["bytes"] == identity["bytes"], f"export size differs from recorded identity: {name}")
            files.append(dict(path=name, kind="profiler", group="profiler_export", original_job=base,
                              stat_guard=current, recorded_sha256=identity["sha256"],
                              retained_evidence=[base + "/costs.json"],
                              payload_hash_policy="Original recorded SHA256 preserved; no new trace hash computed."))

    paths = [row["path"] for row in files]
    require(len(paths) == len(set(paths)), "candidate paths overlap")
    identities = [(row["stat_guard"]["device"], row["stat_guard"]["inode"]) for row in files]
    require(len(identities) == len(set(identities)), "candidate inode accounting overlaps")
    require(not set(paths) & set(records), "cleanup candidate overlaps retained evidence")
    require(sum(row["stat_guard"]["allocated_bytes"] for row in files if row["kind"] == "profiler") == EXPECTED_PROFILER_ALLOCATED,
            "profiler allocated-byte total changed; inspect instead of extending automatically")

    selected = set(paths)
    retained_tensors = []
    for parent, directories, names in os.walk(ROOT / "artifacts", followlinks=False):
        directories[:] = [name for name in directories if Path(parent) / name != HERE]
        for name in names:
            if not name.endswith((".pt", ".f32")):
                continue
            path = relative(Path(parent) / name)
            if path not in selected:
                retained_tensors.append(dict(path=path, stat_guard=guard(path), reason="Not one of the exact 70 authorized successful-local-recovery payloads; preserve for post-cleanup identity verification."))
    failed_profiler = [dict(path=path, stat_guard=guard(path)) for path in enumerate_regular(FAILED_PROFILER)]
    require(all(not row["path"].startswith("artifacts/ffn-local-npu-v1/") for row in files), "failed FFN profiler must stay protected")
    groups = {}
    for group in sorted({row["group"] for row in files}):
        subset = [row for row in files if row["group"] == group]
        groups[group] = dict(count=len(subset), allocated_bytes=sum(row["stat_guard"]["allocated_bytes"] for row in subset),
                             apparent_bytes=sum(row["stat_guard"]["bytes"] for row in subset),
                             retirement_reason=GROUP_REASONS[group])
    preserved = preserve_small_records([prior_identity, *records.values()])
    return dict(schema="tide-cleanup-numerics-plan-v1", state="prepared_not_executed",
                created_utc=datetime.now(timezone.utc).isoformat(), repository=str(ROOT),
                source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                preparer=dict(path=relative(Path(__file__).resolve()), sha256=sha_bytes(Path(__file__).read_bytes())),
                prior_recommendation=prior_identity,
                scope="Exactly 70 previously enumerated payloads and the enumerated vendor/export profiler files from six successful runs; no recursive directory deletion.",
                execution_requirements=["Root agent must review and freeze this plan SHA256.", "Root agent must refresh live process/service/open-file dependencies before execution.", "Revalidate every candidate and protected/evidence identity; unlink only individual listed regular files after per-file lstat guards.", "Keep all noncandidate files and directories, including costs.json, logs, launch/status and failed FFN profiler.", "Do not convert original failed job/qualification states to passed."],
                qualification_note=prior["qualification_note"],
                totals=dict(count=len(files), tensor_count=EXPECTED_TENSOR_COUNT,
                            tensor_allocated_bytes=EXPECTED_TENSOR_ALLOCATED,
                            profiler_count=sum(row["kind"] == "profiler" for row in files),
                            profiler_allocated_bytes=EXPECTED_PROFILER_ALLOCATED,
                            allocated_bytes=sum(row["stat_guard"]["allocated_bytes"] for row in files)),
                groups=groups, candidates=sorted(files, key=lambda row: row["path"]),
                original_jobs=jobs, retained_records=sorted(records.values(), key=lambda row: row["path"]),
                preserved_records=preserved,
                original_57_result_records=[row["path"] for row in prior["checked_result_records"]],
                profiler_summaries=profiler_evidence,
                protected=dict(policy="Everything not listed as a candidate is excluded by default.",
                               retained_tensors=sorted(retained_tensors, key=lambda row: row["path"]),
                               failed_ffn_profiler=failed_profiler,
                               external_and_source_boundaries=[".git", "src", "math", "docs", "configs", "benchmarks", "scripts", "tests", "native", ".venv-instances", "downstream/public/reference paths"],
                               unique_input_boundaries=["artifacts/memory-candidates-20261119/development-snapshot.patch", "artifacts/worktrees/capacity-bindings/artifacts/native-development", "artifacts/worktrees/agg-reception-restore/artifacts"],
                               retained_reason="Cross-path failures, unselected anchors, private diagnostic inputs, raw .f32 data and original small results remain available for bounded numerical investigation."))


def validate_existing(expected_sha):
    data = PLAN.read_bytes()
    digest = sha_bytes(data)
    require(expected_sha == digest, "fixed cleanup-plan SHA256 mismatch")
    plan = json.loads(data)
    require(plan["state"] == "prepared_not_executed" and plan["repository"] == str(ROOT), "plan identity mismatch")
    require(plan["preparer"]["sha256"] == sha_bytes(Path(__file__).read_bytes()), "preparation script changed")
    prior, identity = read_small_record(PRIOR)
    require(identity["sha256"] == PRIOR_SHA256 == plan["prior_recommendation"]["sha256"], "prior candidate manifest changed")
    for row in plan["candidates"]:
        same_guard(row["path"], row["stat_guard"])
    for row in plan["retained_records"]:
        _, identity = read_small_record(row["path"])
        require(identity["sha256"] == row["sha256"], f"retained evidence changed: {row['path']}")
        same_guard(row["path"], row["stat_guard"])
    for row in plan["preserved_records"]:
        _, identity = read_small_record(row["preserved_path"])
        require(identity["sha256"] == row["sha256"], f"preserved record changed: {row['preserved_path']}")
    for group in ("retained_tensors", "failed_ffn_profiler"):
        for row in plan["protected"][group]:
            same_guard(row["path"], row["stat_guard"])
    return dict(state="validated_no_deletion", plan_sha256=digest, totals=plan["totals"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-existing", action="store_true")
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()
    if args.validate_existing:
        require(args.expected_plan_sha256 is not None, "validation requires the reviewed plan SHA256")
        result = validate_existing(args.expected_plan_sha256)
    else:
        require(args.expected_plan_sha256 is None, "expected plan SHA applies only to validation")
        plan = prepare()
        digest = atomic_write(plan)
        result = dict(state="prepared_no_deletion", plan=relative(PLAN), plan_sha256=digest, totals=plan["totals"], groups=plan["groups"])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
