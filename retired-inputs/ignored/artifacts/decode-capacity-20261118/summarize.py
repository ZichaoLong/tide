"""Standard-library snapshot of the independently audited decode-capacity search.

This does not submit jobs, run models, refresh audit.json, or certify terminal
process ownership. A current confirmation can win only when its independent
audit still matches the raw files and it completed within every card's budget.
Historical observations remain a separate reference population.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
RUNNING = {"prepared", "starting", "running", "queued", "waiting", "submitted"}
RAW_FILES = ("inference.json", "reference.json", "status.json", "launch.json", "queue.json", "environment.json")


def relative(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load(path, optional=False):
    if optional and not path.exists():
        return None, None
    data = path.read_bytes()
    return json.loads(data), dict(path=relative(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def same_file(left, right):
    return bool(left and right and left["sha256"] == right["sha256"] and left["bytes"] == right["bytes"])


def atomic(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".summary.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def series_key(scale, kind, cards):
    return f"{scale}/{kind}/{cards}"


def final_state_memory(observations, batch, diagnostics):
    """Untimed final-state sizes; never substitute these for allocator peaks."""
    totals = [(o.get("state_memory") or {}).get("retained_storage_bytes") for o in observations]
    diagnostic_total = ((diagnostics or {}).get("state_memory") or {}).get("retained_storage_bytes")
    valid_batch = isinstance(batch, int) and not isinstance(batch, bool) and batch > 0
    normalized = [value / batch if value is not None and valid_batch else None for value in totals]
    median = statistics.median(totals) if totals and all(value is not None for value in totals) else None
    return dict(
        scope="final private-state unique retained storage across all devices, including view tails; not peak allocated/reserved memory and not a per-card value",
        batch=batch,
        per_batch_item_definition="aggregate retained bytes divided by B, including shared metadata; arithmetic normalization, not a measured individual sequence allocation",
        normal_sample_final_state_positions=[o.get("final_state_position") for o in observations],
        normal_sample_total_retained_storage_bytes=totals,
        normal_sample_retained_storage_bytes_per_batch_item=normalized,
        median_total_retained_storage_bytes=median,
        median_retained_storage_bytes_per_batch_item=median / batch if median is not None and valid_batch else None,
        diagnostic_final_state_position=(diagnostics or {}).get("final_state_position"),
        diagnostic_total_retained_storage_bytes=diagnostic_total,
        diagnostic_retained_storage_bytes_per_batch_item=(diagnostic_total / batch
            if diagnostic_total is not None and valid_batch else None),
        diagnostic_scope="separate untimed diagnostic replay; not pooled with normal samples")


def metrics(observations, batch=None, diagnostics=None):
    if not observations:
        return None
    rates = [o["stable_decode"]["positions_per_second"] for o in observations]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in rates):
        raise ValueError("nonfinite/nonpositive stable throughput")
    devices = sorted({d for o in observations for d in o["allocator"]})
    peaks = {d: {kind: max(o["allocator"][d][f"peak_{kind}_bytes"] for o in observations)
                 for kind in ("allocated", "reserved")} for d in devices}
    median = statistics.median(rates)
    optional_medians = {}
    for key in ("first_any_backbone_output_seconds", "prompt_output_ready_seconds", "first_incremental_call_seconds"):
        values = [o.get(key) for o in observations]
        optional_medians[key] = statistics.median(values) if all(v is not None for v in values) else None
    return dict(samples=len(observations), sample_stable_rates=rates,
                median_stable_decode_positions_per_second=median,
                range_over_median=(max(rates) - min(rates)) / median,
                median_mean_step_ms=1000 * statistics.median(o["stable_decode"]["seconds"] / o["stable_decode"]["calls"] for o in observations),
                median_p95_step_ms=1000 * statistics.median(o["stable_decode"]["p95_call_seconds"] for o in observations),
                participating_devices=devices, per_card_positions_per_second=median / len(devices),
                peak_bytes_by_device=peaks,
                max_card_peak_bytes={kind: max(p[kind] for p in peaks.values()) for kind in ("allocated", "reserved")},
                final_state_memory=final_state_memory(observations, batch, diagnostics),
                **optional_medians)


def normal_result(raw, subject_name=None):
    subject = raw if subject_name is None else raw.get("subjects", {}).get(subject_name, {})
    observed = subject.get("observations", [])
    arguments = raw.get("arguments", {})
    reference = raw.get("matching_reference") if subject_name == "tide" else None
    return dict(state=raw.get("state"), subject_state=subject.get("state"), last_stage=raw.get("last_stage"),
        error=raw.get("error"), source=raw.get("source"), arguments=arguments, schedule=raw.get("schedule"),
        precision=raw.get("precision"), torch=raw.get("torch"), physical_devices=raw.get("physical_devices"),
        resource_identity=raw.get("resource_identity"), cpu_affinity=raw.get("cpu_affinity"),
        torch_threads=raw.get("torch_threads"), host_load_before=raw.get("host_load_before"), host_load_after=raw.get("host_load_after"),
        config=raw.get("config"), spec=subject.get("spec"), parameters=subject.get("parameters"),
        initial_parameters=subject.get("initial_parameters"), placement=subject.get("placement"),
        comparison=subject.get("comparison", raw.get("comparison")),
        input_sha256=subject.get("input_sha256"), input_shape=subject.get("input_shape"),
        shared_input_prefix=subject.get("shared_input_prefix"), diagnostics=subject.get("diagnostics"),
        activity_audit=None if reference is None else reference.get("audit"),
        activity_audit_state="recorded" if reference is not None else "not_recorded",
        matching_reference=reference,
        timing_protocol={k: raw.get(k) for k in ("model_lifetime", "request_boundary", "allocator_boundary", "cold_boundary", "timing_scope") if k in raw},
        sampling=dict(requested_samples=arguments.get("samples"), warmup=arguments.get("warmup"),
                      normal_samples=len(observed), cold_present="cold" in subject, cold_used_in_ranking=False),
        observations=observed, metrics=metrics(observed, arguments.get("batch"), subject.get("diagnostics")))


def within_budget(result):
    metric = result.get("metrics")
    return bool(metric and all(p["allocated"] <= 48 * 2**30 and p["reserved"] <= 56 * 2**30
                               for p in metric["peak_bytes_by_device"].values()))


def finite_diagnostic(result):
    diagnostic = result.get("diagnostics") or {}
    parameters = result.get("initial_parameters") or {}
    return parameters.get("finite") is True and all(diagnostic.get(k) is True for k in
        ("finite_outputs", "finite_private_state", "private_state_detached"))


def current_point(submitted, audited, input_reconstruction=None):
    name = submitted["name"]
    if Path(name).name != name:
        raise ValueError("submission name must be one path component")
    folder = ROOT / "artifacts" / name
    records, files, errors = {}, {}, []
    for filename in RAW_FILES:
        try:
            records[filename], identity = load(folder / filename, optional=True)
            if identity:
                files[filename] = identity
        except (OSError, ValueError) as exc:
            errors.append(filename + ": " + str(exc))
    status = records.get("status.json") or {}
    raw = records.get("inference.json")
    point = dict(name=name, origin="current_search", phase="confirm" if submitted["confirm"] else "screen",
        series=series_key(submitted["scale"], submitted["kind"], submitted["cards"]),
        scale=submitted["scale"], kind=submitted["kind"], cards=submitted["cards"], batch=submitted["batch"],
        submission=submitted, files=files, launch=records.get("launch.json"),
        outer_state=status.get("state"), outer_error=status.get("error"),
        terminal=bool(status.get("state") and status["state"] not in RUNNING) or submitted.get("state") == "launch_failed",
        result=None, validation_errors=errors, eligible_observed_candidate=False, eligible_confirmation=False)
    if raw:
        try:
            point["result"] = normal_result(raw, "tide" if submitted["kind"] == "tide" else None)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append("result summary: " + str(exc))
    stale = []
    if audited is None:
        stale.append("submission has no independent audit entry")
    else:
        if audited.get("submission") != submitted:
            stale.append("submission differs from independently audited snapshot")
        if set(audited.get("files", {})) != set(files):
            stale.append("raw-file inventory differs from independently audited snapshot")
        for filename, expected in audited.get("files", {}).items():
            if not same_file(expected, files.get(filename)):
                stale.append(filename + " changed since independent audit")
        errors.extend("independent audit: " + e for e in audited.get("audit_errors", []))
    point["independent_audit"] = dict(current=not stale, pending_reasons=stale,
        classification=None if audited is None else audited.get("classification"),
        eligible=None if audited is None else audited.get("eligible"),
        input_verification=None if audited is None else audited.get("input_verification"),
        source_trees=None if audited is None else audited.get("source_trees"),
        input_generator=None if audited is None else audited.get("input_generator"),
        budget_breaches=[] if audited is None else audited.get("budget_breaches", []))
    point["input_reconstruction"] = input_reconstruction
    point["classification"] = (audited["classification"] if not stale else
                               "pending_independent_audit" if point["terminal"] else "incomplete")
    # Source/raw mismatches after an already-qualified terminal audit are more
    # serious than ordinary progress of a running record; preserve that fact.
    if stale and audited and audited.get("classification") == "completed_within_budget":
        errors.append("previously completed audited point changed; re-audit required")
    result = point["result"]
    if result:
        point["within_sampled_budget"] = within_budget(result)
        point["finite_diagnostic"] = finite_diagnostic(result)
        reported_error = str(result.get("error") or "")
        point["reported_failure_kind"] = ("sampled_budget_MemoryError" if reported_error.startswith("MemoryError:")
            and "sampled budget" in reported_error else "other_memory_error" if "MemoryError" in reported_error
            else "other_execution_error" if reported_error else None)
        point["eligible_observed_candidate"] = bool(not stale and not errors and audited.get("eligible")
            and audited.get("classification") == "completed_within_budget" and point["outer_state"] == "passed"
            and result["state"] == result["subject_state"] == "completed"
            and point["within_sampled_budget"] and point["finite_diagnostic"])
        point["eligible_confirmation"] = bool(point["eligible_observed_candidate"] and submitted["confirm"]
            and result["sampling"]["requested_samples"] == result["sampling"]["normal_samples"] == 5
            and result["sampling"]["warmup"] == 2
            and (submitted["kind"] != "tide" or result["activity_audit_state"] == "recorded"))
        point["confirmation_activity_missing"] = bool(submitted["confirm"] and submitted["kind"] == "tide"
            and result["state"] == "completed" and result["activity_audit_state"] != "recorded")
    return point


def historical_points(prior, evidence_identity):
    entries = [(entry["result"], entry.get("files", {}), entry.get("outer_state")) for entry in prior["runs"]]
    entries += [(entry["result"], {"inference.json": entry["file"]}, "reported_prior_completed")
                for entry in prior.get("original_comparisons", [])]
    rows = []
    for recorded, expected_files, outer_state in entries:
        name = recorded["name"]
        errors, files, raw = [], {}, None
        for filename, expected in expected_files.items():
            try:
                loaded, identity = load(ROOT / expected["path"])
                files[filename] = identity
                if not same_file(expected, identity):
                    errors.append(filename + " differs from previous evidence fingerprint")
                if filename == "inference.json":
                    raw = loaded
            except (OSError, ValueError) as exc:
                errors.append(filename + ": " + str(exc))
        is_tide = recorded.get("comparison", {}).get("kind") == "tide"
        spec = recorded["spec"]
        kind = "tide" if is_tide else "tf" + str(spec["layers"])
        scale = "17b" if "17b" in name else "9b"
        cards = recorded["metrics"]["participating_devices"]
        point = dict(name=name, origin="historical_reference", phase="historical", kind=kind, scale=scale,
            cards=cards, batch=recorded["arguments"]["batch"], series=series_key(scale, kind, cards),
            evidence_report=evidence_identity, files=files, reported_files=expected_files,
            reported_outer_state=outer_state, validation_errors=errors,
            eligible_confirmation=False, eligible_observed_candidate=False)
        if raw:
            selector = "tide" if is_tide else "C" if name.startswith("original-") else None
            try:
                point["result"] = normal_result(raw, selector)
                result = point["result"]
                # Old source/sample identities are never rewritten to this run.
                for field in ("source", "parameters", "initial_parameters", "input_sha256", "observations"):
                    if result[field] != recorded[field]:
                        errors.append(field + " differs from previous evidence")
                point["eligible_historical_reference"] = bool(not errors and result["state"] == result["subject_state"] == "completed"
                    and recorded["state"] == "completed" and finite_diagnostic(result) and within_budget(result))
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                errors.append("historical result: " + str(exc))
        point.setdefault("eligible_historical_reference", False)
        rows.append(point)
    return rows


def ranked(points):
    ordered = sorted(points, key=lambda p: (-p["result"]["metrics"]["median_stable_decode_positions_per_second"], p["batch"], p["name"]))
    return [dict(rank=index + 1, name=p["name"], origin=p["origin"], phase=p["phase"], batch=p["batch"],
                 cards=p["cards"], source=p["result"]["source"], sampling=p["result"]["sampling"],
                 physical_devices=p["result"]["physical_devices"], metrics=p["result"]["metrics"])
            for index, p in enumerate(ordered)]


def prefix_evidence(point):
    """Retain the distinction between a recorded prefix and CPU reconstruction."""
    result = point["result"]
    rebuilt = point.get("input_reconstruction")
    if (rebuilt and rebuilt.get("state") == "passed"
            and point.get("independent_audit", {}).get("current")
            and rebuilt.get("full_sha256") == result["input_sha256"]):
        return dict(width=rebuilt["common_prefix_width"], sha256=rebuilt["common_prefix_sha256"],
                    basis="independent_cpu_reconstruction")
    recorded = result.get("shared_input_prefix")
    if recorded:
        return dict(width=recorded["width"], sha256=recorded["sha256"], basis="recorded_in_raw_run")
    return None


def comparison(left, right, kind):
    a, b = left["result"], right["result"]
    prefix_a, prefix_b = prefix_evidence(left), prefix_evidence(right)
    shared_prefix = (None if not prefix_a or not prefix_b else
                     prefix_a["width"] == prefix_b["width"] == 768 and prefix_a["sha256"] == prefix_b["sha256"])
    checks = dict(same_batch=left["batch"] == right["batch"], same_card_count=left["cards"] == right["cards"],
        same_schedule=a["schedule"] == b["schedule"], same_requested_precision=a["arguments"]["precision"] == b["arguments"]["precision"],
        same_seed=a["arguments"]["seed"] == b["arguments"]["seed"], same_torch=a["torch"] == b["torch"],
        same_physical_devices=a["physical_devices"] == b["physical_devices"],
        same_source_commit=a["source"]["source_commit"] == b["source"]["source_commit"],
        same_sample_count=a["sampling"]["normal_samples"] == b["sampling"]["normal_samples"],
        shared_768_prefix_equal=shared_prefix)
    workload_aligned = all(checks[k] for k in ("same_card_count", "same_schedule", "same_requested_precision", "same_seed"))
    if kind == "same_batch":
        workload_aligned &= checks["same_batch"]
    ratio = (a["metrics"]["median_stable_decode_positions_per_second"] /
             b["metrics"]["median_stable_decode_positions_per_second"] if workload_aligned else None)
    return dict(kind=kind, scale=left["scale"], cards=left["cards"], transformer_kind=right["kind"],
        tide=ranked([left])[0], transformer=ranked([right])[0], checks=checks,
        input_prefix_evidence=dict(tide=prefix_a, transformer=prefix_b),
        tide_over_transformer_throughput=ratio,
        interpretation=("same batch/request/precision/card count where checks agree; phase, source, physical devices and sample counts remain distinct; not semantic equivalence or isolated executor speedup"
                        if kind == "same_batch" else
                        "each series uses its own highest eligible confirmed batch under the same per-card budgets; not a same-batch speedup or global optimum"))


def comparison_tables(points, historical, series):
    # Prefer confirmation over screen over historical at a shared B. Selection
    # is independent of observed speed, avoiding cherry-picking the fastest run.
    priority = {"confirm": 0, "screen": 1, "historical": 2}
    eligible = [p for p in points if p["eligible_observed_candidate"]]
    eligible += [p for p in historical if p["eligible_historical_reference"]]
    chosen = {}
    for point in sorted(eligible, key=lambda p: (priority[p["phase"]], -p["result"]["sampling"]["normal_samples"], p["name"])):
        chosen.setdefault((point["series"], point["batch"]), point)
    same_batch = []
    for (_, batch), tide in sorted(chosen.items()):
        if tide["kind"] != "tide":
            continue
        for depth in (5, 32, 48):
            key = series_key(tide["scale"], "tf" + str(depth), tide["cards"])
            transformer = chosen.get((key, batch))
            if transformer:
                same_batch.append(comparison(tide, transformer, "same_batch"))
    byname = {p["name"]: p for p in points}
    byseries = {s["series"]: s for s in series}
    budget_best = []
    for tide_series in series:
        if tide_series["kind"] != "tide" or not tide_series["confirmed_winner"]:
            continue
        for depth in (5, 32, 48):
            key = series_key(tide_series["scale"], "tf" + str(depth), tide_series["cards"])
            transformer_series = byseries[key]
            if transformer_series["confirmed_winner"]:
                pair = comparison(byname[tide_series["confirmed_winner"]["name"]],
                                  byname[transformer_series["confirmed_winner"]["name"]], "independent_budget_best")
                pair["both_manually_finished"] = all(s["manual_decision_validation"]["state"] == "finished_with_eligible_selection"
                                                      for s in (tide_series, transformer_series))
                pair["both_manual_selections_match_highest_confirmed"] = all(
                    s["manual_decision_validation"].get("selection_is_highest_confirmed") is True
                    for s in (tide_series, transformer_series))
                budget_best.append(pair)
    return dict(same_batch_selection_rule="current confirmation, then current screen, then historical; more samples then name break same-phase ties, never higher throughput",
                same_batch=same_batch, independently_selected_budget_best=budget_best)


def decision_validation(decision, current, confirmed):
    if decision is None:
        return dict(state="not_recorded", selected_point=None, selection_is_highest_confirmed=None)
    selected = next((p for p in current if p["name"] == decision.get("winner")), None)
    valid = bool(selected and selected["eligible_confirmation"])
    finished = decision.get("state") == "finished"
    return dict(state="finished_with_eligible_selection" if finished and valid else
                      "finished_pending_selection_validation" if finished else "search_not_declared_finished",
                selected_point=decision.get("winner"), selected_is_eligible_confirmation=valid,
                selection_is_highest_confirmed=(bool(confirmed) and decision.get("winner") == confirmed[0]["name"]) if valid else None,
                rule="manual stopping/selection record never grants audit eligibility or changes measured rankings")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=BASE / "audit.json")
    parser.add_argument("--submissions", type=Path, default=BASE / "submissions.json")
    parser.add_argument("--preflight", type=Path, default=BASE / "preflight.json")
    parser.add_argument("--prior", type=Path, default=ROOT / "docs/evidence/transformer-depth-batch-20261118.json")
    parser.add_argument("--decisions", type=Path, default=BASE / "series-decisions.json")
    parser.add_argument("--output", type=Path, default=BASE / "summary.json")
    args = parser.parse_args(argv)
    audit, audit_identity = load(args.audit)
    submissions, submissions_identity = load(args.submissions)
    preflight, preflight_identity = load(args.preflight)
    prior, prior_identity = load(args.prior)
    decisions, decisions_identity = load(args.decisions, optional=True)
    decisions = decisions or {}
    audited = {r["name"]: r for r in audit["runs"]}
    reconstructed = {name: group for group in audit["inputs"].get("groups", []) for name in group["runs"]}
    points = [current_point(s, audited.get(s["name"]), reconstructed.get(s["name"])) for s in submissions]
    historical = historical_points(prior, prior_identity)
    errors = [dict(name=p["name"], origin=p["origin"], error=error)
              for p in points + historical for error in p["validation_errors"]]
    if len({s["name"] for s in submissions}) != len(submissions):
        errors.append(dict(name=None, origin="submissions", error="duplicate submitted run names"))
    plan_series = preflight["plan"]["series"]
    planned = {series_key(s["scale"], s["kind"], s["cards"]) for s in plan_series}
    for point in points:
        if point["series"] not in planned:
            errors.append(dict(name=point["name"], origin="current_search", error="series outside preflight plan"))
            point["eligible_confirmation"] = point["eligible_observed_candidate"] = False
    series = []
    for plan in plan_series:
        key = series_key(plan["scale"], plan["kind"], plan["cards"])
        current = [p for p in points if p["series"] == key]
        old = [p for p in historical if p["series"] == key]
        observed = ranked([p for p in current if p["eligible_observed_candidate"]])
        confirmed = ranked([p for p in current if p["eligible_confirmation"]])
        previous = ranked([p for p in old if p["eligible_historical_reference"]])
        series.append(dict(series=key, **plan, submitted_points=len(current),
            measurement_state="not_submitted" if not current else "no_current_qualified_measurement" if not observed else "measured",
            missing_current_qualified_measurements=not bool(observed), confirmation_missing=not bool(confirmed),
            current_points=[p["name"] for p in current], historical_points=[p["name"] for p in old],
            observed_current_ranking=observed, confirmed_current_ranking=confirmed,
            historical_reference_ranking=previous, confirmed_winner=confirmed[0] if confirmed else None,
            manual_decision=decisions.get(key), manual_decision_validation=decision_validation(decisions.get(key), current, confirmed),
            winner_scope="highest observed eligible current confirmation; not a global maximum or final search conclusion"))
    classes = Counter(p["classification"] for p in points)
    sample_classes, sample_phases = Counter(), Counter()
    for point in points:
        count = len((point.get("result") or {}).get("observations", []))
        sample_classes[point["classification"]] += count
        sample_phases[point["phase"]] += count
    output = dict(schema="tide-decode-capacity-summary-1", recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        summarizer=dict(path=relative(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), python=sys.version),
        inputs=dict(audit=audit_identity, submissions=submissions_identity, preflight=preflight_identity,
                    prior_evidence=prior_identity, manual_decisions=decisions_identity),
        scope="intermediate search/evidence snapshot; no new A/B/D maximum claim; not process-lifecycle audit or final conclusion",
        rules=dict(ranking="descending median of each point's normal-sample stable_decode positions/s; retain screen/confirmation identities and counts, never pool their observations",
            confirmation="only current confirm=true, audited-current, within-budget, finite, warmup2/samples5 points can be confirmed_winner; Tide confirmations also retain a full activity audit",
            historical="separate source and actual sample counts; historical points never become current confirmations implicitly",
            memory="each participating device's maximum normal-request allocated/reserved peak; budgets 48/56 GiB per card",
            terminal="status-file snapshot only; does not inspect MainPID, descendants or queue ownership"),
        plan=preflight["plan"], independent_audit_summary=audit["summary"], independent_input_checks=audit["inputs"],
        independent_audit_manifest_matches_current=same_file(audit.get("submissions"), submissions_identity),
        runs=points, historical_references=historical, series=series,
        comparisons=comparison_tables(points, historical, series),
        manual_decision_issues=[dict(series=key, error="decision is outside preflight series") for key in decisions if key not in planned],
        all_terminal=bool(points) and all(p["terminal"] for p in points),
        all_current_points_audited=bool(points) and all(p["independent_audit"]["current"] for p in points),
        all_series_have_confirmed_winner=bool(series) and all(s["confirmed_winner"] is not None for s in series),
        all_series_manually_finished_with_eligible_selection=bool(series) and all(
            s["manual_decision_validation"]["state"] == "finished_with_eligible_selection" for s in series),
        validation_errors=errors, validation_error_count=len(errors),
        counts=dict(series=len(series), current_runs=len(points), historical_references=len(historical),
            classifications=dict(classes), eligible_observed=sum(p["eligible_observed_candidate"] for p in points),
            eligible_confirmations=sum(p["eligible_confirmation"] for p in points),
            normal_samples=sum(sample_classes.values()), normal_samples_by_classification=dict(sample_classes),
            normal_samples_by_phase=dict(sample_phases),
            normal_samples_scope="all current normal-timing observations, including retained samples before budget refusal; qualification remains separate",
            historical_normal_samples=sum(len((p.get("result") or {}).get("observations", [])) for p in historical)))
    atomic(args.output, output)
    print(json.dumps(dict(output=str(args.output), all_terminal=output["all_terminal"],
                         all_current_points_audited=output["all_current_points_audited"],
                         validation_error_count=len(errors), **output["counts"])))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
