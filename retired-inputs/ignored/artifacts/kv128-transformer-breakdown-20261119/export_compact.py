#!/usr/bin/env python3
"""Freeze a bounded, reviewable view of the four completed breakdown records.

This is an artifact-only stdlib reader. It performs no lifecycle/device checks,
launches no work, and neither changes nor requalifies the observed tensors.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.job import atomic

EXPECTED_SOURCE = "7196fcabd7ab3b401c1ebdb6107442450a99996c"
NAMES = ("infer-kv128", "infer-tf32", "train-kv128", "train-tf32")
MANIFEST = {}


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def fingerprint(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=relative(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read(path):
    path = Path(path)
    data = path.read_bytes()
    identity = dict(path=relative(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    MANIFEST[identity["path"]] = identity
    return json.loads(data), identity


def pick(value, *keys):
    return {key: value[key] for key in keys if key in value}


def without(value, *keys):
    return {key: item for key, item in value.items() if key not in keys}


def bounded_rows(rows, count=20):
    return dict(total_groups=len(rows), retained_groups=min(count, len(rows)),
                selection="largest recorded seconds, descending; remaining groups stay in the fingerprinted cost review",
                rows=sorted(rows, key=lambda row: row.get("seconds", 0), reverse=True)[:count])


def normal(value):
    result = without(value, "samples", "warmup")
    result["samples"] = [without(row, "calls") for row in value.get("samples", [])]
    result["warmup"] = [without(row, "calls") for row in value.get("warmup", [])]
    result["sample_scope"] = "all scalar/timing/allocator/state-summary fields retained; per-call observations are in raw/summary"
    return result


def stages(value):
    result = without(value, "canonical_tree", "selected_subtree_views")
    result["selected_subtree_views"] = [without(row, "paths") for row in value.get("selected_subtree_views", [])]
    result["retention"] = "all broad/detail/mechanism cuts and selected inclusive views; canonical call paths omitted; alternative cuts are not additive"
    return result


def logical_bytes(rows):
    totals = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[(row["phase"], row["category"])][key] += value
    return dict(by_phase_category=[dict(phase=p, category=c, **values) for (p, c), values in sorted(totals.items())],
                definition="sum of the cost review's repeated logical output bytes; not bus traffic, unique allocation, live memory, or peak memory")


def operators(value):
    result = without(value, "by_phase_module_category", "top_operator_shape_groups", "matrix_and_attention_groups",
                     "packed_query_work", "repeated_logical_output_bytes")
    result["top_operator_shape_groups"] = bounded_rows(value.get("top_operator_shape_groups", []))
    result["top_matrix_and_attention_shape_groups"] = bounded_rows(value.get("matrix_and_attention_groups", []))
    packed = value.get("packed_query_work", {})
    result["packed_query_work"] = without(packed, "rows")
    result["packed_query_work"]["rows"] = [without(row, "publish_shape_groups", "down_shape_groups")
                                               for row in packed.get("rows", [])]
    result["repeated_logical_output_bytes"] = logical_bytes(value.get("repeated_logical_output_bytes", []))
    result["retention"] = "phase/category/module sums and top operators retained; bounded top20 shape groups; packed query totals retain padding and preparation counts, omitting full shape-group lists"
    return result


def qualification(value):
    result = pick(value, "state", "observation_issues")
    review = value["review"]
    result["review"] = without(review, "diagnostics")
    result["review"]["diagnostics"] = {}
    for kind, diagnostic in review.get("diagnostics", {}).items():
        trimmed = without(diagnostic, "complete_output_identities", "observation")
        if "complete_output_identities" in diagnostic:
            encoded = json.dumps(diagnostic["complete_output_identities"], sort_keys=True, separators=(",", ":")).encode()
            trimmed["complete_output_identities_reference"] = dict(
                json_location=f"records.<case>.review.diagnostics.{kind}.complete_output_identities",
                count=len(diagnostic["complete_output_identities"]),
                canonical_json_sha256=hashlib.sha256(encoded).hexdigest())
        if "observation" in diagnostic:
            trimmed["observation"] = without(diagnostic["observation"], "matrix_examples")
        result["review"]["diagnostics"][kind] = trimmed
    return result


def original(value):
    result = without(value, "normal")
    if "normal" in value:
        result["normal"] = normal(value["normal"])
    return result


def check_partition(name, value):
    if not name.startswith("train"):
        return None
    partition = value["complete_partition"]
    mean = value["computed"]["complete_seconds"]["mean"]
    total = value["computed"]["complete_seconds_total"]
    mean_error = sum(partition["mean_seconds"].values()) - mean
    total_error = sum(partition["total_seconds"].values()) - total
    assert math.isclose(mean_error, 0., abs_tol=1e-12), (name, "mean partition", mean_error)
    assert math.isclose(total_error, 0., abs_tol=1e-11), (name, "total partition", total_error)
    return dict(mean_residual_seconds=mean_error, total_residual_seconds=total_error,
                interpretation="arithmetic reconciliation only, not an additional timing observation")


def main():
    summary, summary_file = read(HERE / "summary.json")
    cost, cost_file = read(HERE / "cost-review.json")
    qual, qual_file = read(HERE / "qualification-review.json")
    assert summary["state"] == "complete_snapshot" and not summary["summary_errors"]
    assert cost["state"] == qual["state"] == "completed_review"
    assert summary["expected_source"] == qual["expected_source"] == EXPECTED_SOURCE
    assert not qual["pending"]
    jobs = {job["name"]: job for job in summary["jobs"]}
    assert tuple(jobs) == NAMES
    cases, validation = {}, {}
    for name in NAMES:
        job, cv, qv = jobs[name], cost["cases"][name], qual["records"][name]
        raw_id = fingerprint(ROOT / job["file"]["path"])
        assert raw_id["sha256"] == job["file"]["sha256"] == cv["file"]["sha256"] == qv["raw_sha256"], name
        assert raw_id["bytes"] == job["file"]["bytes"] == cv["file"]["bytes"] == qv["raw_bytes"], name
        MANIFEST[raw_id["path"]] = raw_id
        assert job["state"] == cv["state"] == qv["raw_state"] == "completed_finite", name
        assert job["source"]["source_commit"] == EXPECTED_SOURCE, name
        assert job["normal"]["timing_complete"] and job["normal"]["observed_samples"] == 5, name
        assert not qv["observation_issues"], (name, qv["observation_issues"])
        case = without(job, "normal", "diagnostics")
        case["normal"] = normal(job["normal"])
        case["stages"] = stages(cv["stages"])
        case["operators"] = operators(cv["operators"])
        case["qualification"] = qualification(qv)
        case["cost_numerical_scope"] = cv["numerical_scope"]
        case["diagnostic_comparisons"] = {}
        for kind, diagnostic in job["diagnostics"].items():
            case["diagnostic_comparisons"][kind] = without(diagnostic, "report", "calls", "final_state")
        cases[name] = case
        validation[name] = dict(raw_identity_matches_summary_cost_and_qualification=True,
                               source_matches=True, normal_samples=5,
                               complete_partition=check_partition(name, job["normal"]))
    for pair in summary["paired_comparisons"].values():
        assert pair["settings_match"] and pair["shared_input_prefix_matches"] and pair["complete_job_pair"]
    for name in ("export_compact.py", "summarize.py", "review_costs.py", "review_qualification.py"):
        identity = fingerprint(HERE / name)
        MANIFEST[identity["path"]] = identity
    result = dict(schema="tide-kv128-transformer-breakdown-compact-v1",
        generated_at=datetime.now(timezone.utc).isoformat(), state="completed_compact_review",
        expected_source=EXPECTED_SOURCE, generator=MANIFEST[relative(__file__)],
        cases=cases, paired_comparisons=summary["paired_comparisons"],
        original_points=[original(row) for row in summary["original_points"]],
        cost_report_tables=cost["report_tables"],
        qualification_authority=pick(qual, "original_fp32_thresholds", "authority", "limits"),
        units=summary["units"],
        boundaries=summary["boundaries"] + cost["boundaries"] + [
            "This compact export validates record identity and arithmetic only. The parent release audit owns actual service/cgroup/descendant terminal verification.",
            "All five normal scalar samples are retained. Diagnostic call trees, canonical stage trees, complete packed shape lists and most matrix shape groups remain in fingerprinted inputs.",
            "Stage broad/detail/mechanism cuts are alternative nonoverlapping partitions; selected inclusive subtrees overlap parents. Operator shape top20 is a bounded view, not all execution work.",
            "The summaries retain finite nonexact training comparisons and original thresholds. No tolerance was widened and no difference is attributed to rounding by this exporter.",
            "Inference private floating history and all SSD convolution-clock values are not fully witnessed by metadata/output identity replay; consult each qualification state_scope.",
            "No global best-batch, 9B training, A-D matching, full-language-model TTFT, or model-quality conclusion is added by these four points."],
        validation=validation, input_files=list(MANIFEST.values()),
        source_reviews=dict(summary=summary_file, costs=cost_file, qualification=qual_file),
        omitted_views_reference="full views remain at source_reviews and the four raw input_files; paths and exact SHA256 identify their immutable bytes")
    # Check complete input stability before atomically replacing the export.
    for identity in MANIFEST.values():
        assert fingerprint(ROOT / identity["path"]) == identity, ("input changed during export", identity["path"])
    size = len((json.dumps(result, indent=2) + "\n").encode())
    assert size < 1024 * 1024, ("compact evidence exceeds 1 MiB", size)
    output = HERE / "compact-evidence.json"
    atomic(output, result)
    actual = fingerprint(output)
    assert actual["bytes"] == size
    print(json.dumps(dict(output=actual, state=result["state"], input_states=dict(summary=summary["state"], costs=cost["state"], qualification=qual["state"]), validation=validation)))


if __name__ == "__main__":
    main()
