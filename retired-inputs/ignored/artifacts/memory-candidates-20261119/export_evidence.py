"""Export reviewed observations; does not launch or relabel experiments."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.job import atomic


def identity(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read(name):
    path = HERE/name
    return json.loads(path.read_text())


def pick(value, keys):
    return {k: value[k] for k in keys if k in value}


def compact(value, key=""):
    if key == "node_matrix_dispatch" and isinstance(value, list):
        groups = {}
        for row in value:
            group = tuple(row[k] for k in ("path", "base_operator", "in_features", "out_features"))
            if group not in groups:
                groups[group] = dict(zip(("path", "base_operator", "in_features", "out_features"), group))
                groups[group].update(calls=0, base_weight_rows=0, unique_base_weight_rows=0,
                                     repeated_base_weight_rows=0, node_rows_times_rows_per_node=[])
            out = groups[group]
            for field in ("calls", "base_weight_rows", "unique_base_weight_rows", "repeated_base_weight_rows"):
                out[field] += row[field]
            out["node_rows_times_rows_per_node"].append([row["node_rows"], row["rows_per_node"], row["calls"]])
        return dict(original_shape_rows=len(value), aggregation="sum counters by path/operator/in/out; shape triples retain nodes, rows_per_node, calls", groups=list(groups.values()))
    if isinstance(value, dict):
        return {k: compact(v, k) for k, v in value.items()
                if k not in {"by_sample_time", "descendant_identities", "sample_record", "storage_records"}
                and not (k == "calls" and isinstance(v, list))}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def main():
    summary = read("summary.json")
    numeric = read("numerics-review.json")
    release = read("release-audit.json")
    if summary["summary_errors"] or release["state"] != "passed":
        raise RuntimeError("release audit or summary is not ready")
    for group in ("training_b8", "training_b16", "inference", "joint_comparisons"):
        if any(state in {"missing", "running", "planned", "failed"} for state in summary["observed_states"][group].values()):
            raise RuntimeError(f"unfinished/failed main group: {group}")
    numerical = pick(numeric, ["schema", "scope", "comparisons_scope", "counting", "decision", "restore_summary",
        "bf16_boundary", "unresolved_cause", "actual_width_primitive", "five_class_coverage", "remaining_qualification_limits",
        "shared_9b_fp32_region_batch_support"])
    numerical["jobs"] = []
    for job in numeric["jobs"]:
        row = pick(job, ["job", "file", "file_sha256", "source", "state", "lifecycle_state", "physical_devices",
            "precision", "thresholds", "harness_sha256", "aggregate_counts", "discrete_interpretation"])
        row["candidates"] = [pick(case, ["candidate", "state", "config", "comparison_summary", "decision", "bounded_cpu_fp64"])
                             for case in job["rows"]]
        numerical["jobs"].append(row)
    reviews = {}
    for name in ("constructors.json", "implementation-review.json", "installed-review.json", "training-review.json",
                 "cost-review.json", "comparison-review.json", "budget-review.json", "decode-cost-development-review.json", "inference-review.json",
                 "release-supervisor-review.json"):
        path = HERE/name
        if path.exists():
            reviews[name] = dict(file=identity(path), record=compact(json.loads(path.read_text())))
    result = dict(schema="tide-memory-candidates-evidence-v1", generated=datetime.now(timezone.utc).isoformat(),
        scope="five explicit memory/ownership families; finite CPU/single-NPU qualification and performance; no task-quality certification",
        mathematics="4f5b9f20a4030d68ee66ae2d26e34659a5e37234", production_tree="5dba5b15c0c71ce729ff82c054e9468916acd7e5",
        generator=identity(Path(__file__)), summary_file=identity(HERE/"summary.json"), numerical_file=identity(HERE/"numerics-review.json"),
        numerical=numerical, observations=compact(summary), reviews=reviews,
        release_audit=dict(file=identity(HERE/"release-audit.json"), record=release),
        reduction_policy="Per-call timing detail and per-sample/time touch lists remain in fingerprinted raw records; normal sample summaries retained. Dispatch counters grouped with exact shape triples. Original failure records are not rewritten.")
    target = ROOT/"docs/evidence/memory-candidates-20261119.json"
    atomic(target, result)
    print(json.dumps(dict(output=str(target), bytes=target.stat().st_size, numerical_jobs=len(numerical["jobs"]), reviews=list(reviews))))


if __name__ == "__main__":
    main()
