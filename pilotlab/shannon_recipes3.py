"""Resolve new Shannon recipes and bind a fresh seed cohort in a frozen campaign."""
from __future__ import annotations

from pathlib import Path

from .source_migration3 import read_bound


DEFAULT_POLICY = "weak-log-v1"


def new_config(config):
    if config.get("allocation") == "shannon":
        return {"shannon_support_policy": DEFAULT_POLICY, **config}
    return config


def cohort_subjects(subjects, rollout):
    """Only the named Shannon model configurations change in this plan."""
    ids = set(rollout["model_ids"])
    available = {s["id"] for s in subjects if s["model"].get("config", {}).get("allocation") == "shannon"}
    if (rollout["policy"] != DEFAULT_POLICY or not ids or not ids <= available
            or len(ids) != len(rollout["model_ids"])):
        raise ValueError("invalid fresh Shannon cohort")
    result = []
    for subject in subjects:
        if subject["id"] in ids:
            model = subject["model"]
            config = {**model["config"], "shannon_support_policy": DEFAULT_POLICY}
            subject = {**subject, "model": {**model, "config": config}}
        result.append(subject)
    return result


def verify_cohort(rollout, baseline_identity):
    from .prepare_campaign3 import source_inventory

    identity = read_bound(rollout["worker_identity"])
    if any(identity[k] != baseline_identity[k] for k in ("data_manifest_sha256", "runtime_packages", "python_version")):
        raise ValueError("fresh Shannon cohort changed its data or runtime")
    project, base = source_inventory({**rollout, "source_identity": identity})
    if "weak_mass.py" not in base or "worker3.py" not in project:
        raise ValueError("fresh Shannon cohort lacks its weak-log worker")
    cases = set()
    for pair in read_bound(rollout["qualification"]).values():
        failure, = (row for row in pair["pairs"] if row["case"] == "failure")
        if (pair["source_identity"] != rollout["worker_identity"]
                or failure["arms"][DEFAULT_POLICY]["backward_divisor"] != 1):
            raise ValueError("fresh Shannon worker differs from the qualified failure replay")
        cases.add((pair["device"], pair["old_result"]["sha256"]))
    if len(cases) != 4 or {d for d, _ in cases} != {"cpu", "npu:0"} or len({r for _, r in cases}) != 2:
        raise ValueError("incomplete paired Shannon qualification")
    return identity


def cohort_runtime(rollout):
    from .prepare_resume3 import reference

    return {"source": rollout["source"], "foundation": rollout["foundation"],
            "adapter": reference(Path(rollout["source"]) / "pilotlab/resume_adapter3.py")}
