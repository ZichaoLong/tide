"""Study evidence checks and small query records, independent of Torch."""
import json
from pathlib import Path
from source_identity import digest


def standalone_identity(binary, root):
    """A binary must have the exact benchmark sources and successful build record."""
    path = binary.parent.parent/"result.json"
    record = json.loads(path.read_text())
    if (record.get("schema") != "tide-weighted-benchmark-build-v1" or record.get("state") != "passed"
            or record.get("binary_sha256") != digest(binary)):
        raise ValueError("standalone binary/build identity mismatch")
    for name, expected in record["sources"].items():
        if digest(root/name) != expected:
            raise ValueError("standalone benchmark source mismatch: "+name)
    from build_identity import source_hash
    if record["core"]["cpp_source_sha256"] != source_hash(root):
        raise ValueError("standalone core source mismatch")
    if record.get("resident"):
        from build_device_control import component_hash
        if record["resident"]["component_sha256"] != component_hash(root):
            raise ValueError("standalone resident source mismatch")
    return dict(manifest_sha256=digest(path), binary_sha256=digest(binary),
                build_source=record["source"], build_dirty=record["dirty"])


def accept_result(record, result):
    if result.get("state") != "passed" or not result.get("post_measurement_complete_records"):
        raise ValueError("measurement did not establish complete independent equivalence")
    if result.get("workload_id") != record["plan"]["sha256"]:
        raise ValueError("measured workload identity differs")
    if result.get("requested_dtype") != record["execution"]["requested"]["dtype"]:
        raise ValueError("measured precision differs")
    if record["measurement"]["mode"] == "training":
        for key in ("parameter_values_checked", "final_parameter_gradients_checked", "optimizer_values_checked"):
            if result.get(key) is not True:
                raise ValueError("incomplete training verification: "+key)
    if record["measurement"]["checkpoint_check"]:
        resume = result.get("exact_resume", {})
        if resume.get("state") != "passed" or resume.get("exact") is not True:
            raise ValueError("missing exact new-owner checkpoint verification")


def summarize(path):
    """Never promote a failed/live run because a metric or child result exists."""
    r = json.loads(Path(path).read_text())
    if r.get("schema") != "tide-core5-run-v1":
        raise ValueError("unknown run schema")
    result_path = Path(path).parent/"measurement/result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    execution, measurement = r["execution"], r["measurement"]
    metrics = result.get("metrics", [])
    total = sum(x["seconds"] for x in metrics)
    work = {key: sum(x.get(key, 0) for x in metrics)
            for key in ("candidate_updates", "full_calls", "messages", "outputs")}
    return dict(run=str(Path(path).resolve()), state=r["state"], source=r["source"],
        case=r["case_id"], execution_name=r["execution_name"], model_id=r["model_id"], workload_id=r["workload_id"],
        execution_id=execution["execution_id"], measurement_id=measurement["measurement_id"],
        requested=execution["requested"], effective=result.get("effective", result.get("effective_placement")),
        recipe=r["recipe"], parameters=r["parameters"], graph_counts=r["graph_counts"],
        mode=measurement["mode"], phase=result.get("phase"), error=result.get("error", r.get("error")),
        independent_complete=result.get("post_measurement_complete_records", False),
        exact_resume=result.get("exact_resume"), metrics=metrics, measured_seconds=total or None,
        throughput_qualified=r["state"] == "passed" and bool(result.get("post_measurement_complete_records")),
        work=work, memory=result.get("memory"), construction_seconds=result.get("construction_seconds"),
        warmup_seconds=result.get("warmup_seconds"), fixture_sha256=result.get("fixture_sha256"))


def query_records(path):
    """Read live run records or reviewed repository evidence without old logs.

    Evidence hashes identify the reviewed inputs; this does not authenticate or
    re-run them. Preserve failure and check flags instead of trusting timings.
    """
    r = json.loads(Path(path).read_text())
    if r.get("schema") == "tide-core5-run-v1":
        return [summarize(path)]
    if r.get("schema") != "tide-core5-finite-performance-evidence-v1":
        raise ValueError("unknown study evidence schema")
    records = []
    for row in r["runs"]:
        item = dict(row)
        model = r["models"][item["case"]]
        item.update(parameters=model["parameters"], graph_counts=model["counts"], recipe=model["recipe"],
                    evidence_file=str(Path(path).resolve()))
        checks, resume = item.get("checks", {}), item.get("exact_resume") or {}
        complete = checks.get("post_measurement_complete_records") is True
        if item["mode"] == "training":
            complete = complete and all(checks.get(k) is True for k in (
                "parameter_values_checked", "final_parameter_gradients_checked", "optimizer_values_checked"))
            complete = complete and resume.get("state") == "passed" and resume.get("exact") is True
        item["throughput_qualified"] = item["state"] == "passed" and complete
        records.append(item)
    return records
