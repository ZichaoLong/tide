"""Study evidence checks and small query records, independent of Torch."""
import json
from pathlib import Path
from source_identity import digest


def complete_checks(state, mode, checks, resume, *, checkpoint_required=False):
    complete = state == "passed" and checks.get("post_measurement_complete_records") is True
    if mode == "training":
        complete = complete and all(checks.get(k) is True for k in (
            "parameter_values_checked", "final_parameter_gradients_checked", "optimizer_values_checked"))
    if mode == "training" or checkpoint_required:
        complete = complete and (resume or {}).get("state") == "passed" and (resume or {}).get("exact") is True
    return complete


def timing_sample(result):
    """Instrumentation costs remain queryable but never become throughput."""
    return (result.get("throughput_sample") is not False and
            result.get("kind") not in {"profile", "work-observation", "observation"} and
            result.get("measurement_kind") not in {"profile", "work-observation", "observation"})


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
    if not timing_sample(result):
        raise ValueError("profile/work observation is not a throughput measurement")
    if result.get("state") != "passed" or not result.get("post_measurement_complete_records"):
        raise ValueError("measurement did not establish complete independent equivalence")
    if result.get("workload_id") != record["plan"]["sha256"]:
        raise ValueError("measured workload identity differs")
    if result.get("requested_dtype") != record["execution"]["requested"]["dtype"]:
        raise ValueError("measured precision differs")
    if result.get("return_packing", "scalar") != record["execution"]["requested"].get("return_packing", "scalar"):
        raise ValueError("measured complete-return packing differs")
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
    checks = {k: result.get(k, False) for k in ("post_measurement_complete_records", "parameter_values_checked",
              "final_parameter_gradients_checked", "optimizer_values_checked")}
    checkpoint_required = measurement.get("checkpoint_check", False)
    complete = complete_checks(r["state"], measurement["mode"], checks, result.get("exact_resume"),
                               checkpoint_required=checkpoint_required)
    sample = timing_sample(result)
    return dict(run=str(Path(path).resolve()), state=r["state"], source=r["source"],
        case=r["case_id"], execution_name=r["execution_name"], model_id=r["model_id"], workload_id=r["workload_id"],
        execution_id=execution["execution_id"], measurement_id=measurement["measurement_id"],
        requested=execution["requested"], effective=result.get("effective", result.get("effective_placement")),
        recipe=r["recipe"], parameters=r["parameters"], graph_counts=r["graph_counts"],
        mode=measurement["mode"], phase=result.get("phase"), error=result.get("error", r.get("error")),
        independent_complete=result.get("post_measurement_complete_records", False),
        exact_resume=result.get("exact_resume"), metrics=metrics, measured_seconds=total or None,
        kind="performance" if sample else "observation", measurement_kind=result.get("kind", "benchmark"),
        throughput_sample=sample, checkpoint_check=checkpoint_required,
        checks=checks, correctness_qualified=complete, throughput_qualified=complete and sample,
        configuration_facts=r.get("configuration_facts"),
        work=work, memory=result.get("memory"), construction_seconds=result.get("construction_seconds"),
        warmup_seconds=result.get("warmup_seconds"), fixture_sha256=result.get("fixture_sha256"))


def query_records(path, _ancestors=frozenset()):
    """Read live run records or reviewed repository evidence without old logs.

    Evidence hashes identify the reviewed inputs; this does not authenticate or
    re-run them. Preserve failure and check flags instead of trusting timings.
    """
    path = Path(path).resolve()
    if path in _ancestors:
        raise ValueError("cyclic evidence index")
    r = json.loads(path.read_text())
    if r.get("schema") == "tide-core5-evidence-index-v1":
        records = []
        for name in r["reports"]:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("evidence index requires local relative report paths")
            records.extend(query_records(path.parent/relative, _ancestors | {path}))
        return records
    if r.get("schema") == "tide-core5-correctness-evidence-v1":
        records = []
        for row in r["cases"]:
            facts, execution = r["models"][row["workload_id"]], row["execution"]
            checks = row.get("checks", {})
            resume = checks.get("exact_resume")
            complete = complete_checks(row["state"], row["mode"], checks, resume,
                                       checkpoint_required=row.get("checkpoint_check", False))
            records.append(dict(row, source=r["source"], kind="correctness", run=row["test"],
                evidence_file=str(path), execution_name=execution["requested"]["client"],
                requested=execution["requested"], effective=execution["effective"],
                execution_id=execution["execution_id"], configuration_facts=facts,
                model_id=facts["model_id"], workload_id=facts["workload_id"], recipe=facts["recipe"],
                parameters=facts["parameters"], graph_counts=facts["graph_counts"],
                independent_complete=checks.get("post_measurement_complete_records", False),
                correctness_qualified=complete, throughput_qualified=False,
                exact_resume=resume, metrics=[], measured_seconds=None, phase="correctness-check"))
        return records
    if r.get("schema") == "tide-core5-run-v1":
        return [summarize(path)]
    if r.get("schema") != "tide-core5-finite-performance-evidence-v1":
        raise ValueError("unknown study evidence schema")
    records = []
    for row in r["runs"]:
        item = dict(row)
        sample = timing_sample(item)
        model = r["models"][item["case"]]
        item.update(parameters=model["parameters"], graph_counts=model["counts"], recipe=model["recipe"],
                    evidence_file=str(path), reviewed_model=model, kind="performance" if sample else "observation")
        checks, resume = item.get("checks", {}), item.get("exact_resume") or {}
        checkpoint_required = item.get("checkpoint_check", item.get("measurement", {}).get("checkpoint_check", False))
        complete = complete_checks(item["state"], item["mode"], checks, resume,
                                   checkpoint_required=checkpoint_required)
        item["correctness_qualified"] = complete
        item["throughput_qualified"] = complete and sample
        records.append(item)
    return records
