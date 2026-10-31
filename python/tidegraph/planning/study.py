"""Finite study catalog, explicit case expansion and public consumer commands."""
from .execution import ExecutionSpec, MeasurementSpec
from .model import fingerprint
from .recipes import prepare
from .legacy_topology import integer


def expand(study):
    """Expand only the listed groups; no backend/dtype/config Cartesian sweep."""
    if study.get("schema") != "tide-core5-study-v1":
        raise ValueError("unknown study schema")
    if set(study)-{"schema", "models", "window", "measurement", "executions", "policy"}:
        raise ValueError("unknown study field")
    MeasurementSpec(**study["measurement"]).record("validation", "validation")
    result = {}
    for group in study["models"]:
        if set(group)-{"id", "recipe", "families", "window", "question"}:
            raise ValueError("unknown model group field")
        families = group.get("families", [group["recipe"].get("family", "kv")])
        for family in families:
            name = group["id"]+"-"+family
            if name in result:
                raise ValueError("duplicate study case")
            recipe = dict(group["recipe"], family=family)
            result[name] = dict(recipe=recipe, window=dict(study["window"], **group.get("window", {})),
                                question=group["question"])
    for name, spec in study["executions"].items():
        ExecutionSpec(**spec)
    if not result or not study["executions"]:
        raise ValueError("study needs explicit models and executions")
    return result


def case(study, case_id, execution_id):
    cases = expand(study)
    if case_id not in cases or execution_id not in study["executions"]:
        raise ValueError("unknown case/execution ID")
    item = cases[case_id]
    plan, inv = prepare(item["recipe"], **item["window"])
    execution = ExecutionSpec(**study["executions"][execution_id]).record()
    measurement = MeasurementSpec(**study["measurement"]).record(plan["sha256"], execution["execution_id"])
    return dict(study_id=fingerprint(study), case_id=case_id, execution_name=execution_id,
                question=item["question"], plan=plan, inventory=inv, execution=execution, measurement=measurement)


def command(record, *, python, root, output, workload, native_library=None,
            resident_library=None, standalone_binary=None, fixture=None,
            max_materialize_bytes=2*1024**3, resident_limits=None, checkpoint_max_bytes=None):
    """One configuration -> one process, with no omitted requested policies."""
    e, m = record["execution"]["requested"], record["measurement"]
    if e.get("finite_check_packing", "scalar") != "scalar":
        raise ValueError("finite_check_packing is an installed tidegraph run policy; the legacy study runner does not apply it")
    resident = e["placement"] == "resident"
    if checkpoint_max_bytes is not None:
        integer(checkpoint_max_bytes, "checkpoint max bytes")
        if e["client"] != "libtorch" or not m["checkpoint_check"]:
            raise ValueError("native checkpoint byte budget requires LibTorch checkpoint check")
    if e["client"] == "libtorch":
        if not standalone_binary or not fixture:
            raise ValueError("LibTorch needs an explicit binary and exported static fixture")
        argv = [standalone_binary, "--fixture", fixture, "--implementation", "resident" if resident else "native",
                "--checkpoint-check", str(m["checkpoint_check"]).lower()]
        for peer in e["peer_devices"]:
            argv += ["--peer-device", peer]
        if checkpoint_max_bytes is not None:
            argv += ["--checkpoint-max-bytes", str(checkpoint_max_bytes)]
    else:
        implementation = "resident" if resident else "python" if e["client"] == "pytorch" else "native"
        if implementation != "python" and not native_library:
            raise ValueError("native client needs its matching native library")
        if resident and not resident_library:
            raise ValueError("resident client needs its matching resident library")
        argv = [python, root/"scripts/benchmark_weighted.py", "--workload", workload,
                "--implementation", implementation, "--max-materialize-bytes", str(max_materialize_bytes)]
        if native_library:
            argv += ["--native-library", native_library]
        if resident_library:
            argv += ["--resident-library", resident_library]
        if m["checkpoint_check"]:
            argv += ["--checkpoint-check"]
        if e["peer_devices"]:
            argv += ["--peer-devices", *e["peer_devices"]]
    if resident_limits:
        if not resident:
            raise ValueError("resident limits require resident execution")
        argv += ["--resident-limits", resident_limits]
    argv += ["--device", e["device"], "--dtype", e["dtype"], "--scoring-dtype", e["scoring_dtype"],
             "--schedule", e["schedule"], "--placement", "native" if resident else e["placement"],
             "--packing", e["packing"], "--workers", str(e["workers"]),
             "--return-packing", e.get("return_packing", "scalar"),
             "--intraop-threads", str(e["intraop_threads"]), "--interop-threads", str(e["interop_threads"]),
             "--mode", m["mode"], "--steps", str(m["steps"]), "--output-dir", output]
    return list(map(str, argv))
