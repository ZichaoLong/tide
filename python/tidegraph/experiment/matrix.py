"""Explicit finite matrices, each case in its own bounded installed process."""
from copy import deepcopy
import os
from pathlib import Path
import re
import sys
from .config import resolve
from .identity import capture, verify
from .records import fingerprint, now, read_json, write_json
from .site import fields, load as load_site


def _merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_matrix(value, *, site=None, base_dir=None):
    base = Path(base_dir or Path.cwd())
    if isinstance(value, (str, os.PathLike)):
        path = (base/Path(value)).resolve()
        value, base = read_json(path), path.parent
    if value.get("schema") == "tide-experiment-matrix-resolved-v1":
        body = {k: v for k, v in value.items() if k != "matrix_id"}
        if fingerprint(body) != value.get("matrix_id"):
            raise ValueError("matrix_id: resolved matrix content changed")
        if site is not None:
            raise ValueError("site: cannot override a frozen resolved matrix")
        return deepcopy(value)
    fields(value, {"schema", "name", "description", "site", "cases"}, "matrix")
    if value.get("schema") != "tide-experiment-matrix-v1":
        raise ValueError("matrix.schema: expected tide-experiment-matrix-v1")
    if not isinstance(value.get("cases"), list) or not value["cases"]:
        raise ValueError("matrix.cases: supply a nonempty explicit list; no implicit Cartesian sweep")
    selected = load_site(site if site is not None else value.get("site"), base_dir=base)
    cases, names = [], set()
    for i, case in enumerate(value["cases"]):
        fields(case, {"name", "config", "overrides"}, f"matrix.cases[{i}]")
        name = case.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name) or name in names:
            raise ValueError(f"matrix.cases[{i}].name: use a unique safe name of 1–80 characters")
        names.add(name)
        data, config_base = case.get("config"), base
        if isinstance(data, str):
            path = (base/Path(data)).resolve()
            data, config_base = read_json(path), path.parent
        if not isinstance(data, dict) or not isinstance(case.get("overrides", {}), dict):
            raise ValueError(f"matrix.cases[{i}]: config must be a JSON object/path; overrides must be an object")
        requested = _merge(data, case.get("overrides", {}))
        cases.append(dict(name=name, config=resolve(requested, site=selected, base_dir=config_base)))
    body = dict(schema="tide-experiment-matrix-resolved-v1", name=value.get("name", "finite-matrix"),
                description=value.get("description", ""), site=selected, cases=cases,
                timeout_seconds=sum(row["config"]["measurement"]["timeout_seconds"]+30 for row in cases)+30,
                requested=deepcopy(value))
    return dict(body, matrix_id=fingerprint(body))


def run_matrix(value, output_dir):
    """No timing aggregation across contracts; failures remain individual records."""
    from .container_job import prepare, supervise
    matrix = resolve_matrix(value)
    out = Path(output_dir).resolve()
    identities = [capture(row["config"]["site"]["paths"]) for row in matrix["cases"]]
    from .task_identity import identity as task_identity, verify as verify_task
    task_identities = [task_identity(row["config"]["task"]) for row in matrix["cases"]]
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"matrix.json", matrix)
    result = dict(schema="tide-experiment-matrix-result-v1", state="running", matrix_id=matrix["matrix_id"],
                  started=now(), cases=[], qualification="each case keeps its own independent checks")
    write_json(out/"result.json", result)
    try:
        for case, identity, task in zip(matrix["cases"], identities, task_identities):
            result["current_case"] = case["name"]
            write_json(out/"result.json", result)
            config = case["config"]
            verify(identity, config["site"]["paths"])
            verify_task(config["task"], task)
            directory = out/case["name"]
            command = [sys.executable, "-m", "tidegraph", "run", str(directory/"resolved.json"),
                       "--output-dir", str(directory/"run")]
            prepare(directory, command=command, cwd=out, timeout=config["measurement"]["timeout_seconds"],
                    metadata=dict(config_id=config["config_id"], install_identity=identity))
            write_json(directory/"resolved.json", config)
            (directory/"resolved.json").chmod(0o444)
            code = supervise(directory)
            status = read_json(directory/"status.json")
            result["cases"].append(dict(name=case["name"], config_id=config["config_id"], exit_code=code,
                                         state=status["state"], result=str(directory/"run"/"result.json")))
            write_json(out/"result.json", result)
            if status["state"] == "cancelled":
                break
        states = [case["state"] for case in result["cases"]]
        result["state"] = "cancelled" if "cancelled" in states else "passed" if all(x == "passed" for x in states) else "failed"
    except BaseException as error:
        result.update(state="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        result.update(finished=now(), executed_cases=len(result["cases"]), requested_cases=len(matrix["cases"]))
        write_json(out/"result.json", result)
    return result
