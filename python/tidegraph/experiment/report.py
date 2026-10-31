"""Dependency-free query, CSV and self-contained HTML for experiment records."""
import csv
import gzip
import io
import json
from pathlib import Path
import statistics
from .records import read_json, digest, fingerprint, replace_bytes, replace_text
from .evidence import details


COLUMNS = ("run", "platform", "graph", "topology", "parameters", "logical_batch", "positions", "span",
           "mode", "observation", "dtype", "scoring_dtype", "client", "placement", "devices", "communication",
           "return_mode", "packing", "return_packing", "schedule", "threads", "workers", "execution_status", "numerical", "discrete",
           "complete", "checkpoint", "exact_resume", "exploratory", "qualification", "kind", "throughput_sample", "throughput_qualified",
           "median_seconds", "samples_per_second", "config_id", "model_id", "workload_id", "task_id", "execution_id",
           "result_sha256", "timing_contract", "work", "memory", "error", "admission_refusals", "details")

V1_COLUMNS = COLUMNS[:-1]


def _state(value):
    if value is True:
        return "passed"
    if value is False:
        return "failed"
    if isinstance(value, dict):
        return str(value.get("state", value.get("status", "unchecked")))
    return "unchecked" if value is None else str(value)


def _time_rows(metrics):
    result = []
    for row in metrics if isinstance(metrics, list) else []:
        if not isinstance(row, dict) or row.get("phase") in {"warmup", "construction", "resume", "validation"}:
            continue
        for key, scale in (("elapsed_seconds", 1), ("seconds", 1), ("elapsed_ms", .001), ("milliseconds", .001)):
            if type(row.get(key)) in {int, float} and row[key] >= 0:
                result.append(row[key]*scale)
                break
    return result


def summarize(record, path):
    """Execution, independent correctness and resume stay separate columns."""
    path = Path(path)
    resolved = record.get("resolved_config", {})
    if not isinstance(resolved, dict):
        resolved = {}
    if not resolved and (path.parent/"resolved.json").is_file():
        resolved = read_json(path.parent/"resolved.json")
    plan = resolved.get("plan", {})
    execution = resolved.get("execution", {}).get("effective", {})
    measurement = resolved.get("measurement", {})
    validation = record.get("validation", {})
    validation = validation if isinstance(validation, dict) else {}
    checkpoint = record.get("checkpoint", {})
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    states = {k: _state(validation.get(k)) for k in ("discrete", "complete")}
    states["numerical"] = _state(validation.get("numerical", validation.get("numeric")))
    state = record.get("execution_status", record.get("state", "unchecked"))
    explored = record.get("exploratory_policy", validation.get("policy", validation.get("mode", measurement.get("validation"))) == "explore")
    qualified = (state in {"completed", "passed"} and states["numerical"] == "passed"
                 and states["discrete"] == "passed" and states["complete"] in {"passed", "checked"})
    if measurement.get("checkpoint_check", False):
        qualified = qualified and _state(checkpoint.get("state")) == "passed" and checkpoint.get("exact_resume") is True
    if qualified:
        qualification = "strict-checks-passed"
    elif any(value in {"failed", "mismatch", "numerical-mismatch", "discrete-mismatch"} for value in states.values()):
        qualification = "strict-failure"
    else:
        qualification = "not-established"
    observed = measurement.get("observation", "none")
    times = _time_rows(record.get("metrics", [])) if observed == "none" else []
    seconds = statistics.median(times) if times else None
    effective = record.get("work", {})
    sample_count = effective.get("samples", plan.get("samples"))
    recipe = (plan.get("recipe") or {}).get("requested", {})
    row = dict(run=str(path.parent), config_id=record.get("config_id", resolved.get("config_id")),
        **{key: record.get(key, resolved.get(key)) for key in ("model_id", "workload_id", "task_id", "execution_id")},
        result_sha256=digest(path) if path.is_file() else None,
        platform=execution.get("device", "unknown").split(":")[0],
        graph=resolved.get("graph_view", {}).get("family", plan.get("graph", {}).get("kind")),
        topology=recipe.get("topology", "custom"),
        parameters=resolved.get("inventory", plan.get("inventory", {})).get("totals", {}).get("trainable"),
        logical_batch=plan.get("samples"), positions=plan.get("positions"), span=plan.get("span"),
        mode=measurement.get("mode"), observation=observed, dtype=execution.get("dtype"), scoring_dtype=execution.get("scoring_dtype"),
        client=execution.get("client"), placement=execution.get("placement"),
        devices=",".join(execution.get("peer_devices") or [execution.get("device", "unknown")]),
        communication=execution.get("communication", "ordinary"), return_mode=execution.get("return_mode", "complete"),
        packing=execution.get("packing"), return_packing=execution.get("return_packing", "scalar"),
        schedule=execution.get("schedule"), threads=execution.get("intraop_threads"), workers=execution.get("workers"),
        execution_status=state, **states, checkpoint=_state(checkpoint.get("state")),
        exact_resume=checkpoint.get("exact_resume", "unchecked"), exploratory=explored,
        qualification=qualification, kind=record.get("kind", "benchmark"),
        throughput_sample=record.get("throughput_sample", observed == "none"),
        throughput_qualified=record.get("throughput_qualified", qualified and observed == "none"), median_seconds=seconds,
        samples_per_second=sample_count/seconds if seconds and sample_count else None,
        timing_contract=record.get("timing_contract", "unrecorded"),
        work=record.get("work", [{k: v for k, v in row.items() if k not in {
            "seconds", "elapsed_seconds", "elapsed_ms", "milliseconds", "loss", "checkpoint_seconds"}}
            for row in record.get("metrics", []) if isinstance(row, dict)]),
        memory=record.get("memory", "unmeasured"), error=record.get("error"),
        admission_refusals=record.get("admission", {}).get("refused", []), details=details(record, resolved))
    return row


def load(paths):
    """Read only explicit JSON/result.json files, never checkpoints or raw logs."""
    records, seen = [], set()
    for value in paths:
        root = Path(value).resolve()
        if not root.exists():
            raise FileNotFoundError(f"results path does not exist: {root}")
        files = sorted(root.rglob("result.json")) if root.is_dir() else [root]
        for path in files:
            if path in seen:
                continue
            seen.add(path)
            record = json.loads(gzip.decompress(path.read_bytes())) if path.suffix == ".gz" else read_json(path)
            if not isinstance(record, dict):
                raise ValueError(f"{path}: expected a result object")
            if record.get("schema") in {"tide-experiment-evidence-v1", "tide-experiment-evidence-v2"}:
                rows = record.get("results")
                columns = V1_COLUMNS if record["schema"].endswith("v1") else COLUMNS
                if (not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != set(columns) for row in rows)
                        or fingerprint(rows) != record.get("results_sha256")):
                    raise ValueError(f"{path}: invalid evidence rows or content digest")
                records.extend(dict(row, details={"scope": "unrecorded in v1 archive"})
                               if columns == V1_COLUMNS else row for row in rows)
                continue
            # Build/job status records do not become experiment qualifications.
            if not ("resolved_config" in record or "config_id" in record or (path.parent/"resolved.json").is_file()):
                continue
            records.append(summarize(record, path))
    return records


def select(rows, **filters):
    aliases = {"status": "execution_status", "params_min": "parameters", "params_max": "parameters", "batch": "logical_batch"}
    unknown = set(filters)-set(COLUMNS)-set(aliases)
    if unknown:
        raise ValueError("unknown result filter: "+sorted(unknown)[0])
    result = []
    for row in rows:
        match = True
        for key, expected in filters.items():
            if expected is None:
                continue
            actual = row.get(aliases.get(key, key))
            if key in {"params_min", "params_max"}:
                match = match and actual is not None and (actual >= expected if key.endswith("min") else actual <= expected)
            else:
                match = match and str(actual) == str(expected)
        if match:
            result.append(row)
    return result


def _cell(value):
    if value is None:
        return "unmeasured"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def save(path, content):
    """A .gz suffix compresses the same report, with deterministic gzip metadata."""
    if Path(path).suffix == ".gz":
        replace_bytes(path, gzip.compress(content.encode(), mtime=0))
    else:
        replace_text(path, content)


def render(rows, *, format="table"):
    if format == "archive":
        return json.dumps(dict(schema="tide-experiment-evidence-v2", results=rows,
            results_sha256=fingerprint(rows),
            scope="Portable result projection; source hashes identify original records, not retained checkpoint bytes or revalidation"),
            indent=2, ensure_ascii=False, allow_nan=False)+"\n"
    if format == "json":
        return json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False)+"\n"
    if format == "csv":
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([_cell(row.get(k)) for k in COLUMNS])
        return stream.getvalue()
    if format == "html":
        from .html_report import render_html
        return render_html(rows)
    if format != "table":
        raise ValueError("report format must be table, csv, html, json or archive")
    columns = ("platform", "graph", "topology", "parameters", "logical_batch", "dtype", "client", "placement",
               "execution_status", "qualification", "checkpoint", "median_seconds")
    cells = [[_cell(row.get(k)) for k in columns] for row in rows]
    widths = [max(len(name), *(len(row[i]) for row in cells)) if cells else len(name) for i, name in enumerate(columns)]
    line = lambda values: " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))
    return line(columns)+"\n"+"-+-".join("-"*n for n in widths)+"\n"+"\n".join(map(line, cells))+"\n"
