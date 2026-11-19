"""Render existing bounded-search records; never rerun or instantiate models."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from benchmarks.references import file_identity


def write_text_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name+".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_metric_files(record):
    seen, identities = {}, [record["frozen_config"]] if "frozen_config" in record else []
    identities += [row["result_file"] for row in record.get("audits", []) if row.get("result_file")]
    if record.get("calibration_record"):
        identities.append(record["calibration_record"])
    for line in record["lines"]:
        for row in line["observations"]:
            identities += [row[key] for key in ("result_file", "reference_record", "calibration_record") if row.get(key)]
    for identity in identities:
        if identity["path"] in seen:
            actual = seen[identity["path"]]
        else:
            actual = file_identity(identity["path"])
            seen[identity["path"]] = actual
        if any(actual[key] != identity[key] for key in ("sha256", "bytes")):
            raise ValueError("raw result identity changed: "+identity["path"])
    return len(seen)


def render(records):
    lines = ["# Bounded throughput search", "",
             "These are finite observed candidates, not proven maximum capacity or mathematical/quality equivalence.",
             "Times include complete selected-mode work; memory and intrusive audits retain their declared boundaries.",
             "The common-shape table and independent fixed-model capacity table answer different questions.", ""]
    number = lambda value, digits=3: "—" if value is None else f"{value:.{digits}f}"
    for identity, record in records:
        args = record["arguments"]
        lines += [f"## {Path(args['config']).stem} / {args['precision']} / {args['mode']} / {args['table']}", "",
                  f"Source `{record['source']}`; search state `{record['state']}`; children `{json.dumps(record['child_states'], sort_keys=True)}`.",
                  f"[Raw search](<{identity['path']}>) SHA256 `{identity['sha256']}`.",
                  f"Device budget {args['device_budget_gib']} GiB; sampled child RSS budget {args['rss_gib']} GiB; axis `{args['axis']}` values `{record['axis_values']}`.", ""]
        if args["table"] == "common":
            lines += ["| B/L | Tide seconds | Input positions/s | Timed peak GiB | A time ratio | B | C | D | Stable subjects / admitted |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---|" ]
            for point in record["common_shape_table"]:
                tide = next((r for r in point["subjects"] if r["subject"] == "tide" and r["admitted"]), {})
                peak = tide.get("timed_peak_device_bytes")
                ratios = {r["joint_class"]: r["tide_over_transformer_time"] for r in point["comparisons"]}
                stable = sum(r["stable"] for r in point["subjects"])
                admitted = sum(r["admitted"] for r in point["subjects"])
                cells = [f"{point['batch']}/{point['length']}", number(tide.get("median_seconds")),
                         number(tide.get("input_positions_per_second"), 2), number(None if peak is None else peak/2**30),
                         *[number(ratios.get(name)) for name in "ABCD"], f"{stable}/{admitted}"]
                lines.append("| "+" | ".join(cells)+" |")
            lines += ["", "A–D values are Tide/Transformer elapsed-time ratios at the same B/L; greater than one means Tide takes longer.", ""]
        else:
            calibration = record.get("calibration_record")
            if calibration:
                lines += [f"Frozen calibration SHA256 `{calibration['sha256']}`. L remains {args['length']}.", ""]
            lines += ["| Subject | Best stable B | Seconds | Input positions/s | Timed peak GiB | Initial Tide active minus calibration | Stop |",
                      "|---|---:|---:|---:|---:|---:|---|" ]
            for item in record["capacity_table"]:
                best = item["best"] or {}
                peak = best.get("timed_peak_device_bytes")
                cells = [item["subject"], str(best.get("batch", "—")), number(best.get("median_seconds")),
                         number(best.get("input_positions_per_second"), 2), number(None if peak is None else peak/2**30),
                         number(best.get("activity", {}).get("initial_minus_calibration"), 2), item["stop_reason"]]
                lines.append("| "+" | ".join(cells)+" |")
            lines += ["", "Different best B values are independent load optima under the declared budget, not same-workload speedups.",
                      "A missing active difference means Tide was not audited at that B; it does not mean zero activity change.", ""]
        failed = [child for child in record["children"] if child["state"] != "completed"]
        if failed:
            lines += ["Retained failed/limited children:", ""]
            for child in failed:
                lines.append(f"- `{child['identity']}`: `{child['state']}`; [log](<{child['log']}>).")
            lines.append("")
        lines += ["Stop reasons: "+"; ".join(str(line["stop_reason"]) for line in record["lines"])+".", ""]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new report identity")
    records = []
    for path in args.inputs:
        record = json.loads(path.read_text())
        if record.get("schema") != "tide-capacity-search-v1" or record.get("state") in {"running", "planned"}:
            parser.error("only terminal search records can be rendered")
        verify_metric_files(record)
        records.append((file_identity(path), record))
    write_text_atomic(args.output, render(records))
    print(json.dumps(dict(output=str(args.output), search_records=len(records))))


if __name__ == "__main__":
    main()
