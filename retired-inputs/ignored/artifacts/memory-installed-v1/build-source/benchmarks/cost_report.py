"""Render finite cost records without rerunning models or merging timing scopes."""
import argparse
from collections import Counter
import json
from pathlib import Path

from benchmarks.references import file_identity
from benchmarks.search_report import write_text_atomic


def _number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def _table(lines, columns, rows):
    lines += ["| "+" | ".join(columns)+" |", "|"+"|".join("---" for _ in columns)+"|"]
    lines.extend("| "+" | ".join(str(value).replace("|", "\\|") for value in row)+" |" for row in rows)
    lines.append("")


def _shape(tensor):
    return "×".join(map(str, tensor["shape"]))+" "+tensor["dtype"].removeprefix("torch.")+"@"+tensor["device"]


def render(records, limit=12):
    lines = ["# Complete-training cost observations", "",
             "Normal timing includes fresh prefill, required Timed flush, readout/loss, backward, AdamW, synchronization/scalar and release. "
             "Diagnostic passes restore the exact first measured training anchor. Construction, input transfer, warmup, snapshot/restore/hash and diagnostics are outside normal timing.", "",
             "Stage inclusive times nest; saved-storage peaks overlap; host and device trace events may overlap. "
             "None of those columns may be added to form normal wall-time shares. These observations do not qualify numerical equivalence or model quality.", ""]
    for identity, record in records:
        args = record["arguments"]
        lines += [f"## {Path(args['config']).stem} / {args['precision']} / B{args['batch']} L{args['length']}", "",
                  f"Source `{record['source']}`; harness `{record['harness_source']}`; state `{record['state']}`. "
                  f"Resource identity: {record['resource_identity']}.",
                  f"[Raw record](<{identity['path']}>) SHA256 `{identity['sha256']}`.", ""]
        if record.get("initial_audit_error"):
            lines += ["Initial matching audit failed: `"+record["initial_audit_error"]+"`.", ""]
        observations = record.get("observations", [])
        standards = {row["subject"]: row.get("normal", {}).get("median_seconds")
                     for row in observations if row["candidate"] == "standard"}
        rows = []
        for row in observations:
            normal = row.get("normal", {})
            seconds = normal.get("median_seconds")
            base = standards.get(row["subject"])
            peak = (normal.get("device_memory") or {}).get("peak_allocated_bytes")
            rows.append((row["subject"]+"/"+row["candidate"], normal.get("state", "unmeasured")+" / "+row["state"],
                         _number(seconds), _number(normal.get("input_positions_per_second"), 2),
                         _number(None if peak is None else peak/2**30, 3),
                         _number(None if seconds is None or not base else seconds/base, 3),
                         _number(normal.get("sample_range_over_median"), 3)))
        _table(lines, ("Subject/candidate", "Normal / all passes", "Normal median s", "Input positions/s",
                       "Normal peak GiB", "Time / standard", "Range / median"), rows)
        lines += ["Candidate time ratios are same-subject normal times at the same anchor; values below one are faster in this observation. "
                  "A failed diagnostic does not erase completed normal samples. No global baseline optimum is claimed.", ""]
        if record.get("joint_matches"):
            _table(lines, ("Class", "Block target", "Parameter target", "Blocks", "Width / FFN", "Actual parameters", "Parameter deviation"),
                   [(name, match["block_target_kind"], match["parameter_target_kind"], match["actual"]["blocks"],
                     str(match["constructor"]["d"])+" / "+str(match["constructor"]["ffn_width"]), match["actual"]["parameters"],
                     _number(match["relative_deviation"]["parameters"], 5)) for name, match in record["joint_matches"].items()])
        for row in observations:
            lines += ["### "+row["subject"]+" / "+row["candidate"], ""]
            if row.get("error"):
                lines += ["Retained failure: `"+row["error"]+"`.", ""]
            for kind, diagnostic in row.get("diagnostics", {}).items():
                if diagnostic["state"] != "completed_finite":
                    lines += [kind+" diagnostic: `"+diagnostic["state"]+"`; `"+diagnostic.get("error", "")+"`.", ""]
                    continue
                report = diagnostic.get("report", {})
                if kind == "stages":
                    lines += ["Synchronized intrusive stage costs. The largest exclusive rows include parent dispatch, "
                              "unwrapped work and observer overhead; they are not pure kernel or Python costs.", ""]
                    _table(lines, ("Nested path", "Calls", "Inclusive s", "Exclusive s"),
                           [(r["path"], r["calls"], _number(r["inclusive_seconds"]), _number(r["exclusive_seconds"]))
                            for r in sorted(report["rows"], key=lambda r: r["exclusive_seconds"], reverse=True)[:limit]])
                    phases = [r for r in report["rows"] if r["path"].count("/") == 1]
                    _table(lines, ("Whole intrusive phase", "Inclusive s"),
                           [(r["path"], _number(r["inclusive_seconds"])) for r in phases])
                elif kind == "saved":
                    total = report["total"]
                    lines += [f"Saved-tensor simultaneous union peak: {total['peak_live_unique_storage_bytes']/2**30:.4f} GiB; "
                              f"excluding parameter storages: {total['peak_live_additional_storage_bytes']/2**30:.4f} GiB. "
                              "Per-site peaks below overlap and must not be summed. These are saved storage capacities, not allocator peaks.", ""]
                    _table(lines, ("Save site", "Save events", "Payload event GiB", "Peak live GiB", "Peak nonparameter GiB"),
                           [(r["path"], r["pack_calls"], _number(r["saved_payload_event_bytes"]/2**30),
                             _number(r["peak_live_unique_storage_bytes"]/2**30), _number(r["peak_live_additional_storage_bytes"]/2**30))
                            for r in sorted(report["rows"], key=lambda r: r["peak_live_additional_storage_bytes"], reverse=True)[:limit]])
                elif kind == "runtime":
                    counts = Counter()
                    matrix_ops = {r["operator"] for r in report["matrix_groups"]}
                    for operator in report["operators"]:
                        if operator["operator"] not in matrix_ops:
                            counts[operator["operator"]] += operator["calls"]
                    lines += [f"Separate complete-training dispatch pass: {report['dispatched_calls']} calls. "
                              "High-count non-matrix operators are scheduling/materialization evidence, not measured time contributions.", ""]
                    _table(lines, ("Non-matrix operator", "Calls"), counts.most_common(limit))
                    _table(lines, ("Matrix/attention path", "Operator", "Calls", "Input shapes, dtype and device"),
                           [(r["path"], r["operator"], r["calls"], "; ".join(_shape(t) for t in r["inputs"][:4]))
                            for r in sorted(report["matrix_groups"], key=lambda r: r["calls"], reverse=True)[:limit]])
                    lines += ["Full operand strides, output dtypes and all materialization counters are in the raw record. "
                              "Logical materialized bytes are not bus traffic. Native attention/scan internals remain opaque to dispatch.", ""]
                elif kind == "work":
                    lines += ["Separate forward/flush semantic work at the shared training anchor:", ""]
                    if "work" in report:
                        _table(lines, ("Work", "Count", "Per actual input position"),
                               [(name, value, _number(report["work_per_input_position"].get(name), 3)) for name, value in report["work"].items()])
                        _table(lines, ("Packing", "Calls", "Selected rows", "Executed rows"),
                               [(name, r["calls"], r["selected_rows"], r["executed_rows"]) for name, r in report["query_packing"].items()])
                        total = sum(r["dense_multiply_adds"] for r in report["node_matrices"].values())
                        lines += [f"NodeLinear forward multiply-add pairs including padding/adapters: {total}; "
                                  "excludes attention, regional ordinary Linear and other work. This is not total FLOPs.", ""]
                    else:
                        names = ("attention_calls", "effective_attention_pairs", "attention_operand_rectangle_pairs",
                                 "masked_rectangle_pairs", "linear_forward_multiply_adds", "attention_rectangle_multiply_adds")
                        _table(lines, ("Transformer work", "Count"), [(name, report[name]) for name in names])
                        lines += [report["interpretation"]+".", ""]
                elif kind == "trace":
                    trace = diagnostic["trace"]
                    lines += [f"[Separate device/operator trace](<{trace['path']}>) SHA256 `{trace['sha256']}`. "
                              "Raw category names are preserved; host inclusive and concurrent device durations are not additive wall time.", ""]
                    for category, groups in diagnostic["summary"]["top_by_category"].items():
                        _table(lines, ("Trace category: "+category, "Calls", "Inclusive μs"),
                               [(r["name"], r["calls"], _number(r["inclusive_duration_us"], 1)) for r in groups[:limit]])
            if row.get("normal_endpoint_work", {}).get("effective_touched"):
                lines += ["Post-normal endpoint effective touched mean (separate audit): "+
                          _number(row["normal_endpoint_work"]["effective_touched"]["mean"], 2)+
                          ". Matching remained frozen at initialization; routing work can change during updates.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("limit must be positive")
    if Path(args.output).exists():
        parser.error("output already exists; choose a new report identity")
    records = []
    for path in args.records:
        record = json.loads(Path(path).read_text())
        if record.get("schema") != "tide-training-costs-v1":
            parser.error("not a training-cost record: "+path)
        for row in record.get("observations", []):
            trace = row.get("diagnostics", {}).get("trace", {}).get("trace")
            if trace:
                actual = file_identity(trace["path"])
                if any(actual[key] != trace[key] for key in ("sha256", "bytes")):
                    parser.error("raw trace identity changed: "+trace["path"])
        records.append((file_identity(path), record))
    write_text_atomic(args.output, render(records, args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
