#!/usr/bin/env python3
"""Read frozen inference records and atomically refresh a bounded review.

No Torch import, model/test/device execution, job submission, or raw mutation.
Running/missing records remain pending. Re-run after the parent confirms all
four jobs are terminal. The reader hashes precisely the JSON bytes it parses.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.job import atomic

PRODUCTION_TREE = "89f2707e71d989340c12de310797a4c7f4c99f64"
EXPECTED_TOOL = "46317a8dd73d0611ff0f7b18f4057e97c5baa630"
COST_PATH = ROOT / "artifacts/kv128-transformer-breakdown-20261119/review_costs.py"
QUAL_PATH = ROOT / "artifacts/kv128-transformer-breakdown-20261119/review_qualification.py"
COST = runpy.run_path(str(COST_PATH), run_name="read_only_cost_helpers")
QUAL = runpy.run_path(str(QUAL_PATH), run_name="read_only_qualification_helpers")
MATRIX_OPS = {"aten.mm.default", "aten.bmm.default", "aten.matmul.default",
              "aten.addmm.default", "aten.baddbmm.default"}


def fingerprint(path, payload=None):
    payload = path.read_bytes() if payload is None else payload
    return dict(path=str(path.relative_to(ROOT)), bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest())


def read(path):
    payload = path.read_bytes()
    return json.loads(payload), fingerprint(path, payload)


def require(condition, message, issues):
    if not condition:
        issues.append(message)


def aggregate(rows, key, seconds="seconds"):
    groups = defaultdict(lambda: dict(seconds=0., calls=0, source_rows=0))
    for row in rows:
        value = groups[key(row)]
        value["seconds"] += row[seconds]
        value["calls"] += row["calls"]
        value["source_rows"] += 1
    return groups


def phase(path):
    return COST["phase"](path)


def is_ffn(parts):
    return any(x in {"ffn", "ffn_norm", "ffn.rows.apply", "ffn.rows.bucket",
                     "ffn.grouped.plan", "ffn.grouped.projection", "ffn.grouped.matmul"}
               for x in parts)


def mechanism(path, subject):
    """A disjoint cut; priority keeps metadata and FFN out of parent scopes."""
    parts = path.split("/")
    leaf = parts[-1]
    if "observation.metadata" in parts:
        return "observer.metadata"
    if leaf in {"allocation.sorted", "allocation.bisect", "allocation.save_face"}:
        return leaf
    if "allocation.validate_solve" in parts:
        return "allocation.validate_solve.remainder"
    if "allocation" in parts:
        return "allocation.control.remainder"
    if is_ffn(parts):
        return "FFN_including_normalization_and_row_preparation"
    return COST["mechanism_module"](path, subject)


def category(row):
    parts, op = row["path"].split("/"), row["operator"]
    if "observation.metadata" in parts:
        return "observer_metadata"
    owned = any(x.startswith(("node_linear.", "node_vector.")) or x == "node_linears.fused"
                for x in parts)
    first = row["inputs"][0] if row["inputs"] else None
    bank = (owned and op == "aten.index.Tensor" and first is not None
            and first["dtype"].startswith("torch.float")
            and (len(first["shape"]) == 3 or len(first["shape"]) == 2
                 and any(x.startswith("node_vector.") for x in parts)))
    if bank:
        return "parameter_bank_gather_in_projection_scope"
    # A floating activation index_select inside a projection is not a bank read.
    return COST["operator_class"]({**row, "path": ""})


def source_check(run, commit, raw, issues):
    frozen = run / "source"
    if not frozen.exists():
        return dict(state="pending_frozen_source")
    entries = subprocess.check_output(["git", "ls-tree", "-r", "-z", commit]).split(b"\0")
    count, mismatches = 0, []
    for entry in entries:
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        mode, kind, expected = meta.split()
        if kind != b"blob":
            continue
        file = frozen / os.fsdecode(path)
        if not file.exists():
            mismatches.append(dict(path=os.fsdecode(path), reason="missing"))
            continue
        payload = os.readlink(file).encode() if mode == b"120000" else file.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest().encode()
        if actual != expected:
            mismatches.append(dict(path=os.fsdecode(path), reason="different git blob"))
        count += 1
    tree = subprocess.check_output(["git", "rev-parse", commit + ":src/tide"], text=True).strip()
    require(tree == PRODUCTION_TREE, "production tree differs from locked implementation", issues)
    require(not mismatches, "frozen source differs from recorded commit", issues)
    content = {}
    for name, directory in (("tide", frozen / "src/tide"), ("harness", frozen / "benchmarks")):
        digest = hashlib.sha256()
        for file in sorted(directory.rglob("*.py")):
            digest.update(str(file.relative_to(directory)).encode())
            digest.update(file.read_bytes())
        content[name + "_content_sha256"] = digest.hexdigest()
    if raw:
        require(all(raw["source"].get(k) == v for k, v in content.items()),
                "recorded source content SHA differs from frozen bytes", issues)
    return dict(state="verified" if not mismatches else "mismatch", commit=commit,
                production_tree=tree, files=count, mismatches=mismatches, **content)


def allocation_summary(facts):
    sorted_rows = [x for x in facts if x["kind"] == "quad_sorted"]
    bisect = [x for x in facts if x["kind"] == "bisect"]
    return dict(sorted_calls=len(sorted_rows), sorted_rows=sum(x["rows"] for x in sorted_rows),
                sorted_invalid_rows=sum(x["invalid_rows"] for x in sorted_rows),
                bisect_calls=len(bisect), bisect_requested_rows=sum(x["requested_rows"] for x in bisect),
                bisect_executed_rows=sum(x["executed_rows"] for x in bisect),
                candidate_widths=sorted({x["candidates"] for x in bisect}),
                bisection_iterations=sorted({x["bisection_iterations"] for x in bisect}),
                correction_iterations=sorted({x["correction_iterations"] for x in bisect}))


def allocation_phases(diagnostic, kind, issues):
    report = diagnostic["report"]
    facts, rows = report.get("allocation_work", []), report["rows"]
    counts = Counter()
    if kind == "stages":
        for row in rows:
            if row["path"].split("/")[-1] in {"allocation.sorted", "allocation.bisect"}:
                counts[phase(row["path"])] += row["calls"]
    else:
        # In this frozen observer each sorted_facts/bisect_facts callback reads
        # exactly one scalar. Facts append in capture order; no payload replay.
        for row in rows:
            if "observation.metadata" in row["path"].split("/") and "_local_scalar_dense" in row["operator"]:
                counts[phase(row["path"])] += row["calls"]
    require(sum(counts.values()) == len(facts), kind + " allocation fact partition differs", issues)
    offset, by_phase = 0, {}
    for label in ("prefill", "decode"):
        current = facts[offset:offset + counts[label]]
        offset += counts[label]
        by_phase[label] = allocation_summary(current)
    return dict(by_phase=by_phase, all_calls=allocation_summary(facts),
                partition_method="Chronological metadata facts split by phase event counts; stage sorted/bisect invocations or one metadata scalar read per callback in the pinned operator observer.",
                limitation="Sorted invalid rows include inactive rows. Only bisect requested_rows counts fallback requests; executed_rows includes the full solver rectangle. Metadata reads are observer cost.")


def stage_summary(diagnostic, subject, decode, issues):
    if diagnostic.get("state") != "completed_finite":
        return dict(state="pending", raw_state=diagnostic.get("state"), calls_captured=len(diagnostic.get("calls", [])))
    parsed = COST["stage_review"](diagnostic, subject)
    checks = parsed["checks"]
    require(checks["maximum_tree_reconciliation_absolute_seconds"] < 1e-9
            and not checks["negative_exclusive_rows"] and not checks["nonfinite_rows"]
            and not checks["observer_boundary_failures"], "stage tree reconciliation failed", issues)
    rows = diagnostic["report"]["rows"]
    groups = aggregate(rows, lambda r: (phase(r["path"]), mechanism(r["path"], subject)), "exclusive_seconds")
    cut = [dict(phase=p, mechanism=m, exclusive_seconds=v["seconds"],
                scope_invocations=v["calls"], source_rows=v["source_rows"])
           for (p, m), v in sorted(groups.items())]
    calls = Counter(x["phase"] for x in diagnostic["calls"])
    require(calls["decode"] == decode, "stage decode denominator differs from requested full schedule", issues)
    for p in ("prefill", "decode"):
        require(math.isclose(sum(x["exclusive_seconds"] for x in cut if x["phase"] == p),
                             parsed["explicit_scope_seconds"][p], abs_tol=1e-9), "stage phase partition differs", issues)
    return dict(state="completed", phase_calls=dict(calls), exclusive_cut=cut,
                scope_seconds=parsed["explicit_scope_seconds"], tree_checks=checks,
                canonical_tree=parsed["canonical_tree"], allocation=allocation_phases(diagnostic, "stages", issues),
                report_boundary=diagnostic["report"]["boundary"],
                pass_elapsed_including_excluded_checks=diagnostic.get("pass_elapsed_seconds_including_excluded_work"),
                interpretation="Prefill is total captured forward/readout time. Decode average divides all16 captured decode calls, including the first. Release has a separate root; metadata remains an explicit observer category. This is not a normal-wall share.")


def operator_summary(diagnostic, subject, decode, issues):
    if diagnostic.get("state") != "completed_finite":
        return dict(state="pending", raw_state=diagnostic.get("state"), calls_captured=len(diagnostic.get("calls", [])))
    report = diagnostic["report"]
    rows = report["rows"]
    require(not report.get("failed_calls") and not report.get("failures"), "operator failure recorded", issues)
    require(sum(x["calls"] for x in rows) == report["dispatched_calls"], "operator count mismatch", issues)
    require(math.isclose(sum(x["seconds"] for x in rows), report["operator_seconds"], rel_tol=1e-12), "operator sum mismatch", issues)
    groups = aggregate(rows, lambda r: (phase(r["path"]), mechanism(r["path"], subject), category(r)))
    cut = [dict(phase=p, mechanism=m, category=c, **v) for (p, m, c), v in sorted(groups.items())]
    op_groups = aggregate(rows, lambda r: (phase(r["path"]), r["operator"],
                          "observer_metadata" if "observation.metadata" in r["path"] else "model"))
    matrices, matrix_totals, banks, movements = [], defaultdict(lambda: dict(calls=0, seconds=0., dense_multiply_add_pairs=0)), [], []
    packing = defaultdict(lambda: dict(selected_rows=0, publish_calls=0, executed_ffn_rows=0, ffn_down_calls=0,
                                        prepare_projection_rows=0, prepare_projection_calls=0))
    devices, matrix_output_types, scalars = Counter(), Counter(), Counter()
    for row in rows:
        p, parts, op = phase(row["path"]), row["path"].split("/"), row["operator"]
        for item in row["inputs"] + row["outputs"]:
            devices[item["device"], item["dtype"]] += row["calls"]
        if "_local_scalar_dense" in op:
            scalars[p, "observer_metadata" if "observation.metadata" in parts else "model"] += row["calls"]
        matrix = COST["matrix_work"](row) if op in MATRIX_OPS else None
        if matrix:
            entry = dict(path=row["path"], phase=p, operator=op, calls=row["calls"], seconds=row["seconds"],
                         inputs=row["inputs"], outputs=row["outputs"], **matrix)
            matrices.append(entry)
            target = matrix_totals[p, mechanism(row["path"], subject), op]
            target["calls"] += row["calls"]; target["seconds"] += row["seconds"]
            target["dense_multiply_add_pairs"] += matrix["all_calls_dense_multiply_adds"]
        if op in MATRIX_OPS or "attention" in op or "convolution" in op:
            for item in row["outputs"]:
                matrix_output_types[item["device"], item["dtype"]] += row["calls"]
        if category(row) == "parameter_bank_gather_in_projection_scope" and is_ffn(parts):
            banks.append(row)
        if any(t in op for t in ("index", "gather", "scatter", "cat.", "clone", "copy")):
            movements.append(row)
        memory = next((s.split(".")[1] for s in parts if s.startswith("memory.")), None)
        if memory is None or p not in {"prefill", "decode"}:
            continue
        target = packing[p, memory]
        if parts[-1] == "queries.publish" and op == "aten.index_put.default":
            require(len(row["inputs"]) == 5, "publish operand schema changed", issues)
            target["selected_rows"] += row["inputs"][-1]["shape"][0] * row["calls"]
            target["publish_calls"] += row["calls"]
        if f"node_linear.memories.{memory}.ffn.down" in parts and op in MATRIX_OPS:
            target["executed_ffn_rows"] += math.prod(row["outputs"][0]["shape"][:-1]) * row["calls"]
            target["ffn_down_calls"] += row["calls"]
        projection = "kv" if memory == "K" else "input"
        if parts[-1] == f"node_linear.memories.{memory}.{projection}" and op in MATRIX_OPS:
            target["prepare_projection_rows"] += math.prod(row["outputs"][0]["shape"][:-1]) * row["calls"]
            target["prepare_projection_calls"] += row["calls"]
    for value in packing.values():
        value["padding_rows"] = value["executed_ffn_rows"] - value["selected_rows"]
    def movement_summary(source_rows):
        result = defaultdict(lambda: dict(calls=0, seconds=0., logical_output_bytes=0))
        for row in source_rows:
            target = result[phase(row["path"]), mechanism(row["path"], subject), row["operator"]]
            target["calls"] += row["calls"]; target["seconds"] += row["seconds"]
            size = COST["logical_bytes"](row["outputs"])
            require(size is not None, "unknown dtype in logical-byte accounting", issues)
            if size is not None:
                target["logical_output_bytes"] += size * row["calls"]
        return [dict(phase=p, mechanism=m, operator=op, **v) for (p, m, op), v in sorted(result.items())]
    return dict(state="completed", dispatched_calls=report["dispatched_calls"], operator_seconds=report["operator_seconds"],
                phase_mechanism_category_cut=cut,
                by_phase_operator=[dict(phase=p, operator=op, scope=s, **v) for (p, op, s), v in sorted(op_groups.items())],
                matrix_totals=[dict(phase=p, mechanism=m, operator=op, **v) for (p, m, op), v in sorted(matrix_totals.items())],
                actual_matrix_shapes=matrices, ffn_weight_bank_gathers=movement_summary(banks),
                movement_events=movement_summary(movements),
                packing_from_operands=[dict(phase=p, memory_kind=k, **v) for (p, k), v in sorted(packing.items())],
                allocation=allocation_phases(diagnostic, "operators", issues),
                scalar_reads=[dict(phase=p, scope=s, calls=n) for (p, s), n in sorted(scalars.items())],
                observed_dtype_device_presence=[dict(device=d, dtype=t, calls=n) for (d, t), n in sorted(devices.items())],
                matrix_output_dtype_device_presence=[dict(device=d, dtype=t, calls=n) for (d, t), n in sorted(matrix_output_types.items())],
                pass_elapsed_including_excluded_checks=diagnostic.get("pass_elapsed_seconds_including_excluded_work"),
                boundary="Independent operator pass. Dense M*K*N counts exclude fused attention/opaque kernels; logical output-byte events are not traffic, bandwidth or peak memory. Preparation rows and selected/FFN rows come from actual operands of this pass only.")


def review(device, subject, run):
    name = f"{device}-{subject}"
    result = dict(name=name, state="pending", path=str(run.relative_to(ROOT)), issues=[])
    original = ROOT / f"artifacts/ffn-infer-{device}-stages-{subject}-v1"
    if run != original and (original / "status.json").exists():
        previous, identity = read(original / "status.json")
        result["previous_attempts"] = [dict(path=str(original.relative_to(ROOT)),
             state=previous.get("state"), exit_code=previous.get("exit_code"), identity=identity)]
        log = original / "stderr.log"
        if log.exists():
            payload = log.read_bytes()
            result["previous_attempts"][0]["stderr_identity"] = fingerprint(log, payload)
            result["previous_attempts"][0]["last_error_line"] = next(
                (s for s in reversed(payload.decode(errors="replace").splitlines()) if s.strip()), None)
    for filename, key in (("status.json", "supervisor"), ("queue.json", "queue"), ("dependency.json", "dependency")):
        if (run / filename).exists():
            data, identity = read(run / filename)
            result[key] = dict(state=data.get("state", data.get("status")), exit_code=data.get("exit_code", data.get("return_code")),
                               started=data.get("started"), finished=data.get("finished"), identity=identity)
    if not (run / "breakdown.json").exists():
        if result.get("supervisor", {}).get("state") == "failed":
            result["state"] = "failed_before_raw"
            log = run / "stderr.log"
            if log.exists():
                payload = log.read_bytes()
                result["stderr_identity"] = fingerprint(log, payload)
                lines = payload.decode(errors="replace").splitlines()
                result["last_error_line"] = next((s for s in reversed(lines) if s.strip()), None)
        else:
            result["pending_reason"] = "raw result absent; inspect recorded supervisor/dependency state"
        return result
    raw, identity = read(run / "breakdown.json")
    issues = result["issues"]
    commit = raw["source"]["source_commit"]
    require(commit == EXPECTED_TOOL, "tool revision differs from selected four-job plan", issues)
    result.update(raw_identity=identity, raw_state=raw["state"], last_stage=raw.get("last_stage"), source=raw["source"],
                  source_verification=source_check(run, commit, raw, issues),
                  arguments=raw["arguments"], schedule=raw["schedule"], parameters=raw.get("parameters"),
                  initial_parameters=raw.get("initial_parameters"), final_parameters=raw.get("final_parameters"),
                  input_sha256=raw.get("input_sha256"), input_shape=raw.get("input_shape"),
                  shared_input_prefix=raw.get("shared_input_prefix"), config=raw.get("config"), spec=raw.get("spec"),
                  hardware={k: raw.get(k) for k in ("physical_devices", "cpu_affinity", "torch_threads", "torch", "torch_npu", "allow_hf32", "resource_identity", "precision", "host_load_before", "host_load_after")},
                  normal=raw["normal"], boundary=raw["boundary"], process_peak_rss_kib=raw.get("process_peak_rss_kib"))
    for filename in ("launch.json", "environment.json"):
        if (run / filename).exists():
            _, value = read(run / filename)
            result[filename.removesuffix(".json") + "_identity"] = value
    require(raw["arguments"]["decode"] == 16 and raw["arguments"]["prefill"] == 128 and raw["arguments"]["batch"] == 8,
            "requested inference denominator/workload changed", issues)
    diagnoses = raw.get("diagnostics", {})
    result["stages"] = stage_summary(diagnoses.get("stages", {}), raw["subject"], 16, issues)
    result["operators"] = operator_summary(diagnoses.get("operators", {}), raw["subject"], 16, issues)
    if raw["state"] == "completed_finite":
        result["qualification"] = QUAL["infer_review"](raw, issues)
        result["state"] = "completed_review" if not issues else "completed_with_issues"
    elif raw["state"] == "failed":
        result.update(state="failed_raw", error=raw.get("error"), failure_kind=raw.get("failure_kind"))
    else:
        result["pending_reason"] = "raw job still active; completed diagnostic passes below remain separately identified"
    return result


def text_atomic(path, content):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".review.")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for device in ("cpu", "npu"):
        for subject in ("baseline", "transformer"):
            version = "v2" if subject == "transformer" else "v1"
            parser.add_argument(f"--{device}-{subject}-run", type=Path,
                default=Path(f"artifacts/ffn-infer-{device}-stages-{subject}-{version}"),
                help="explicit replacement run; original v1 lifecycle identity remains in previous_attempts")
    args = parser.parse_args()
    cases = []
    for device in ("cpu", "npu"):
        for subject in ("baseline", "transformer"):
            path = getattr(args, f"{device}_{subject}_run")
            path = path if path.is_absolute() else ROOT / path
            path = path.resolve()
            if not path.is_relative_to(ROOT / "artifacts"):
                parser.error("run directories must be inside this repository's artifacts")
            cases.append(review(device, subject, path))
    tables = {}
    for p in ("prefill", "decode"):
        available = [c for c in cases if c.get("stages", {}).get("state") == "completed"]
        names = sorted({r["mechanism"] for c in available for r in c["stages"]["exclusive_cut"] if r["phase"] == p})
        scale = 1 if p == "prefill" else 1000 / 16
        rows = [dict(mechanism=name, **{c["name"]: scale * sum(r["exclusive_seconds"] for r in c["stages"]["exclusive_cut"] if r["phase"] == p and r["mechanism"] == name) for c in available}) for name in names]
        tables[p] = dict(unit="seconds for all128 prefill positions" if p == "prefill" else "milliseconds per decode call; all16 captured calls including first",
                         rows=rows, totals={c["name"]: c["stages"]["scope_seconds"][p] * scale for c in available})
    ready = all(c["state"] == "completed_review" and c.get("supervisor", {}).get("state") == "passed" for c in cases)
    result = dict(schema="tide-inference-stage-review-v1", state="completed_review" if ready else "partial_snapshot",
                  reviewed_at=datetime.now(timezone.utc).isoformat(), generator=fingerprint(Path(__file__).resolve()),
                  tool_commit=EXPECTED_TOOL, production_tree=PRODUCTION_TREE,
                  collector_arguments={key: str(value) for key, value in vars(args).items()},
                  helper_identities=[fingerprint(COST_PATH), fingerprint(QUAL_PATH)], cases=cases, stage_tables=tables,
                  boundaries=[
                      "Read-only collector; no model/test/device work. Missing/running cases remain pending and failed evidence remains explicit.",
                      "Normal, synchronized stages and synchronized operators are independent requests starting from zero private state, with real prefill/decode continuation inside each request. No checkpoint recovery occurs in these inference jobs; dependencies on other recovery jobs do not qualify inference restoration.",
                      "Each job has one normal sample. Formal normal repeats are separate evidence. Normal stable decode omits the first of16 calls; diagnostic decode mean includes all16. Prefill is the total of four32-position chunks. Neither is language-model TTFT.",
                      "Normal output payloads/hashes are absent. Diagnostic complete readout/raw Tide values/strengths have full SHA with8 MiB copy pieces; matching hashes qualify only the two diagnostic outputs, not normal/VJP or complete private state.",
                      "Actual output int64 time values and named .times private tensors are recorded. SSD ConvState .conv.0 clocks have only shape/stride/dtype/device metadata, not value hashes; floating history payloads are not fully hashed. Metadata equality is not all private-clock/state value equality or an independent eviction/write oracle.",
                      "Allocation metadata callbacks add scalar reads after solver scopes. They are observer overhead, not production host synchronizations; solver invalid rows differ from active fallback requests.",
                      "Stage exclusive categories each partition one replay. Parent remainders include hook/dispatch and entry waits; release is a separate root. Operator sums are separate and are not normal-wall percentages or kernel-only times.",
                      "CPU synchronization helper is a no-op, while NPU hooks force device completion. Affinity, threads, hardware and activity differences preclude a controlled hardware speedup conclusion from these diagnostic times.",
                      "FFN bank gather events, actual padded matrices, KV-history operations, SSD preparation before Full and state organization are separately visible. Logical byte events are not traffic or allocator peaks; matrix M*K*N counts omit opaque attention/native internals.",
                      "Normal allocator peaks precede diagnostics; diagnostic allocator peaks include between-call checks. CPU process lifetime RSS includes construction and host observation, not a device-memory or activation ledger."])
    atomic(OUT / "inference-stage-review.json", result)
    lines = ["# Inference stage review", "", f"State: **{result['state']}**. Tool `{EXPECTED_TOOL[:8]}`, production tree `{PRODUCTION_TREE}`.", "",
             "Re-run: `python artifacts/ffn-execution-20261119/collect_inference_stages.py`. Optional `--cpu-transformer-run artifacts/<replacement>` and analogous flags select explicit replacement identities while retaining original v1 lifecycle records. This only reads frozen records and replaces the two review files atomically; it runs no models or devices.", "",
             "| Case | Review / raw state | Supervisor | Completed diagnostic passes |", "|---|---|---|---|"]
    for c in cases:
        complete = [p for p in ("stages", "operators") if c.get(p, {}).get("state") == "completed"]
        lines.append(f"| {c['name']} | {c['state']} / {c.get('raw_state', 'missing')} | {c.get('supervisor', {}).get('state', 'missing')} | {', '.join(complete) or 'pending'} |")
    for c in cases:
        if c.get("last_error_line"):
            lines += ["", f"{c['name']} failed before producing raw results: `{c['last_error_line']}`."]
        for previous in c.get("previous_attempts", []):
            lines += ["", f"{c['name']} selected `{c['path']}`; original `{previous['path']}` remains `{previous['state']}` (exit {previous['exit_code']})."]
    lines += ["", "B8, prompt128/chunk32/decode16, FP32. Tide KV128 has7.268409B parameters; fixed Transformer32 d4352/FFN15208/68×64 heads/window128 has8.778371B. This is an auxiliary reference, not A–D or equal parameter count. Each job contributes one normal witness; formal repeated normal timing is separate.", "",
              "Normal, stage and operator passes are independent requests. Prefill tables show the full128-position total; decode tables average all16 calls, including the first. Normal stable decode has15 calls. Stage exclusive categories partition only that replay; release is separately recorded. No diagnostic percentage is computed against normal timing."]
    for p, table in tables.items():
        lines += ["", f"Stage exclusive partition: **{p}**, {table['unit']}.", "", "| Mechanism | CPU Tide | CPU Transformer32 | NPU Tide | NPU Transformer32 |", "|---|---:|---:|---:|---:|"]
        order = ["cpu-baseline", "cpu-transformer", "npu-baseline", "npu-transformer"]
        for row in table["rows"]:
            lines.append("| " + row["mechanism"] + " | " + " | ".join(f"{row[k]:.6f}" if k in row else "pending" for k in order) + " |")
        lines.append("| Total captured | " + " | ".join(f"{table['totals'][k]:.6f}" if k in table["totals"] else "pending" for k in order) + " |")
    for c in cases:
        if c.get("operators", {}).get("state") != "completed":
            continue
        lines += ["", f"**{c['name']} separate operator evidence:**"]
        for p in ("prefill", "decode"):
            a = c["operators"]["allocation"]["by_phase"][p]
            banks = [r for r in c["operators"]["ffn_weight_bank_gathers"] if r["phase"] == p]
            lines += ["", f"{p}: fallback {a['bisect_calls']} calls, {a['bisect_requested_rows']} requested / {a['bisect_executed_rows']} executed rows; FFN float-bank gathers {sum(r['calls'] for r in banks)} calls, {sum(r['logical_output_bytes'] for r in banks):,} logical output bytes. Counts here cover the whole phase, not its mean time."]
            for row in c["operators"]["packing_from_operands"]:
                if row["phase"] == p:
                    lines.append(f"Memory {row['memory_kind']}: preparation projection rows {row['prepare_projection_rows']:,}; selected rows {row['selected_rows']:,}; executed FFN rows {row['executed_ffn_rows']:,} across {row['ffn_down_calls']} calls.")
        q = c.get("qualification")
        if q:
            counts = {k: len(v["final_int64_tensors_without_value_hash"]) for k, v in q["diagnostics"].items()}
            lines += ["", f"Diagnostic metadata identical: {q['independently_recomputed_metadata_identical']}; complete output SHA identical: {q['independently_recomputed_output_identity_identical']}. Final int64 tensors without value hashes: {counts}. This does not compare normal output payloads or qualify checkpoint recovery."]
    lines += ["", "Actual output int64 times and named private `.times` values retain their recorded checks. SSD convolution `.conv.0` clocks have only metadata; floating KV/SSD/history values are not fully hashed. Matching diagnostic metadata/output SHA does not establish equality of all private state or clock values. Output hashing uses bounded8 MiB host pieces outside capture.", "",
              "CPU32-thread affinity and NPU4-thread physical-device identities remain in JSON. CPU synchronization calls are no-ops; NPU observers force completion. Diagnostic hardware ratios, best-batch performance and task quality are not inferred. Actual matrix/stride records, backend/dtype observations, KV-history/SSD preparation/state cuts, source/input/output fingerprints, numerical boundaries and pending/failed details are in [inference-stage-review.json](inference-stage-review.json)."]
    text = "\n".join(lines) + "\n"
    for before, after in (("has7.", "has 7."), ("has8.", "has 8."), ("full128", "full 128"), ("all128", "all 128"), ("all16", "all 16"), ("has15", "has 15"), ("four32", "four 32"), ("bounded8", "bounded 8"), ("CPU32", "CPU 32"), ("NPU4", "NPU 4")):
        text = text.replace(before, after)
    text_atomic(OUT / "inference-stage-review.md", text)
    print(json.dumps(dict(state=result["state"], cases={c["name"]: c["state"] for c in cases},
                          issues={c["name"]: c["issues"] for c in cases if c["issues"]},
                          outputs=[fingerprint(OUT / ("inference-stage-review." + suffix)) for suffix in ("json", "md")])))


if __name__ == "__main__":
    main()
