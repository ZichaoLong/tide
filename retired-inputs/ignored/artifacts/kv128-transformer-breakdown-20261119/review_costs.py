#!/usr/bin/env python3
"""Read-only raw-record cost review; writes only the adjacent cost-review.json.

No Torch import, model construction, device admission, job launch or raw edits.
Running and missing records remain visible. Re-run after the four jobs finish.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("cost-review.json")
JOBS = ("infer-kv128", "infer-tf32", "train-kv128", "train-tf32")
EXPECTED_SOURCE = "7196fcabd7ab3b401c1ebdb6107442450a99996c"


def atomic(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".cost-review-", delete=False) as f:
        temporary = Path(f.name)
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def canonical(path):
    return re.sub(r"(?<=/)call_\d+(?=/|$)", "request_call", path)


def phase(path):
    parts = path.split("/")
    if parts[0] == "training":
        return "training." + (parts[1] if len(parts) > 1 else "root_exclusive")
    if parts[0] == "release":
        return "release." + parts[1]
    if parts[0] in {"prefill", "decode"}:
        return parts[0]
    return "outside_training_scope"


def broad_module(path, subject):
    parts = path.split("/")
    if subject == "transformer":
        for part in parts:
            if part in {"ffn", "attention", "attention_norm", "ffn_norm", "qkv", "output",
                        "rope", "layout.rope_masks", "final_norm", "readout"}:
                return "transformer." + part
        if "block" in parts:
            return "transformer.block_remainder"
    else:
        for part in parts:
            if part in {"reception", "description", "regional.scoring", "allocation", "messages.gather",
                        "regions.group", "input.projection", "state.join", "state.split", "readout"} or part.startswith("memory."):
                return part
    if "backward" in parts:
        return "autograd_unattributed"
    if "optimizer" in parts:
        return "AdamW"
    return "unattributed_scope_remainder"


def mathematical_module(path, subject):
    parts = path.split("/")
    broad = broad_module(path, subject)
    kind = next((part.split(".")[1] for part in parts if part.startswith("memory.")), None)
    prefix = "memory."+kind+"." if kind else "transformer." if subject == "transformer" else ""
    if "ffn" in parts:
        return prefix+"ffn"
    if "queries.pack" in parts:
        return prefix+"queries.pack"
    if "queries.publish" in parts:
        return prefix+"queries.publish"
    if "attention" in parts:
        return prefix+"attention" if kind or subject == "transformer" else broad+".attention"
    if "head_query" in parts:
        return prefix+"query_projection"
    for part in reversed(parts):
        if part.startswith(("kv.history.", "ssd.", "node_linear.", "node_vector.")):
            return part
    return broad


def fine_scope(path):
    parts = path.split("/")
    leaf = parts[-1]
    if leaf == "ffn":
        return "ffn.scope_remainder"
    if "ffn" in parts:
        return "ffn."+leaf
    if leaf.startswith(("memory.", "call_")) or leaf in {"prefill", "decode", "backbone", "chunk", "regions.evaluate", "region.evaluate", "block", "forward", "training"}:
        return "scope_remainder"
    return leaf


def mechanism_module(path, subject):
    """One disjoint mechanism cut; includes pre-FFN norm on both subjects."""
    parts = path.split("/")
    broad = broad_module(path, subject)
    if "ffn" in parts or "ffn_norm" in parts:
        return "FFN_including_its_normalization"
    if "queries.pack" in parts:
        return "Full_query_packing"
    if "queries.publish" in parts:
        return "Full_query_publication"
    if "attention" in parts:
        return "Attention_read_scope." + broad
    if "kv.history.order" in parts:
        return "KV_history_order_including_compaction"
    if "kv.history.gather" in parts:
        return "KV_history_field_gather"
    if broad == "memory.K.prepare":
        return "KV_prepare_other"
    if broad == "memory.K.full":
        return "KV_Full_other"
    if broad == "memory.S.full":
        return "SSD_Full_other"
    if broad == "regional.scoring":
        return "regional_scoring_other"
    return broad


def aggregate_rows(rows, key_function, value_key, count_key="calls"):
    groups = defaultdict(lambda: dict(seconds=0., calls=0, source_rows=0))
    for row in rows:
        target = groups[key_function(row)]
        target["seconds"] += row[value_key]
        target["calls"] += row[count_key]
        target["source_rows"] += 1
    return groups


def stage_review(diagnostic, subject):
    report = diagnostic.get("report")
    if report is None:
        return dict(state=diagnostic.get("state", "missing"), report_available=False)
    rows = report["rows"]
    tree = defaultdict(lambda: dict(calls=0, inclusive_seconds=0., exclusive_seconds=0.))
    for row in rows:
        node = tree[canonical(row["path"])]
        for key in node:
            node[key] += row[key]
    parent = {}
    for path in tree:
        pieces = path.split("/")
        parent[path] = next(("/".join(pieces[:i]) for i in range(len(pieces)-1, 0, -1)
                             if "/".join(pieces[:i]) in tree), None)
    children = defaultdict(float)
    for path, node in tree.items():
        if parent[path] is not None:
            children[parent[path]] += node["inclusive_seconds"]
    errors = {path: node["inclusive_seconds"]-node["exclusive_seconds"]-children[path]
              for path, node in tree.items()}
    broad = aggregate_rows(rows, lambda r: (phase(r["path"]), broad_module(r["path"], subject)), "exclusive_seconds")
    detail = aggregate_rows(rows, lambda r: (phase(r["path"]), broad_module(r["path"], subject), fine_scope(r["path"])), "exclusive_seconds")
    mechanism = aggregate_rows(rows, lambda r: (phase(r["path"]), mechanism_module(r["path"], subject)), "exclusive_seconds")
    totals = defaultdict(float)
    for (p, _), value in broad.items():
        totals[p] += value["seconds"]
    explicit = {path: node["inclusive_seconds"] for path, node in tree.items()
                if path in {"prefill", "decode", "training"} or
                path.startswith("training/") and path.count("/") == 1}
    views = []
    labels = {"ffn", "attention", "queries.pack", "queries.publish", "head_query", "allocation",
              "kv.history.order", "kv.history.compaction", "kv.history.gather", "ssd.convolution",
              "ssd.matrix", "ssd.summaries", "state.join", "state.split"}
    for p in sorted(totals):
        phase_rows = [r for r in rows if phase(r["path"]) == p]
        for label in sorted(labels):
            selected = [r for r in phase_rows if r["path"].split("/")[-1] == label]
            selected = [r for r in selected if not any(r["path"].startswith(other["path"]+"/") for other in selected if r is not other)]
            if selected:
                views.append(dict(phase=p, label=label, inclusive_seconds=sum(r["inclusive_seconds"] for r in selected),
                                  calls=sum(r["calls"] for r in selected), paths=[r["path"] for r in selected]))
    checks = dict(maximum_tree_reconciliation_absolute_seconds=max(map(abs, errors.values()), default=0.),
                  negative_exclusive_rows=[r for r in rows if r["exclusive_seconds"] < -1e-9],
                  nonfinite_rows=[r for r in rows if not math.isfinite(r["inclusive_seconds"]+r["exclusive_seconds"])],
                  observer_boundary_failures=report.get("boundary_failures", []))
    return dict(state=diagnostic.get("state"), report_available=True, explicit_scope_seconds=explicit,
        summed_exclusive_by_phase=dict(totals),
        nonoverlapping_broad_cut=[dict(phase=p, module=m, **v) for (p, m), v in sorted(broad.items())],
        nonoverlapping_detail_cut=[dict(phase=p, module=m, detail=d, **v) for (p, m, d), v in sorted(detail.items())],
        nonoverlapping_mechanism_cut=[dict(phase=p, module=m, **v) for (p, m), v in sorted(mechanism.items())],
        selected_subtree_views=views,
        canonical_tree=[dict(path=p, parent=parent[p], children_inclusive_seconds=children[p],
                             reconciliation_error_seconds=errors[p], **v) for p, v in sorted(tree.items())],
        checks=checks, boundary=report.get("boundary"),
        interpretation="Broad/detail cuts each assign every scope's exclusive time exactly once. They are alternative cuts, not additive tables. Selected inclusive subtree views overlap parent scopes and each other; do not add them to either cut. Canonical request_call merges call IDs only. Entry waits for prior work can remain in an enclosing exclusive row; remainders are not pure Python overhead.")


def weight_index(row):
    path, op = row["path"], row["operator"].lower()
    owned = any(token in path for token in ("node_linear.", "head_query", "node_vector.", "node_linears.fused"))
    return (owned and ("index" in op or "gather" in op) and "backward" not in op
            and bool(row["inputs"]) and "float" in row["inputs"][0]["dtype"])


def operator_class(row):
    op = row["operator"].lower()
    if weight_index(row):
        return "parameter_index_in_owned_projection_scope"
    if "attention" in op:
        return "native_attention_or_attention_backward"
    if any(token in op for token in ("matmul", ".mm.", ".bmm.", "addmm", "baddbmm", ".linear.")):
        return "matrix_or_matrix_backward"
    if "convolution" in op:
        return "convolution"
    if any(token in op for token in ("gather", "index", "scatter", "slice_backward", "select_backward")):
        return "data_index_scatter_or_backward"
    if any(token in op for token in ("cumsum", "cumprod")):
        return "prefix_scan"
    if any(token in op for token in (".cat.", ".stack.")):
        return "concatenation"
    if any(token in op for token in ("copy", "clone", "contiguous")):
        return "copy_or_clone"
    if any(token in op for token in ("layer_norm", "rms_norm", "native_norm")):
        return "native_normalization"
    if any(token in op for token in (".item.", "_local_scalar_dense")):
        return "host_scalar"
    if any(token in op for token in ("view", "reshape", "transpose", "permute", "split", "unbind", "expand", "as_strided", ".slice.", ".select.", "detach", "squeeze", ".t.")):
        return "view_or_metadata"
    if any(token in op for token in (".sum.", ".mean.", ".amax.", ".amin.", ".min.", ".max.", ".all.", ".any.", ".prod.", ".arg")):
        return "reduction"
    if any(token in op for token in ("empty", "zeros", "ones", "full", "arange", "scalar_tensor", "new_")):
        return "tensor_creation"
    if "foreach" in op:
        return "foreach_update"
    return "pointwise_or_other"


def matrix_work(row):
    op, values = row["operator"], row["inputs"]
    if "backward" in op:
        return None
    offset = 1 if "addmm" in op or "baddbmm" in op else 0
    if len(values) < offset+2:
        return None
    a, b = values[offset]["shape"], values[offset+1]["shape"]
    if len(a) < 2 or len(b) < 2 or a[-1] != b[-2]:
        return None
    left, right = a[:-2], b[:-2]
    left, right = [1]*(max(len(left), len(right))-len(left))+left, [1]*(max(len(left), len(right))-len(right))+right
    if any(u != v and u != 1 and v != 1 for u, v in zip(left, right)):
        return None
    batch = math.prod(max(u, v) for u, v in zip(left, right))
    m, k, n = a[-2], a[-1], b[-1]
    return dict(batch=batch, M=m, K=k, N=n, per_call_dense_multiply_adds=batch*m*k*n,
                all_calls_dense_multiply_adds=batch*m*k*n*row["calls"])


def logical_bytes(outputs):
    sizes = {"torch.float32": 4, "torch.float64": 8, "torch.bfloat16": 2, "torch.float16": 2,
             "torch.int64": 8, "torch.int32": 4, "torch.int16": 2, "torch.int8": 1, "torch.uint8": 1, "torch.bool": 1}
    if any(t["dtype"] not in sizes for t in outputs):
        return None
    return sum(math.prod(t["shape"])*sizes[t["dtype"]] for t in outputs)


def packing_work(rows):
    """Recover semantic publish rows and actual FFN rectangles from operands.

    Queries.publish has exactly one index_put with three selected-row indices.
    Each independent FFN has exactly one down matrix contraction per pack.
    This relies on the pinned production source, never on cache capacity.
    """
    groups = defaultdict(lambda: dict(selected_rows=0, executed_rows=0, publish_calls=0, ffn_down_calls=0,
                                      prepare_projection_rows=0, prepare_projection_calls=0,
                                      publish_shape_groups=[], down_shape_groups=[]))
    invalid = []
    for row in rows:
        kind = next((part.split(".")[1] for part in row["path"].split("/") if part.startswith("memory.")), None)
        if kind is None or phase(row["path"]) not in {"training.forward", "prefill", "decode"}:
            continue
        group = groups[phase(row["path"]), kind]
        prepare_name = "kv" if kind == "K" else "input"
        if (row["path"].endswith("node_linear.memories."+kind+"."+prepare_name) and
                row["operator"] in {"aten.bmm.default", "aten.mm.default", "aten.matmul.default"}):
            group["prepare_projection_rows"] += math.prod(row["outputs"][0]["shape"][:-1])*row["calls"]
            group["prepare_projection_calls"] += row["calls"]
        if row["path"].endswith("queries.publish") and row["operator"] == "aten.index_put.default":
            inputs = row["inputs"]
            if (len(inputs) != 5 or len(inputs[-1]["shape"]) != 2 or
                    any(t["shape"] != [inputs[-1]["shape"][0]] or t["dtype"] != "torch.int64" for t in inputs[1:4])):
                invalid.append(row)
                continue
            selected = inputs[-1]["shape"][0]
            group["selected_rows"] += selected*row["calls"]
            group["publish_calls"] += row["calls"]
            group["publish_shape_groups"].append(dict(path=canonical(row["path"]), calls=row["calls"],
                selected_per_call=selected, output_shape=row["outputs"][0]["shape"]))
        if (row["path"].endswith("node_linear.memories."+kind+".ffn.down") and
                row["operator"] in {"aten.bmm.default", "aten.mm.default", "aten.matmul.default"}):
            shape = row["outputs"][0]["shape"]
            executed = math.prod(shape[:-1])
            group["executed_rows"] += executed*row["calls"]
            group["ffn_down_calls"] += row["calls"]
            group["down_shape_groups"].append(dict(path=canonical(row["path"]), calls=row["calls"],
                executed_per_call=executed, inputs=row["inputs"], output_shape=shape))
    for group in groups.values():
        group["publish_down_calls_match"] = group["publish_calls"] == group["ffn_down_calls"]
        group["padding_rows"] = group["executed_rows"]-group["selected_rows"]
        group["execution_to_selected_ratio"] = group["executed_rows"]/group["selected_rows"] if group["selected_rows"] else None
    return dict(rows=[dict(phase=p, memory_kind=k, **v) for (p, k), v in sorted(groups.items())],
        invalid_publish_operand_rows=invalid,
        definition="Selected rows from actual Queries.publish index_put index/value lengths; executed rows from one independent FFN down contraction per pack; prepare rows from KV kv or SSD input projection operands. Tensor-rectangle padding only, not hidden backend tile padding. All derive from this operator pass; not semantic history capacity or an activity trajectory for normal samples.")


def operator_review(diagnostic, subject):
    report = diagnostic.get("report")
    if report is None:
        return dict(state=diagnostic.get("state", "missing"), report_available=False)
    rows = report["rows"]
    enriched = [dict(row, phase=phase(row["path"]), module=mathematical_module(row["path"], subject),
                     category=operator_class(row)) for row in rows]
    grouped = aggregate_rows(enriched, lambda r: (r["phase"], r["module"], r["category"]), "seconds")
    phase_category = aggregate_rows(enriched, lambda r: (r["phase"], r["category"]), "seconds")
    modules = aggregate_rows(enriched, lambda r: (r["phase"], r["module"]), "seconds")
    operators = aggregate_rows(enriched, lambda r: (r["phase"], r["operator"]), "seconds")
    phases = aggregate_rows(enriched, lambda r: r["phase"], "seconds")
    matrix_groups = {}
    for row in enriched:
        if row["category"] in {"matrix_or_matrix_backward", "native_attention_or_attention_backward"}:
            fields = dict(path=canonical(row["path"]), phase=row["phase"], module=row["module"],
                          operator=row["operator"], inputs=row["inputs"], outputs=row["outputs"])
            key = json.dumps(fields, sort_keys=True)
            target = matrix_groups.setdefault(key, dict(**fields, calls=0, seconds=0.))
            target["calls"] += row["calls"]
            target["seconds"] += row["seconds"]
    matrices = list(matrix_groups.values())
    for row in matrices:
        row["dense_matrix_work"] = matrix_work(row) if "attention" not in row["operator"] else None
    byte_groups = defaultdict(lambda: dict(calls=0, logical_output_bytes=0, seconds=0., unknown_byte_rows=0))
    for row in enriched:
        if row["category"] not in {"parameter_index_in_owned_projection_scope", "data_index_scatter_or_backward", "concatenation", "copy_or_clone"}:
            continue
        group = byte_groups[row["phase"], row["module"], row["category"]]
        size = logical_bytes(row["outputs"])
        group["calls"] += row["calls"]
        group["seconds"] += row["seconds"]
        if size is None:
            group["unknown_byte_rows"] += 1
        else:
            group["logical_output_bytes"] += size*row["calls"]
    total = sum(row["seconds"] for row in rows)
    return dict(state=diagnostic.get("state"), report_available=True,
        intrusive_pass_elapsed_seconds=diagnostic.get("intrusive_elapsed_seconds", diagnostic.get("pass_elapsed_seconds_including_excluded_work")),
        elapsed_boundary="training elapsed includes observer and scalar return; inference elapsed additionally includes hashes/checks/JSON IO. Neither is normal timing.",
        operator_seconds=total, dispatched_calls=sum(row["calls"] for row in rows),
        by_phase=[dict(phase=p, **v) for p, v in sorted(phases.items())],
        by_phase_category=[dict(phase=p, category=c, **v) for (p, c), v in sorted(phase_category.items())],
        by_phase_mathematical_module=[dict(phase=p, module=m, **v) for (p, m), v in sorted(modules.items())],
        by_phase_module_category=[dict(phase=p, module=m, category=c, **v) for (p, m, c), v in sorted(grouped.items())],
        top_operators_by_phase={p: [dict(operator=op, **v) for (q, op), v in sorted(operators.items(), key=lambda item: -item[1]["seconds"]) if q == p][:25] for p in phases},
        top_operator_shape_groups=sorted(enriched, key=lambda row: -row["seconds"])[:40],
        matrix_and_attention_groups=sorted(matrices, key=lambda row: -row["seconds"]),
        packed_query_work=packing_work(enriched),
        repeated_logical_output_bytes=[dict(phase=p, module=m, category=c, **v) for (p, m, c), v in sorted(byte_groups.items())],
        checks=dict(reported_seconds_difference=total-report["operator_seconds"],
                    reported_call_difference=sum(row["calls"] for row in rows)-report["dispatched_calls"],
                    observer_failed_calls=report.get("failed_calls", 0), observer_failures=report.get("failures", [])),
        boundary=report.get("boundary"),
        interpretation="Every dispatch row belongs to one phase/module/category. These are alternative partitions of one operator pass, never added to stages or normal timing. Floating index/gather within NodeLinear/HeadQuery/NodeVector/fused projection scope is called parameter indexing; other indexing remains data indexing. Ordinary backward has no mathematical owner labels except observed recomputation scopes. Matrix metadata is actual operands; dense MAdds only cover recognizable forward contractions, not opaque native attention, backward or complete FLOPs. Repeated logical output bytes are not bus traffic, unique allocation or peak memory; in-place destinations may be counted repeatedly.")


def review(name):
    path = ROOT/"artifacts"/("breakdown-"+name+"-v1")/"breakdown.json"
    if not path.exists():
        return dict(state="missing", path=str(path.relative_to(ROOT)))
    data = path.read_bytes()
    identity = dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    try:
        raw = json.loads(data)
    except Exception as error:
        return dict(state="parse_error", file=identity, error=repr(error))
    result = dict(file=identity, state=raw["state"], last_stage=raw.get("last_stage"), source=raw["source"],
        subject=raw["subject"], arguments=raw["arguments"], spec=raw.get("spec"), parameters=raw.get("parameters"),
        physical_devices=raw.get("physical_devices"), precision=raw.get("precision"), input_shape=raw.get("input_shape"),
        input_sha256=raw.get("input_sha256"), shared_input_prefix=raw.get("shared_input_prefix"),
        anchor=raw.get("anchor"), normal_summary=raw.get("normal_summary", raw.get("normal", {}).get("summary") if isinstance(raw.get("normal"), dict) else None),
        normal_boundary=raw.get("boundary", {}).get("normal"), source_matches=raw["source"]["source_commit"] == EXPECTED_SOURCE)
    normal = raw.get("normal", [])
    observations = normal.get("observations", []) if isinstance(normal, dict) else normal
    result["normal_observations"] = observations
    result["diagnostic_boundaries"] = raw.get("boundary")
    diagnostics = raw.get("diagnostics", {})
    result["stages"] = stage_review(diagnostics.get("stages", {}), raw["subject"])
    result["operators"] = operator_review(diagnostics.get("operators", {}), raw["subject"])
    result["numerical_scope"] = {key: {k: value[k] for k in ("state", "numerical_state", "state_before", "state_after", "capture_start_position", "loss") if k in value} for key, value in diagnostics.items()}
    result["replay_checks"] = raw.get("replay_checks")
    return result


def report_tables(cases):
    result = {}
    for phase_name, case_names, unit, multiplier in (
            ("training.forward", ["train-kv128", "train-tf32"], "seconds", 1.),
            ("prefill", ["infer-kv128", "infer-tf32"], "seconds_for_full_prefill", 1.),
            ("decode", ["infer-kv128", "infer-tf32"], "milliseconds_per_captured_decode_call", 1000.)):
        values, divisors = {}, {}
        for name in case_names:
            case = cases[name]
            if case["state"] != "completed_finite" or not case.get("stages", {}).get("report_available"):
                continue
            tree = case["stages"]["canonical_tree"]
            denominator = next(row["calls"] for row in tree if row["path"] == "decode") if phase_name == "decode" else 1
            divisors[name] = denominator
            values[name] = {row["module"]: row["seconds"]*multiplier/denominator
                            for row in case["stages"]["nonoverlapping_mechanism_cut"] if row["phase"] == phase_name}
        modules = sorted({module for table in values.values() for module in table})
        result[phase_name] = dict(unit=unit, captured_call_denominator=divisors,
            rows=[dict(module=module, **{name: values[name].get(module, 0.) if name in values else None
                                        for name in case_names}) for module in modules],
            totals={name: sum(table.values()) for name, table in values.items()},
            boundary="A disjoint sum of stage exclusive rows. FFN includes its pre-normalization in both subjects. Decode averages all 16 captured calls, including the first; ordinary stable decode omits the first. No division by normal timing.")
    result["backward_operator_classes"] = {
        name: sorted([row for row in cases[name].get("operators", {}).get("by_phase_category", [])
                      if row["phase"] == "training.backward"], key=lambda row: -row["seconds"])
        for name in ("train-kv128", "train-tf32")}
    return result


def main():
    cases = {name: review(name) for name in JOBS}
    result = dict(schema="tide-kv128-transformer-breakdown-cost-review-v1", observed=datetime.now(timezone.utc).isoformat(),
        generator=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        state="completed_review" if all(v["state"] == "completed_finite" for v in cases.values()) else "partial_snapshot",
        cases=cases, report_tables=report_tables(cases),
        boundaries=["No source/raw mutation or new experiment; each raw fingerprint identifies exactly the bytes parsed in this snapshot.",
            "Stage and operator passes independently replay; their seconds must never be added, or divided by normal latency as wall-time shares.",
            "Stage timing includes completion synchronization and hook/host overhead. Entry waits before a child timer can remain in its parent's exclusive time.",
            "Operator seconds include dispatch and completion waits but exclude entry waits and descriptor bookkeeping; they are not native hardware kernel time.",
            "Training normal backward seconds include concatenation/loss; separate stage loss/backward are the only isolated loss/backward measurements.",
            "Medium training uses d192/KV32 versus a new 32-layer near-total constructor; this is not 9B training. Inference uses d768/KV128 versus fixed d4352/FFN15208/32-layer reference.",
            "Normal samples are finite engineering observations, not task-quality equivalence. Same-candidate diagnostic output/state/update qualifications remain separate.",
            "Autograd backward operators without forward module scopes retain unknown mathematical ownership; shapes alone do not establish FFN attribution."])
    atomic(OUT, result)
    print(json.dumps(dict(state=result["state"], output=str(OUT.relative_to(ROOT)), cases={k: v["state"] for k, v in cases.items()})))


if __name__ == "__main__":
    main()
