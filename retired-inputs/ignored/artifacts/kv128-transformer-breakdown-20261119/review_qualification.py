"""Read frozen breakdown records; write only this directory's review artifact."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.job import atomic

EXPECTED_SOURCE = "7196fcabd7ab3b401c1ebdb6107442450a99996c"
KINDS = ("infer-kv128", "infer-tf32", "train-kv128", "train-tf32")
DESTINATION = Path(__file__).resolve().parent / "qualification-review.json"


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(test, description, issues):
    if not test:
        issues.append(description)


def operator_review(diagnostic, issues):
    report = diagnostic.get("report", {})
    rows = report.get("rows", [])
    require(sum(row["calls"] for row in rows) == report.get("dispatched_calls"),
            "operator grouped-call sum differs from reported total", issues)
    require(math.isclose(sum(row["seconds"] for row in rows), report.get("operator_seconds", -1),
                         rel_tol=1e-12, abs_tol=1e-12), "operator grouped-time sum differs from total", issues)
    require(not report.get("failed_calls") and not report.get("failures"), "operator failure recorded", issues)
    require(all(math.isfinite(row["seconds"]) and row["seconds"] >= 0 for row in rows),
            "operator elapsed time nonfinite or negative", issues)
    operations, dtypes_devices, matrices = Counter(), Counter(), []
    for index, row in enumerate(rows):
        operations[row["operator"]] += row["calls"]
        for dtype, device in {(item["dtype"], item["device"]) for item in row["inputs"]+row["outputs"]}:
            dtypes_devices[dtype, device] += row["calls"]
        if any(token in row["operator"] for token in ("mm.", "bmm.", "addmm.", "matmul.", "linear.", "attention", "convolution")):
            matrices.append(dict(raw_row=index, **row))
    matrix_dtypes_devices = Counter()
    for row in matrices:
        for dtype, device in {(item["dtype"], item["device"]) for item in row["outputs"]}:
            matrix_dtypes_devices[dtype, device] += row["calls"]
    return dict(
        grouped_rows=len(rows), dispatched_calls=report.get("dispatched_calls"),
        failed_calls=report.get("failed_calls"), operator_seconds=report.get("operator_seconds"),
        observed_operators=dict(sorted(operations.items())),
        tensor_dtype_device_call_presence=[dict(dtype=dtype, device=device, calls=calls)
            for (dtype, device), calls in sorted(dtypes_devices.items())],
        matrix_group_count=len(matrices), matrix_calls=sum(row["calls"] for row in matrices),
        matrix_operators=sorted({row["operator"] for row in matrices}),
        matrix_output_dtype_device_call_presence=[dict(dtype=dtype, device=device, calls=calls)
            for (dtype, device), calls in sorted(matrix_dtypes_devices.items())],
        matrix_examples=sorted(matrices, key=lambda row: row["seconds"], reverse=True)[:12],
        boundary=report.get("boundary"), interpretation=report.get("interpretation"),
        limits="metadata is from intercepted dispatcher tensors on this one replay; a call can contain several native kernels or no device kernel; dtype presence is not a native-kernel precision proof; scalar argument values are not grouping keys",
    )


def stage_review(diagnostic, issues):
    report = diagnostic.get("report", {})
    rows = report.get("rows", [])
    require(not report.get("boundary_failures"), "stage synchronization failure recorded", issues)
    require(all(math.isfinite(row[key]) and row[key] >= 0 for row in rows
                for key in ("inclusive_seconds", "exclusive_seconds")),
            "stage elapsed time nonfinite or negative", issues)
    return dict(grouped_rows=len(rows), boundary=report.get("boundary"),
                interpretation=report.get("interpretation"),
                boundary_failures=report.get("boundary_failures"))


def train_review(record, issues):
    args, anchor = record["arguments"], record["anchor"]
    normal, warmup = record["normal"], record["warmup"]
    require(len(normal) == args["samples"] and len(warmup) == args["warmup"],
            "training sample count differs from frozen arguments", issues)
    previous = 0
    finite_fields = ("finite_output", "finite_parameters", "finite_gradients")
    continuation_fields = ("optimizer_finite", "private_state_finite", "private_state_detached")
    for row in warmup + normal:
        require(row["state_before"] == previous and row["state_after"] == previous+args["length"],
                "training normal continuation position differs", issues)
        require(all(row[key] for key in finite_fields) and
                all(row["continuation"][key] for key in continuation_fields),
                "normal training finite/detached invariant failed", issues)
        previous = row["state_after"]
    require(anchor["private_state_position"] == warmup[-1]["state_after"],
            "training anchor position is not final warmup position", issues)
    reference = record["first_measured_update"]
    anchor_keys = ("model_sha256", "optimizer_sha256", "gradients_sha256", "rng_sha256")
    fields = {"parameters", "optimizer", "gradients", "private_state", "output"}
    diagnoses = {}
    for kind in args["diagnostics"]:
        row = record["diagnostics"][kind]
        comparisons = row["normal_update_comparison"]
        require(row["state"] == "completed_finite", kind+" training diagnostic incomplete", issues)
        restored = row["anchor_restored"] == {key: anchor[key] for key in anchor_keys}
        private_restored = row["private_state_start_sha256"] == anchor["private_state_sha256"]
        require(restored and private_restored, kind+" training anchor identity differs", issues)
        require(row["state_before"] == anchor["private_state_position"] and
                row["state_after"] == normal[0]["state_after"], kind+" training replay position differs", issues)
        require(set(comparisons) == fields and set(row["update_identity"]) == fields,
                kind+" training comparison fields incomplete", issues)
        require(all(c["atol"] == 1e-6 and c["rtol"] == 2e-3 for c in comparisons.values()),
                kind+" training thresholds changed", issues)
        require(all(c["finite"] and not c["metadata_mismatches"] and not c["support_mismatches"]
                    for c in comparisons.values()), kind+" structural or finite comparison failure", issues)
        exact_hashes = {key: row["update_identity"][key] == reference[key] for key in fields}
        require(all(exact_hashes[key] == comparisons[key]["exact"] for key in fields),
                kind+" complete SHA identity disagrees with tensor exactness", issues)
        require(all(row["continuation"][key] for key in continuation_fields),
                kind+" diagnostic continuation invariant failed", issues)
        diagnoses[kind] = dict(
            state=row["state"], numerical_state=row["numerical_state"],
            model_optimizer_gradients_rng_anchor_identical=restored,
            private_state_anchor_identical=private_restored,
            start_position=row["state_before"], final_position=row["state_after"],
            loss=row["loss"], first_normal_loss=normal[0]["loss"],
            exact_loss_identical=row["loss"] == normal[0]["loss"],
            complete_update_sha256=row["update_identity"],
            exact_hash_identity_by_field=exact_hashes,
            full_tensor_comparisons=comparisons,
            finite_nonexact_fields=sorted(key for key, value in comparisons.items()
                                         if value["finite"] and not value["exact"]),
            original_threshold_failed_fields=sorted(key for key, value in comparisons.items()
                                                   if not value["within_threshold"]),
            continuation=row["continuation"],
            observation=operator_review(row, issues) if kind == "operators" else stage_review(row, issues))
    first, second = (record["diagnostics"][kind] for kind in args["diagnostics"])
    return dict(
        anchor=anchor, first_measured_update=reference, diagnostics=diagnoses,
        cross_diagnostic_exact_sha_by_field={key: first["update_identity"][key] == second["update_identity"][key]
                                            for key in sorted(fields)},
        normal_state_ranges=[[row["state_before"], row["state_after"]] for row in normal],
        normal_state_dtypes=[row["state_memory"]["dtypes"] for row in normal],
        input_requires_grad=False,
        normal_retention="all warmup/normal updates retain scalar loss, finite flags, timings and state metadata; only the first normal update retains complete aggregate SHAs of joined readout, parameters, gradients, AdamW and private state; actual CPU snapshots were compared in-process but not saved as durable payload; per-call raw Tide values/strengths and input gradients are not qualification fields here",
        restoration_scope="in-process CPU anchor restore of warmed model, AdamW slots, parameter gradients, RNG and detached private continuation; a genuine following update is compared; this is not a disk checkpoint roundtrip",
        numerical_scope="same mathematical candidate and warmed trajectory with and without intrusive hooks; original atol=1e-6, rtol=2e-3 retained; finite nonexactness is reported without assigning a rounding cause; does not supersede older candidate qualification failures",
        localization_limit="Exact retained forward readout/private state with nonexact parameter gradients locates the first retained difference at gradients, not at a particular backward operator: intermediate forward activations and per-operator VJPs were not retained, so this does not establish the cause.",
    )


def replay_metadata(row):
    return dict(outputs=row["outputs"], final_state=row["final_state"], calls=[
        {**{key: call[key] for key in ("index", "phase", "start", "stop", "input_positions",
                                      "state_before", "state_after", "private_state")},
         "output": {key: value for key, value in call["output"].items() if key != "identities"}}
        for call in row["calls"]])


def infer_review(record, issues):
    args, normal = record["arguments"], record["normal"]
    expected = [("prefill", start, min(start+args["prefill_chunk"], args["prefill"]))
                for start in range(0, args["prefill"], args["prefill_chunk"])]
    expected += [("decode", start, start+1) for start in
                 range(args["prefill"], args["prefill"]+args["decode"])]
    require(normal["state"] == "completed" and len(normal["observations"]) == args["samples"]
            and len(normal["warmup"]) == args["warmup"], "inference normal sample count/state differs", issues)
    require(record["initial_parameters"] == record["final_parameters"],
            "inference final parameters differ", issues)
    for sample in normal["observations"]:
        require([(r["phase"], r["start"], r["stop"]) for r in sample["calls"]] == expected,
                "normal inference schedule differs", issues)
        require(sample["diagnostic_checks_in_timing"] is False,
                "normal inference includes diagnostic checks", issues)
    diagnoses = {}
    for kind in args["diagnostics"]:
        row = record["diagnostics"][kind]
        require(row["state"] == "completed_finite", kind+" inference diagnostic incomplete", issues)
        require(row["initial_parameters"] == record["initial_parameters"] and
                row["input_sha256"] == record["input_sha256"], kind+" inference replay inputs differ", issues)
        require([(r["phase"], r["start"], r["stop"]) for r in row["calls"]] == expected,
                kind+" diagnostic schedule differs", issues)
        for call in row["calls"]:
            private, output = call["private_state"], call["output"]
            require(call["state_before"] == call["start"] and call["state_after"] == call["stop"],
                    kind+" inference state positions differ", issues)
            require(private["finite"] and private["detached"] and output["finite_through_call"],
                    kind+" inference finite/detached invariant failed", issues)
            if record["subject"] == "tide":
                require(private["times_int64"] is True and private["private_times"] and
                        output["time"].get("dtype") == "torch.int64", kind+" raw int64 time evidence absent", issues)
                require(set(output["identities"]) == {"readout", "raw_values", "raw_strengths"},
                        kind+" Tide output identity fields missing", issues)
            else:
                require(private["times_int64"] is None and not private["private_times"],
                        kind+" Transformer claims nonexistent private time tensor", issues)
                require(set(output["identities"]) == {"readout"},
                        kind+" Transformer output identity fields differ", issues)
        diagnoses[kind] = dict(
            state=row["state"], calls=len(row["calls"]),
            start_position=row["capture_start_position"], final_position=row["final_state"]["position"],
            output_observations=row["outputs"],
            metadata_sha256=sha(replay_metadata(row)),
            complete_output_identities=[dict(index=c["index"], phase=c["phase"], start=c["start"],
                stop=c["stop"], identities=c["output"]["identities"]) for c in row["calls"]],
            final_private_state_summary={key: row["final_state"][key] for key in
                ("position", "storage", "finite", "detached", "times_int64", "clock_scope")},
            final_private_clock_tensor_count=len(row["final_state"]["private_times"]),
            private_clock_hash_coverage_by_call=[dict(index=call["index"],
                int64_tensors=sum(t["dtype"] == "torch.int64" for t in call["private_state"]["tensors"]),
                named_clock_value_hashes=len(call["private_state"]["private_times"])) for call in row["calls"]],
            final_int64_tensors_without_value_hash=[tensor for tensor in row["final_state"]["tensors"]
                if tensor["dtype"] == "torch.int64" and tensor["path"] not in
                {clock["path"] for clock in row["final_state"]["private_times"]}],
            observation=operator_review(row, issues) if kind == "operators" else stage_review(row, issues))
    first, second = (record["diagnostics"][kind] for kind in args["diagnostics"])
    metadata_identical = replay_metadata(first) == replay_metadata(second)
    differences = []
    require(len(first["calls"]) == len(second["calls"]), "diagnostic call counts differ", issues)
    for left, right in zip(first["calls"], second["calls"]):
        a, b = left["output"]["identities"], right["output"]["identities"]
        changed = {key: dict(reference=a.get(key), current=b.get(key))
                   for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
        if changed:
            differences.append(dict(index=right["index"], phase=right["phase"],
                                    start=right["start"], stop=right["stop"], identities=changed))
    reported = record["replay_checks"]
    require(metadata_identical and reported["independent_diagnostic_metadata_identical"] == metadata_identical,
            "inference support/time/private metadata differs or reported comparison disagrees", issues)
    require(reported["output_identity"]["exact_hashes_identical"] == (not differences) and
            reported["output_identity"]["differing_calls"] == differences,
            "reported inference SHA comparison disagrees with complete identity fields", issues)
    return dict(
        diagnostics=diagnoses, independently_recomputed_metadata_identical=metadata_identical,
        independently_recomputed_output_identity_identical=not differences,
        differing_output_calls=differences, recorded_replay_checks=reported,
        normal_retention="normal samples retain timings, shapes/counts and state-memory metadata, but no per-call output bytes, numerical values or output SHA; availability/first backbone output is filled from a separate diagnostic support witness; no per-value equality to normal samples can be concluded",
        state_scope="named .times fields and positive KV-time/support metadata are observed and compared; SSD ConvState NamedTuple clocks use .conv.0 paths and have only shape/stride/dtype/device metadata, not clock-value hashes; floating KV/SSD/history payload is neither fully hashed nor numerically compared in inference; metadata equality is not full state equality, equality of every private clock value, or an independent write/eviction oracle",
        numerical_scope="two separate intrusive same-candidate replays compare all readout and raw Tide value/strength bytes using complete hashes; hashes do not provide finite error magnitudes or a tolerance/VJP qualification; any differing hashes remain unexplained finite differences",
    )


def main():
    rows, pending = {}, []
    for name in KINDS:
        path = ROOT / "artifacts" / ("breakdown-"+name+"-v1") / "breakdown.json"
        if not path.exists():
            rows[name] = dict(state="pending_raw", path=str(path.relative_to(ROOT)))
            pending.append(name)
            continue
        payload = path.read_bytes()
        record = json.loads(payload)
        row = rows[name] = dict(raw=str(path.relative_to(ROOT)), raw_sha256=hashlib.sha256(payload).hexdigest(),
            raw_bytes=len(payload), raw_state=record["state"], raw_last_stage=record.get("last_stage"),
            source=record.get("source"), observed_physical_devices=record.get("physical_devices"),
            requested_device=record.get("arguments", {}).get("device"), precision_policy=record.get("precision"),
            resource_identity=record.get("resource_identity"), torch=record.get("torch"),
            torch_npu=record.get("torch_npu"), recorded_allow_hf32=record.get("allow_hf32"),
            subject=record.get("subject"), parameters=record.get("parameters"), spec=record.get("spec"),
            input_sha256=record.get("input_sha256"), input_shape=record.get("input_shape"),
            shared_input_prefix=record.get("shared_input_prefix"), observation_issues=[])
        if record["state"] not in {"completed_finite", "failed"}:
            row["state"] = "pending_raw"
            pending.append(name)
            continue
        if record["state"] == "failed":
            row.update(state="raw_failed", error=record.get("error"),
                       failure_kind=record.get("failure_kind"), diagnostics=record.get("diagnostics", {}).keys())
            row["diagnostics"] = list(row["diagnostics"])
            continue
        issues = row["observation_issues"]
        require(all(record["source"][key] == EXPECTED_SOURCE for key in ("source_commit", "harness_commit")),
                "frozen source/harness revision differs from assigned identity", issues)
        row["review"] = (train_review(record, issues) if name.startswith("train-") else infer_review(record, issues))
        row["state"] = "completed_review" if not issues else "review_issues"
    result = dict(
        schema="tide-breakdown-qualification-review-v1", observed=datetime.now(timezone.utc).isoformat(),
        state="pending_raw_records" if pending else "completed_review",
        reviewer_script=str(Path(__file__).resolve().relative_to(ROOT)),
        reviewer_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        expected_source=EXPECTED_SOURCE, pending=pending, records=rows,
        original_fp32_thresholds=dict(atol=1e-6, rtol=2e-3),
        authority="read-only review of frozen raw records; no production source or raw artifact mutation; no device workload launched; resource/process-tree terminal verification belongs to the parent",
        limits=[
            "Different Tide and Transformer functions are not numerically compared to each other.",
            "Stage/operator passes synchronize and perturb scheduling; their durations are not normal wall shares or native kernel times.",
            "Entry waits are excluded from the entering stage/operator; nested pre-entry work remains in the parent's exclusive time.",
            "Only completed worker comparisons are reviewed; durable tensor payload is unavailable for independent value recomputation.",
            "This evidence does not replace independent local formulas/VJPs, whole-candidate qualification or older preserved FP32/BF16 failures.",
            "Medium training is d192/KV32 ratio representative, not d768/KV128 or 9B training qualification.",
            "The fixed32 Transformer is an auxiliary different-function reference; not an A-D joint match or model-quality comparison.",
            "Artifact suffix tf32 denotes the 32-layer Transformer, not TensorFloat-32 arithmetic; the requested precision is FP32.",
        ])
    atomic(DESTINATION, result)
    print(json.dumps(dict(state=result["state"], pending=pending,
        records={key: dict(state=row["state"], raw_last_stage=row.get("raw_last_stage"),
                          observation_issues=row.get("observation_issues")) for key, row in rows.items()})))


if __name__ == "__main__":
    main()
