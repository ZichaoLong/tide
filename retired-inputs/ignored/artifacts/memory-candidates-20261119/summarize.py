#!/usr/bin/env python3
"""Read existing memory-candidate records and atomically refresh summary.json.

No experiment is launched. Missing, running and failed records remain visible.
Optional supplementary raw inference/suite records can be registered in
extra-inference.json as [{"label": "...", "path": "artifacts/.../inference.json",
"scope": "..."}], or supplied with --extra-inference LABEL=PATH.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.job import atomic

CANDIDATES = ("x", "all-ssd", "reverse", "kv384", "kv256", "kv128", "memory-shared")
AD_CANDIDATES = ("x", "all-ssd", "reverse", "kv128", "memory-shared")
INPUTS, PARSE_ERRORS = {}, []
COMPLETE = {"completed", "completed_finite", "completed_with_budget_stop", "completed_after_budget_fallback"}
TERMINAL = COMPLETE | {"failed", "budget_refused", "error", "cancelled", "timeout"}


class Review:
    """Keep independent scalar checks and failures without rewriting raw evidence."""
    def __init__(self):
        self.checked = 0
        self.failures = []

    def check(self, name, condition, **details):
        self.checked += 1
        if not condition:
            self.failures.append(dict(check=name, **details))

    def equal(self, name, actual, expected):
        self.check(name, actual == expected, actual=actual, expected=expected)

    def close(self, name, actual, expected):
        self.check(name, isinstance(actual, (int, float)) and math.isfinite(actual)
                   and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), actual=actual, expected=expected)

    def result(self):
        return dict(state="passed" if not self.failures else "failed", checks=self.checked, failures=self.failures)


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def pick(value, *keys):
    return {key: value[key] for key in keys if key in value}


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT/path
    identity = {"path": rel(path), "state": "missing"}
    data = {}
    try:
        raw = path.read_bytes()
        identity.update(state="available", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise TypeError("expected a JSON object")
    except FileNotFoundError:
        pass
    except Exception as error:
        identity.update(state="read_error", error=repr(error))
        PARSE_ERRORS.append(identity)
        data = {}
    INPUTS[rel(path)] = identity
    return data, identity


def parsed(path, transform):
    data, identity = read(path)
    result = dict(file=identity, state=data.get("state", identity["state"]))
    if data:
        result.update(pick(data, "source", "harness_source", "physical_devices", "resource_identity", "arguments",
                           "precision", "config", "spec", "last_stage", "error", "failure_kind"))
        try:
            result.update(transform(data))
        except Exception as error:
            result["summary_error"] = repr(error)
            PARSE_ERRORS.append(dict(file=identity, error=repr(error)))
    return result


def median(values):
    return statistics.median(values) if values and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) else None


def all_observed(values):
    return all(values) if values else None


def shape_summary(config):
    if not config or "d" not in config:
        return None
    d = config["d"]
    kv = None if config.get("memory") == "S" else config.get("kv_width") or d
    return dict(content_width=d, summary_width=d//4, ffn_width=config.get("ffn_width") or 4*d,
                kv_total_qkv_width=kv, kv_heads=None if kv is None else kv//32, memory=config.get("memory"),
                sharing=config.get("sharing"), memory_sharing=config.get("memory_sharing"),
                adapter_rank=config.get("adapter_rank"), region_batch_size=config.get("region_batch_size"),
                checkpoint_reception=config.get("checkpoint_reception"),
                label_scope="kv384/kv256/kv128 labels name the d768 candidates; d192 representatives actually use96/64/32; Q/K/V total-width narrowing, not GQA")


def memory_peak(memories):
    return {kind: max((m.get("peak_"+kind+"_bytes", 0) for m in memories if m), default=None)
            for kind in ("allocated", "reserved")}


def audit_summary(audit):
    if audit is None:
        return None
    result = pick(audit, "parameters", "definition", "deduplication", "input_positions", "sample_time_slots", "logical_axis",
                  "work", "work_per_input_position", "query_packing", "node_matrices", "node_matrix_dispatch",
                  "matrix_cost_definition", "matrix_dispatch_definition", "retention")
    result["effective_touched"] = {key: value for key, value in audit.get("effective_touched", {}).items() if key != "by_sample_time"}
    result["stages"] = {key: {k: v for k, v in value.items() if k != "by_sample_time"} for key, value in audit.get("stages", {}).items()}
    return result


def checkpoint_summary(checkpoint):
    if not checkpoint:
        return {"state": "not_yet_recorded"}
    result = pick(checkpoint, "state", "boundary", "save_seconds", "load_seconds", "file_bytes", "losses", "update_windows")
    result["comparisons"] = {}
    for name, value in checkpoint.items():
        if isinstance(value, dict) and "finite" in value:
            row = pick(value, "exact", "finite", "within_threshold", "atol", "rtol", "max_abs", "max_scaled_error",
                       "floating_tensors", "metadata_mismatches", "support_mismatches")
            row["numerical_failure_count"] = len(value.get("numerical_failures", []))
            row["numerical_failure_sample"] = value.get("numerical_failures", [])[:8]
            result["comparisons"][name] = row
    result["full_failure_detail"] = "original checkpoint comparison in raw continuous JSON; thresholds unchanged"
    return result


def training(data):
    rows = data.get("observations", [])
    stable = rows[2:20]
    seconds = sum(row["complete_seconds"] for row in stable)
    positions = sum(row["input_positions"] for row in stable)
    rate = positions/seconds if stable and seconds > 0 else None
    args = data.get("arguments", {})
    expected = args.get("batch", 0)*(args.get("prefill", 0)+args.get("decode", 0))
    return dict(parameters=data.get("parameters"), initial_parameters_sha256=data.get("initial_parameters_sha256"),
        input_sha256=data.get("input_sha256"), input_shape=data.get("input_shape"),
        shape=shape_summary(data.get("spec")),
        initialization=dict(model_seed=args.get("seed"), input_seed=None if args.get("seed") is None else args["seed"]+1,
                            boundary="same generated input reused for continuous windows; next private state continues and detaches after each update"),
        timing=data.get("timing"), training_boundary=data.get("training_boundary"), observed_updates=len(rows),
        normal_training=dict(stable_updates_target=[3, 20], stable_observed_updates=[3, 2+len(stable)] if stable else [],
            stable_input_positions=positions, stable_complete_seconds=seconds,
            stable_input_positions_per_second=rate, provisional=len(rows) != 20 or data.get("state") != "completed",
            normal_peak_bytes=memory_peak([row.get("device_memory", {}) for row in rows]),
            last_state_memory=rows[-1].get("state_memory") if rows else None,
            last_state_position=rows[-1].get("state_position") if rows else None,
            last_continuation=rows[-1].get("continuation") if rows else None,
            update_seconds=[row.get("complete_seconds") for row in rows], losses=[row.get("loss") for row in rows]),
        checks=dict(ordered_update_indices=all_observed([row.get("index") == i for i, row in enumerate(rows)]),
            input_denominators=all_observed([row.get("input_positions") == expected for row in rows]),
            output_parameters_gradients_finite=all_observed([all(row.get(k) is True for k in ("finite_output", "finite_parameters", "finite_gradients")) for row in rows]),
            state_optimizer_finite_detached=all_observed([all(row.get("continuation", {}).get(k) is True for k in ("private_state_finite", "private_state_detached", "optimizer_finite")) for row in rows])),
        paired_checkpoint=checkpoint_summary(data.get("checkpoint")), checkpoint_device_memory=data.get("checkpoint_device_memory"),
        recovery_scope="this continuous record contains same-process paired save/load/update; it does not itself establish a fresh-process restore")


def restored(data):
    result = pick(data, "parameters", "checkpoint_source", "checkpoint_file", "checkpoint_stat_before_load", "loaded_state_position",
        "restored_state_position", "loaded_state_memory", "restored_state_memory", "parameter_storage", "gradient_storage",
        "optimizer_before", "optimizer_after", "optimizer_step_increment_errors", "parameter_tensors_with_gradients",
        "parameter_update_sample", "private_state_detached", "loaded_finite", "updated_finite", "load_complete_seconds",
        "next_update_complete_seconds", "input_positions", "next_update_input_positions_per_second", "load_peak_device_bytes",
        "next_update_peak_device_bytes", "phases", "input_sha256", "input_shape", "comparison_qualification")
    result["next_input_seed"] = data.get("arguments", {}).get("seed")
    result["recovery_scope"] = "fresh interpreter loads checkpoint and performs an actual full next update; no original-owner numerical reference in this child"
    return result


def ledger(data):
    rows = []
    for observation in data.get("observations", []):
        phases = {name: pick(value, "categories", "unique_storage_union_bytes", "unique_storage_union_by_device", "allocator",
                            "live_allocator_minus_visible_union_bytes", "unclassified_scope", "ownership_priority")
                  for name, value in observation.get("phases", {}).items()}
        rows.append(dict(segment=observation.get("segment"), finite=observation.get("finite"), state_position=observation.get("state_position"),
            phases=phases, phase_allocator_peak_scope=observation.get("phase_allocator_peak_scope"),
            saved_for_backward=pick(observation.get("saved_for_backward", {}), "pack_calls", "unpack_calls", "saved_payload_event_bytes",
                "peak_live_saved_payload_bytes", "peak_live_unique_storage_bytes", "peak_live_additional_storage_bytes", "current_live_unique_storage_bytes",
                "current_live_unique_storage_count", "additional_scope", "lifetime", "hook_scope"),
            runtime=observation.get("runtime"), actual_work=observation.get("actual_work"), work_measurement=observation.get("work_measurement")))
    return dict(parameters=data.get("parameters"), initial_parameter_sha256=data.get("initial_parameter_sha256"),
        input_sha256=data.get("input_sha256"), segments=rows,
        initialization=dict(model_seed=data.get("arguments", {}).get("seed"),
            input_seed=None if data.get("arguments", {}).get("seed") is None else data["arguments"]["seed"]+1),
        interpretation="separate intrusive two-update storage/operator diagnostic with model seed1107/input1108; normal continuous seed751/input752 uses different parameters/input/activity. Ledger memory is not attribution of the normal run. Same-instant unclassified is not scratch-only; saved/state/category peaks overlap and must not be added to normal peaks")


def lifecycle(run):
    return {name: parsed(run/(name+".json"), lambda data: pick(data, "source_path", "unit", "unit_observation", "state", "status",
                "exit_code", "return_code", "physical_ids", "logical_devices", "started", "finished", "elapsed", "peak_rss_bytes",
                "max_rss_bytes", "descendant_identities")) for name in ("launch", "status", "queue")}


def train_suite(candidate):
    run = ROOT/"artifacts"/f"memory-train-{candidate}-v1"
    suite, identity = read(run/"results/suite.json")
    children = [pick(c, "identity", "state", "phase", "batch", "pid", "outcome", "review_state", "review", "error", "review_error", "output",
                     "result_sha256", "terminal_review", "descendant_identities") for c in suite.get("children", [])]
    return dict(candidate=candidate, lifecycle=lifecycle(run), suite=dict(file=identity, **pick(suite, "state", "source", "physical_devices",
        "selected_training_batch", "decisions", "error"), children=children),
        train=parsed(run/"results/train-b8.json", training), restore=parsed(run/"results/restore-b8.json", restored),
        ledger=parsed(run/"results/ledger-b8.json", ledger),
        fallback_children=[parsed(c["output"], {"train": training, "restore": restored, "ledger": ledger}[c["phase"]])
                           for c in suite.get("children", []) if c.get("batch") != 8 and c.get("phase") in {"train", "restore", "ledger"}])


def inference_metrics(observations):
    result = {"normal_samples": len(observations)}
    for name, getter in (
        ("first_backbone_output_seconds", lambda row: row.get("first_any_backbone_output_seconds")),
        ("prompt_output_ready_seconds", lambda row: row.get("prompt_output_ready_seconds")),
        ("first_incremental_call_seconds", lambda row: row.get("first_incremental_call_seconds")),
        ("prefill_positions_per_second", lambda row: row.get("prefill", {}).get("positions_per_second")),
        ("stable_decode_positions_per_second", lambda row: row.get("stable_decode", {}).get("positions_per_second")),
        ("stable_decode_median_step_seconds", lambda row: row.get("stable_decode", {}).get("median_call_seconds")),
        ("stable_decode_p95_step_seconds", lambda row: row.get("stable_decode", {}).get("p95_call_seconds")),
        ("complete_positions_per_second", lambda row: row.get("complete_positions_per_second"))):
        values = [getter(row) for row in observations]
        result[name] = dict(median=median(values), samples=values)
    devices = {device for row in observations for device in row.get("allocator", {})}
    result["normal_peak_bytes_by_device"] = {device: memory_peak([row.get("allocator", {}).get(device, {}) for row in observations]) for device in sorted(devices)}
    result["final_state_memory"] = [row.get("state_memory") for row in observations]
    result["final_state_positions"] = [row.get("final_state_position") for row in observations]
    return result


def inference_result(data):
    result = pick(data, "schedule", "torch", "torch_threads", "cpu_affinity", "request_boundary", "allocator_boundary", "cold_boundary")
    result["subjects"] = {}
    for name, subject in data.get("subjects", {}).items():
        observations = subject.get("observations", [])
        normal = inference_metrics(observations)
        normal.update(requested_samples=data.get("arguments", {}).get("samples"),
                      provisional=data.get("state") not in COMPLETE or subject.get("state") != "completed" or
                      len(observations) != data.get("arguments", {}).get("samples"))
        result["subjects"][name] = dict(**pick(subject, "state", "error", "spec", "parameters", "placement", "comparison", "initial_parameters",
            "input_sha256", "input_shape", "shared_input_prefix", "diagnostics"),
            normal=normal,
            cold=subject.get("cold"), audit=audit_summary(data.get("matching_reference", {}).get("audit") if name == "tide" else None))
    return result


def inference_suite(path):
    data, identity = read(path)
    if "subjects" in data:
        return parsed(path, inference_result)
    result = dict(file=identity, state=data.get("state", identity["state"]),
        **pick(data, "source", "physical_devices", "arguments", "config", "policy", "decisions", "stop_reason", "error",
               "best_common_or_screen", "best_formal_measured", "all_executed_children_successful"), points=[])
    result["selection_is_provisional"] = data.get("state") not in COMPLETE
    for child in data.get("children", []):
        if child.get("phase") == "inference":
            result["points"].append(dict(child=pick(child, "identity", "batch", "pid", "formal", "warmup", "samples", "state", "outcome", "error",
                "review_state", "review_error", "result_sha256", "terminal_review", "descendant_identities"), result=parsed(child["output"], inference_result)))
    available = {row["child"]["identity"]: index for index, row in enumerate(result["points"])}
    result["selection_indices"] = {key: available.get(value) for key, value in {
        "common_b1": "common-b1", "common_b8": "common-b8", "budget_fallback_b4": "budget-fallback-b4",
        "formal_best": data.get("best_formal_measured"), "best_common_or_screen": data.get("best_common_or_screen")}.items()}
    result["ranking_scope"] = "finite measured candidates only; common, exploratory screening and formal confirmation retain separate identities"
    return result


def ad_review(data):
    """Independent arithmetic and record-identity audit; no tensor replay."""
    review = Review()
    args, topology = data["arguments"], data["topology"]
    modes = ("inference", "grad_forward", "train")
    subjects = ("tide", "transformer_A", "transformer_B", "transformer_C", "transformer_D")
    positions = args["batch"]*args["length"]
    review.equal("completed", data["state"], "completed_finite")
    review.equal("declared_modes", args["modes"], list(modes))
    review.equal("input_position_denominator", data["input_positions"], positions)
    review.equal("mode_subject_coverage", sorted((row["mode"], row["subject"]) for row in data["observations"]),
                 sorted((mode, subject) for mode in modes for subject in subjects))
    matches, identity_groups, activity = [], {}, []
    max_deviation = 0
    for mode in modes:
        audit = data["initial_audits"][mode]
        touched = audit["effective_touched"]
        touches = touched["by_sample_time"]
        review.equal(f"{mode}.touched_slots", len(touches), positions)
        review.equal(f"{mode}.touched_unique_slots", len({(r["sample"], r["time"]) for r in touches}), positions)
        review.equal(f"{mode}.touched_sum", touched["sum"], sum(r["parameters"] for r in touches))
        review.close(f"{mode}.touched_mean", touched["mean"], touched["sum"]/positions)
        total = audit["parameters"]["unique_parameters"]
        for name in "ABCD":
            match = data["joint_matches_by_mode"][mode][name]
            ctor = match["constructor"]
            blocks = topology["nodes"] if name in "AB" else topology["layers"]
            target = total if name in "AC" else touched["mean"]
            d, width = ctor["d"], ctor["ffn_width"]
            count = blocks*(4*d*d+3*d*width+2*d)+d
            deviation = (count-target)/target
            prefix = f"{mode}.{name}"
            review.equal(prefix+".target_blocks", match["targets"]["blocks"], blocks)
            review.equal(prefix+".constructor_blocks", ctor["layers"], blocks)
            review.equal(prefix+".actual_blocks", match["actual"]["blocks"], blocks)
            review.equal(prefix+".target_parameters", match["targets"]["parameters"], target)
            review.equal(prefix+".formula_parameters", match["actual"]["parameters"], count)
            review.close(prefix+".parameter_deviation", match["relative_deviation"]["parameters"], deviation)
            review.close(prefix+".block_deviation", match["relative_deviation"]["blocks"], 0)
            review.check(prefix+".legal_widths", d >= 32 and d % 16 == 0 and width % 8 == 0 and 2*d <= width <= 6*d,
                         d=d, ffn_width=width)
            review.check(prefix+".head_divisibility", ctor["head_dim"] in (16, 32, 64) and d % ctor["head_dim"] == 0,
                         d=d, head_dim=ctor["head_dim"])
            # Enumerate the complete legal FFN grid, independent of the harness's
            # floor/ceiling candidate reduction. Larger d has minimum count above
            # target plus the observed error and cannot be a closer constructor.
            maximum = max(32, 16*math.ceil(math.sqrt((target+abs(count-target))/(10*blocks))/16))
            points = ((abs(n-target)/target, abs(f/w-4), w, f, n)
                      for w in range(32, maximum+1, 16) for f in range(2*w, 6*w+1, 8)
                      for n in (blocks*(4*w*w+3*w*f+2*w)+w,))
            closest = min(points)
            review.equal(prefix+".nearest_legal_parameters", [d, width, count], list(closest[2:]))
            max_deviation = max(max_deviation, abs(deviation))
            matches.append(dict(mode=mode, joint_class=name, blocks=blocks, parameter_target=target,
                                parameters=count, relative_deviation=deviation, constructor=ctor))

    for row in data["observations"]:
        mode, subject = row["mode"], row["subject"]
        prefix = f"{mode}.{subject}"
        times = row["seconds"]
        review.equal(prefix+".completed", row["state"], "completed_finite")
        review.equal(prefix+".sample_count", len(times), args["steps"])
        review.check(prefix+".positive_finite_seconds", bool(times) and all(math.isfinite(t) and t > 0 for t in times))
        measured = statistics.median(times)
        review.close(prefix+".median_seconds", row["median_seconds"], measured)
        review.close(prefix+".input_positions_per_second", row["input_positions_per_second"], positions/measured)
        review.close(prefix+".window_latency_ms", row["window_latency_ms"], 1000*measured)
        review.equal(prefix+".sample_memory_count", len(row["sample_device_memory"]), args["steps"])
        storage = row["actual_storage"]
        review.equal(prefix+".parameter_storage_elements", sum(r["elements"] for r in storage["parameters"].values()), row["parameters"])
        review.equal(prefix+".fp32_parameter_storage_bytes", sum(r["bytes"] for r in storage["parameters"].values()), 4*row["parameters"])
        review.check(prefix+".actual_parameter_dtypes", all(k.endswith(":torch.float32") for k in storage["parameters"]))
        if mode == "train":
            review.check(prefix+".gradient_storage_present", sum(r["elements"] for r in storage["gradients"].values()) > 0)
            review.check(prefix+".optimizer_storage_present", sum(r["elements"] for r in storage["optimizer"].values()) > 0)
            review.check(prefix+".finite_losses", len(row["losses"]) == args["steps"] and all(math.isfinite(v) for v in row["losses"]))
        else:
            review.equal(prefix+".no_parameter_gradients", storage["gradients"], {})
            review.equal(prefix+".no_optimizer", storage["optimizer"], {})
        spec = row["spec"]
        if subject == "tide":
            initial = data["initial_audits"][mode]
            review.equal(prefix+".initial_parameters_identity", row["initial_parameter_sha256"], data["initial_tide_parameter_sha256"])
            review.equal(prefix+".initial_input_identity", row["input_sha256"], data["audit_input_sha256"])
            review.equal(prefix+".total_parameter_target", row["parameters"], initial["parameters"]["unique_parameters"])
            final = row["post_timing_audit"]
            activity.append(dict(mode=mode,
                effective_touched=dict(initial=initial["effective_touched"]["mean"], after_timing=final["effective_touched"]["mean"],
                                       change=final["effective_touched"]["mean"]-initial["effective_touched"]["mean"]),
                work={key: dict(initial=value, after_timing=final["work"].get(key),
                    change=final["work"].get(key, 0)-value) for key, value in initial["work"].items()}))
        else:
            match = data["joint_matches_by_mode"][mode][subject[-1]]
            review.equal(prefix+".frozen_match", spec["joint_match"], match)
            for key, value in match["constructor"].items():
                review.equal(prefix+".constructor."+key, spec[key], value)
            review.equal(prefix+".constructed_parameter_count", row["parameters"], match["actual"]["parameters"])
            review.equal(prefix+".spec_parameter_count", spec["parameter_count"], row["parameters"])
            review.equal(prefix+".actual_heads", spec["heads"], spec["d"]//spec["head_dim"])
        identity_key = json.dumps({key: value for key, value in spec.items() if key != "joint_match"}, sort_keys=True)
        identity_groups.setdefault(identity_key, []).append(row)
    for index, rows in enumerate(identity_groups.values()):
        review.equal(f"constructor{index}.parameter_identity_across_modes", len({r["initial_parameter_sha256"] for r in rows}), 1)
        review.equal(f"constructor{index}.input_identity_across_modes", len({r["input_sha256"] for r in rows}), 1)
    return dict(**review.result(), observation_count=len(data["observations"]), joint_match_count=len(matches),
        maximum_absolute_relative_parameter_deviation=max_deviation, matches=matches, activity_after_timing=activity,
        formula="Transformer parameters = layers*(4*d*d + 3*d*ffn_width + 2*d) + d; complete legal FFN grid independently enumerated",
        training_scope=dict(initial_matching="initial parameters, fixed input and zero state; no update in matching audit",
            optimizer_updates_before_timing=args["warmup"], timed_optimizer_updates=args["steps"],
            state="each call starts from zero state; training parameters continue through warmup and timed updates",
            rematching=False, continuous_training_throughput=False),
        qualification="record arithmetic, constructor matching and identities only; no numerical tensor replay or downstream quality claim")


def ad_result(data):
    observations = []
    for row in data.get("observations", []):
        observations.append(dict(**pick(row, "subject", "mode", "state", "error", "spec", "parameters", "seconds", "median_seconds",
            "input_positions_per_second", "window_latency_ms", "initial_parameter_sha256", "input_sha256", "sample_device_memory",
            "sample_device_memory_scope", "sample_range_over_median", "actual_storage", "numerical_equivalence_qualification"),
            normal_peak_bytes=memory_peak(row.get("sample_device_memory", [])),
            post_timing_audit=audit_summary(row.get("post_timing_audit")), post_timing_runtime_audit=row.get("post_timing_runtime_audit")))
    return dict(timing=data.get("timing"), input_positions=data.get("input_positions"), topology=data.get("topology"),
        initial_tide_parameter_sha256=data.get("initial_tide_parameter_sha256"), audit_input_sha256=data.get("audit_input_sha256"),
        joint_matches_by_mode=data.get("joint_matches_by_mode"), matching_policy=data.get("matching_policy"),
        initial_audits={mode: audit_summary(audit) for mode, audit in data.get("initial_audits", {}).items()}, observations=observations,
        independent_review=ad_review(data),
        interpretation="A/B jointly match node blocks and total/effective parameters; C/D jointly match depth and total/effective parameters; mode-specific constructors and deviations retained; no quality/equivalence claim")


def ad_programme(candidate):
    run = ROOT/"artifacts"/f"memory-ad-{candidate}-v1"
    programme, identity = read(run/"results/programme.json")
    return dict(candidate=candidate, lifecycle=lifecycle(run), file=identity, state=programme.get("state", identity["state"]),
        source=programme.get("source"), children=[dict(lifecycle=pick(child, "identity", "state", "pid", "exit_code", "inner_state", "error", "descendant_identities"),
            result=parsed(child["output"], ad_result)) for child in programme.get("children", [])])


def training_review(result):
    review, rows = Review(), []
    by_b16 = {r["candidate"]: r["result"] for r in result["training_b16"]}
    for suite in result["training_b8"]:
        candidate = suite["candidate"]
        train, restore, memory = suite["train"], suite["restore"], suite["ledger"]
        higher = by_b16[candidate]
        for batch, run in ((8, train), (16, higher)):
            prefix = f"{candidate}.b{batch}"
            review.equal(prefix+".state", run["state"], "completed")
            review.equal(prefix+".updates", run.get("observed_updates"), 20)
            review.equal(prefix+".seed", run.get("initialization", {}).get("model_seed"), 751)
            review.equal(prefix+".batch", run.get("arguments", {}).get("batch"), batch)
            for name, value in run.get("checks", {}).items():
                review.equal(prefix+"."+name, value, True)
            checkpoint = run.get("paired_checkpoint", {})
            review.equal(prefix+".paired_checkpoint", checkpoint.get("state"), "within_threshold")
            review.check(prefix+".checkpoint_comparisons_present", bool(checkpoint.get("comparisons")))
            for name, comparison in checkpoint.get("comparisons", {}).items():
                for flag in ("finite", "within_threshold"):
                    review.equal(prefix+".checkpoint."+name+"."+flag, comparison.get(flag), True)
                for field in ("metadata_mismatches", "support_mismatches"):
                    review.equal(prefix+".checkpoint."+name+"."+field, comparison.get(field), [])
                review.equal(prefix+".checkpoint."+name+".numerical_failure_count", comparison.get("numerical_failure_count"), 0)
            review.equal(prefix+".last_state_position", run.get("normal_training", {}).get("last_state_position"), 2560)
        review.equal(candidate+".same_b8_b16_initial_parameters", train.get("initial_parameters_sha256"), higher.get("initial_parameters_sha256"))
        review.equal(candidate+".same_b8_b16_parameter_count", train.get("parameters"), higher.get("parameters"))
        review.equal(candidate+".restore_state", restore.get("state"), "completed_finite")
        review.equal(candidate+".restore_parameter_count", restore.get("parameters"), train.get("parameters"))
        review.equal(candidate+".restore_loaded_position", restore.get("loaded_state_position"), 2560)
        review.equal(candidate+".restore_updated_position", restore.get("restored_state_position"), 2688)
        review.equal(candidate+".restore_private_state_detached", restore.get("private_state_detached"), True)
        review.equal(candidate+".restore_optimizer_step_increment_errors", restore.get("optimizer_step_increment_errors"), [])
        review.check(candidate+".restore_has_parameter_gradients", restore.get("parameter_tensors_with_gradients", 0) > 0)
        review.check(candidate+".restore_actual_update_sample", restore.get("parameter_update_sample", {}).get("changed_scalars", 0) > 0)
        for phase in ("loaded_finite", "updated_finite"):
            groups = restore.get(phase, {})
            review.check(candidate+".restore."+phase+".present", bool(groups))
            for name, group in groups.items():
                review.equal(candidate+".restore."+phase+"."+name, group.get("finite"), True)
        review.equal(candidate+".ledger_state", memory.get("state"), "completed_finite")
        review.equal(candidate+".ledger_parameters", memory.get("parameters"), train.get("parameters"))
        review.equal(candidate+".ledger_seed", memory.get("initialization", {}).get("model_seed"), 1107)
        review.check(candidate+".ledger_parameter_identity_separate", memory.get("initial_parameter_sha256") != train.get("initial_parameters_sha256"))
        review.check(candidate+".ledger_input_identity_separate", memory.get("input_sha256") != train.get("input_sha256"))
        review.equal(candidate+".ledger_segment_positions", [r["state_position"] for r in memory.get("segments", [])], [128, 256])
        review.equal(candidate+".ledger_segments_finite", [r["finite"] for r in memory.get("segments", [])], [True, True])
        for child in suite["suite"]["children"]:
            key = {"train": "train", "restore": "restore", "ledger": "ledger"}.get(child["phase"])
            if child.get("batch") == 8 and key:
                review.equal(candidate+"."+child["identity"]+".record_hash", child.get("result_sha256"), suite[key]["file"].get("sha256"))
                review.equal(candidate+"."+child["identity"]+".reported_descendants_terminal",
                             child.get("terminal_review", {}).get("all_recorded_descendants_terminal"), True)
        rows.append(dict(candidate=candidate, parameters=train.get("parameters"), shape=train.get("shape"),
            throughput_b8=train.get("normal_training", {}).get("stable_input_positions_per_second"),
            throughput_b16=higher.get("normal_training", {}).get("stable_input_positions_per_second"),
            normal_peak_bytes_b8=train.get("normal_training", {}).get("normal_peak_bytes"),
            normal_peak_bytes_b16=higher.get("normal_training", {}).get("normal_peak_bytes"),
            checkpoint_b8=train.get("paired_checkpoint", {}).get("state"), checkpoint_b16=higher.get("paired_checkpoint", {}).get("state"),
            fresh_process_restore_b8=restore.get("state"), fresh_process_restore_b16="not_measured"))
    return dict(**review.result(), candidates=rows,
        scope="independent record arithmetic/finite/checkpoint checks; original numerical thresholds unchanged, no tensor replay",
        reviewer_corrections=["Initial reviewer expected restore/ledger state='completed'; schema actually uses 'completed_finite'. Corrected reviewer expectation only; fourteen initially reported schema failures were not experiment/numerical failures, and raw records are unchanged."],
        ledger_boundary="seed1107/input1108 two-update intrusive ledgers have different parameters/input/activity from seed751/input752 normal training; do not attribute normal timing or peaks to them",
        process_boundary="B8 fresh-process restored actual update is separate from B8/B16 paired same-process checkpoint tolerance checks; current live descendant audit belongs to the final lifecycle review")


def process_budget(result):
    plan, identity = read(HERE/"plan.json")
    cap = plan.get("bounds", {}).get("max_npu_benchmark_processes", 96)
    rows, measured_refusals, projected_refusals = [], [], []

    def add(group, label, started, allowance, state, **details):
        rows.append(dict(group=group, label=label, workload_starts=started,
                         remaining_upper_bound=max(0, allowance-started), state=state, **details))

    def direct(run, output):
        data, file_id = read(output)
        queue, queue_id = read(run/"queue.json")
        if data:
            evidence = "workload_result_written"
        elif queue.get("status") in {"running", "completed", "cancelled"} or (queue.get("status") == "failed" and "return_code" in queue):
            evidence = "queue_dispatch_committed_before_child_spawn; conservatively counted without_result"
        else:
            evidence = "no_workload_start_evidence; outer dependency/supervisor is excluded"
        return int(bool(data) or evidence.startswith("queue_dispatch")), dict(file=file_id, queue=queue_id,
                    state=data.get("state", queue.get("status", "not_started")), evidence=evidence)

    def children(group, label, records, state, allowance, suite_file):
        starts = sum(isinstance(r.get("pid"), int) for r in records)
        add(group, label, starts, starts if state in TERMINAL else allowance, state, file=suite_file,
            evidence="one recorded child PID is one workload; compiler/helper descendants excluded")
        for child in records:
            if child.get("outcome") == "budget_refused":
                measured_refusals.append(dict(group=group, candidate=label, **pick(child, "identity", "batch", "state", "outcome", "pid", "error")))

    def forecasts(label, decisions):
        for decision in decisions:
            for prediction in decision.get("predictions", []):
                if prediction.get("state") == "projected_budget_refusal":
                    projected_refusals.append(dict(candidate=label,
                        decision=pick(decision, "kind", "measurement_opportunity", "selected_batch"), **prediction))
            if decision.get("state") == "projected_budget_refusal":
                projected_refusals.append(dict(candidate=label, **decision))

    for suite in result["training_b8"]:
        value = suite["suite"]
        children("training_b8", suite["candidate"], value["children"], value.get("state"), 3, value["file"])
    for suite in result["training_b16"]:
        run = ROOT/"artifacts"/f"memory-train-{suite['candidate']}-b16-v1"
        started, details = direct(run, run/"continuous.json")
        add("training_b16", suite["candidate"], started, 1, details.pop("state"), **details)
    for programme in result["joint_comparisons"]:
        children("joint_comparisons", programme["candidate"], [r["lifecycle"] for r in programme["children"]],
                 programme["state"], 1, programme["file"])
    run = ROOT/"artifacts/memory-width768-fp32-v1"
    started, details = direct(run, run/"formulas.json")
    add("actual_width_formula", "d768 widths384/256/128", started, 1, details.pop("state"), **details)
    for suite in result["inference"]:
        value = suite["suite"]
        children("main_inference", suite["candidate"], [r["child"] for r in value["points"]], value["state"], 6, value["file"])
        forecasts(suite["candidate"], value.get("decisions", []))
    supplementary = []
    seen = set()
    for extra in result["supplementary_inference"]:
        value = extra["result"]
        if "points" in value:
            started = sum(isinstance(r["child"].get("pid"), int) for r in value["points"])
            details = dict(state=value["state"], file=value["file"], evidence="recorded child PIDs")
            for point in value["points"]:
                child = point["child"]
                if child.get("outcome") == "budget_refused":
                    measured_refusals.append(dict(group="supplementary_inference", candidate=extra["label"], **child))
            forecasts(extra["label"], value.get("decisions", []))
        else:
            path = ROOT/extra["path"]
            started, details = direct(path.parent, path)
        if value["file"]["path"] in seen:
            continue
        seen.add(value["file"]["path"])
        supplementary.append(dict(label=extra["label"], workload_starts=started, **details))
    for name in ("x", "ssd", "kv128", "shared"):
        run = ROOT/"artifacts"/f"memory-cost-{name}-v1"
        started, details = direct(run, run/"costs.json")
        supplementary.append(dict(label=f"cost-{name}", workload_starts=started, **details))
    supplementary_starts = sum(row["workload_starts"] for row in supplementary)
    # Four near-total B1/B8, one region-batch1 B8, two sustained decode and
    # four cost workloads. Unregistered/missing points remain reserved.
    add("supplementary", "near4 + region1 + sustained2 + costs4", supplementary_starts, 11, "finite_reserved", points=supplementary)
    for name in ("x", "kv128"):
        run = ROOT/"artifacts"/f"memory-decode-cost-{name}-v1"
        launch, launch_id = read(run/"launch.json")
        command = launch.get("command", [])
        output = run/"costs.json"
        for index, argument in enumerate(command[:-1]):
            if argument == "--output":
                output = Path(command[index+1])
                if not output.is_absolute():
                    output = Path(launch.get("source_path", run/"source"))/output
        started, details = direct(run, output)
        add("decode_cost_diagnostics", name, started, 1, details.pop("state"), launch=launch_id,
            source=launch.get("source"), output_resolution="launch command --output, otherwise pending costs.json placeholder",
            planned=dict(batch=8, prefill=128, prefill_chunk=32, decode=16, diagnostics=["stages", "runtime"],
                         workload_processes=1, timeout_seconds=900, allocated_gib=48, reserved_gib=56,
                         scope="intrusive actual-history decode cost evidence; one pass per diagnostic, no search"), **details)
    observed = sum(row["workload_starts"] for row in rows)
    remaining = sum(row["remaining_upper_bound"] for row in rows)
    return dict(plan=identity, limit=cap, workload_starts=observed, remaining_finite_upper_bound=remaining,
        maximum_final_workloads=observed+remaining, margin=cap-observed-remaining,
        within_limit=observed+remaining <= cap and supplementary_starts <= 11, rows=rows,
        actual_budget_refusals=measured_refusals, predicted_budget_refusals=projected_refusals,
        original_worst_case=dict(training=28, joint_comparisons=5, actual_width_formula=1, main_inference=42, supplementary=11, total=87),
        authorized_extension=dict(decode_cost_diagnostics=2, original_plus_extension_worst_case=89,
            scope="X and KV128 actual-history decode diagnostics; current upper bound releases unused slots of terminal main suites"),
        definitions=["One actual workload process consumes one point, including unsuccessful measured budget attempts. A suite/programme/queue supervisor and compiler helpers do not add workload points.",
            "Child PIDs and output records establish starts; a direct queue dispatch without output is conservatively charged once. An outer dependency-waiting service alone is zero starts.",
            "A projected budget refusal does not start a workload. It remains separate from a measured refusal and is never a measured throughput point.",
            "Completed/failed main suites reserve zero future children; running or missing main suites retain up to six total children. No retries or extra screens implied.",
            "Two separately reserved decode-cost workloads are identified by memory-decode-cost-x-v1 and memory-decode-cost-kv128-v1; future launch commands resolve their actual output paths. Each is charged once, not once per intrusive diagnostic pass.",
            "Small precision qualification/resume processes use the separate40 bound and are excluded here; the explicit d768 formula process is included once."])


def old_tf32():
    evidence, identity = read(ROOT/"docs/evidence/decode-capacity-20261118.json")
    if not evidence:
        return dict(file=identity, state=identity["state"])
    winners = [row for row in evidence["confirmed_winners"] if row["series"] == "9b/tf32/1"]
    selected = {row["name"] for row in winners} | {"cap-9b-tf32-c1-b8-screen-v1"}
    records = [row for row in evidence["records"]["summary"]["runs"] if row["name"] in selected]
    records += [row for row in evidence["historical_batch_1"] if row["series"] == "9b/tf32/1"]
    points = []
    for row in records:
        data = row["result"]
        points.append(dict(name=row["name"], origin=row["origin"], phase=row["phase"], batch=row["batch"], files=row["files"],
            state=data.get("state"), **pick(data, "source", "physical_devices", "spec", "parameters", "initial_parameters", "input_sha256",
                "input_shape", "shared_input_prefix", "schedule", "sampling", "precision", "comparison"),
            normal=inference_metrics(data.get("observations", []))))
    return dict(file=identity, state="reused_existing_evidence", model_shapes=[row for row in evidence["model_shapes"] if row["series"] == "9b/tf32/1"],
        points=points, selected_budget_best_names=[row["name"] for row in winners],
        interpretation="unchanged common 32-layer near9B auxiliary reference, not a substitute for A-D; old source/path/card/sample identities retained; B8 is a two-sample screen, B1/B40 have five samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"summary.json")
    parser.add_argument("--extra-inference", action="append", default=[], metavar="LABEL=PATH")
    args = parser.parse_args()
    INPUTS.clear(); PARSE_ERRORS.clear()
    extra_path = HERE/"extra-inference.json"
    extras = []
    if extra_path.exists():
        raw = extra_path.read_bytes()
        extras = json.loads(raw)
        INPUTS[rel(extra_path)] = dict(path=rel(extra_path), state="available", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    for value in args.extra_inference:
        label, path = value.split("=", 1)
        extras.append(dict(label=label, path=path, scope="explicit supplementary inference input"))
    raw_script = Path(__file__).read_bytes()
    result = dict(schema="tide-memory-candidates-summary-v1", generated_at=datetime.now(timezone.utc).isoformat(),
        generator=dict(path=rel(__file__), sha256=hashlib.sha256(raw_script).hexdigest(), bytes=len(raw_script), python=sys.version),
        scope="read-only summary snapshot; missing/running/failed evidence is retained; no launch, source rewrite or relabelled qualification",
        units=dict(memory="bytes, divided by 2**30 only when explicitly labelled GiB", throughput="independent input positions/s", latency="seconds unless field explicitly says ms"),
        boundaries=["Normal continuous training summary targets updates3-20 and is provisional until twenty updates and checkpoint finish.",
            "B16 continuous command includes paired checkpoint comparison; fresh-process restore and intrusive storage qualification remain separate B8 records.",
            "Cold, normal samples, work audit and intrusive ledger have separate boundaries. Do not sum category/phase peaks or infer throughput from retained cache size.",
            "First backbone output excludes task heads and is not complete language-model TTFT. Engine performance is not downstream model quality.",
            "A-D matching is mode-specific and mathematical functions differ; the old fixed32 reference retains old source/hardware identity."],
        training_b8=[train_suite(candidate) for candidate in CANDIDATES],
        training_b16=[dict(candidate=candidate, lifecycle=lifecycle(ROOT/"artifacts"/f"memory-train-{candidate}-b16-v1"),
            result=parsed(ROOT/"artifacts"/f"memory-train-{candidate}-b16-v1/continuous.json", training)) for candidate in CANDIDATES],
        inference=[dict(candidate=candidate, lifecycle=lifecycle(ROOT/"artifacts"/f"memory-infer-{candidate}-v1"),
            suite=inference_suite(ROOT/"artifacts"/f"memory-infer-{candidate}-v1/results/suite.json")) for candidate in CANDIDATES],
        supplementary_inference=[dict(**row, result=inference_suite(Path(row["path"]) if Path(row["path"]).is_absolute() else ROOT/row["path"])) for row in extras],
        joint_comparisons=[ad_programme(candidate) for candidate in AD_CANDIDATES], fixed_transformer32=old_tf32())
    result["training_review"] = training_review(result)
    result["workload_process_budget"] = process_budget(result)
    result["joint_comparison_review"] = dict(
        completed_candidates=sum(row["state"] in COMPLETE for row in result["joint_comparisons"]),
        reviews=[dict(candidate=row["candidate"], file=child["result"]["file"],
                      **child["result"].get("independent_review", {"state": "not_available"}))
                 for row in result["joint_comparisons"] for child in row["children"]])
    result["inputs"] = list(INPUTS.values())
    result["summary_errors"] = PARSE_ERRORS
    result["file_availability"] = dict(Counter(row["state"] for row in INPUTS.values()))
    result["observed_states"] = dict(training_b8={row["candidate"]: row["suite"].get("state", row["suite"]["file"]["state"]) for row in result["training_b8"]},
        training_b16={row["candidate"]: row["result"]["state"] for row in result["training_b16"]},
        inference={row["candidate"]: row["suite"]["state"] for row in result["inference"]},
        joint_comparisons={row["candidate"]: row["state"] for row in result["joint_comparisons"]})
    result["state"] = "summary_errors" if PARSE_ERRORS else "snapshot"
    atomic(args.output, result)
    print(json.dumps(dict(output=str(args.output), state=result["state"], file_availability=result["file_availability"],
                          summary_errors=len(PARSE_ERRORS), observed_states=result["observed_states"],
                          training_review=result["training_review"]["state"],
                          ad_reviews={r["candidate"]: r["state"] for r in result["joint_comparison_review"]["reviews"]},
                          process_budget=pick(result["workload_process_budget"], "workload_starts", "maximum_final_workloads", "margin", "within_limit"))))
    return int(bool(PARSE_ERRORS))


if __name__ == "__main__":
    raise SystemExit(main())
