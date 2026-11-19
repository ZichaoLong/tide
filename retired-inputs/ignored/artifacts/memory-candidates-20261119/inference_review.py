#!/usr/bin/env python3
"""Independent read-only arithmetic/identity review; CPU input hash reconstruction.

Run with TORCH_DEVICE_BACKEND_AUTOLOAD=0 and the recorded private Python. No
model, NPU initialization, benchmark helper import or experiment is performed.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
    raise RuntimeError("TORCH_DEVICE_BACKEND_AUTOLOAD=0 is required before importing torch")
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.job import atomic

CANDIDATES = ("x", "all-ssd", "reverse", "kv384", "kv256", "kv128", "memory-shared")
INPUTS, INPUT_CACHE = {}, {}


def load(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT/path
    key = str(path.relative_to(ROOT))
    identity = dict(path=key, state="missing")
    try:
        raw = path.read_bytes()
        identity.update(state="available", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        data = json.loads(raw)
    except FileNotFoundError:
        data = {}
    INPUTS[key] = identity
    return data, identity


class Review:
    def __init__(self):
        self.checks, self.failures = 0, []

    def check(self, name, condition, **details):
        self.checks += 1
        if not condition:
            self.failures.append(dict(check=name, **details))

    def equal(self, name, actual, expected):
        self.check(name, actual == expected, actual=actual, expected=expected)

    def close(self, name, actual, expected):
        self.check(name, isinstance(actual, (int, float)) and math.isfinite(actual)
                   and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), actual=actual, expected=expected)

    def result(self):
        return dict(state="failed" if self.failures else "passed", checks=self.checks, failures=self.failures)


def independent_input(batch, length, width, seed):
    key = (batch, length, width, seed)
    if key not in INPUT_CACHE:
        # Independent implementation of the declared feature-wise seeded input
        # contract; no call to benchmarks.workloads.paired_input/tensor_hash.
        values = torch.empty(batch, length, width, dtype=torch.float32, device="cpu")
        for feature in range(width):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed+feature)
            values[:, :, feature] = torch.randn((batch, length), generator=generator,
                                                dtype=torch.float32, device="cpu")
        array = values.numpy()
        sample_hashes = [hashlib.sha256(array[index].tobytes(order="C")).hexdigest() for index in range(batch)]
        INPUT_CACHE[key] = dict(batch=batch, positions=length, width=width, seed=seed,
            dtype=str(values.dtype), device=str(values.device), sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            independent_sample_sha256=sample_hashes, distinct_sample_count=len(set(sample_hashes)))
    return INPUT_CACHE[key]


def expected_calls(schedule):
    prompt, chunk = schedule["prefill"], schedule["prefill_chunk"]
    rows = [("prefill", start, min(prompt, start+chunk)) for start in range(0, prompt, chunk)]
    end = prompt+schedule["decode_steps"]
    rows += [("decode", start, start+1) for start in range(prompt, end)]
    rows += [("stream", start, min(end+schedule["stream_positions"], start+schedule["stream_chunk"]))
             for start in range(end, end+schedule["stream_positions"], schedule["stream_chunk"])]
    return rows


def review_request(row, label, data, diagnostic, review):
    args, schedule = data["arguments"], data["schedule"]
    batch = args["batch"]
    calls = row["calls"]
    expected = expected_calls(schedule)
    positions = schedule["prefill"]+schedule["decode_steps"]+schedule["stream_positions"]
    review.equal(label+".call_schedule", [(r["phase"], r["start"], r["stop"]) for r in calls], expected)
    review.equal(label+".input_positions", row["input_positions"], batch*positions)
    review.equal(label+".output_slots", row["output_slots"], sum(r["output_slots"] for r in calls))
    review.equal(label+".final_state_position", row["final_state_position"], positions)
    review.equal(label+".diagnostic_checks_excluded", row["diagnostic_checks_in_timing"], False)
    review.check(label+".positive_complete_time", math.isfinite(row["complete_seconds"]) and row["complete_seconds"] > 0)
    review.check(label+".complete_encloses_calls", row["complete_seconds"] >= calls[-1]["ready_seconds"])
    review.close(label+".complete_positions_per_second", row["complete_positions_per_second"], batch*positions/row["complete_seconds"])
    previous_ready, previous_position = 0, 0
    for index, call in enumerate(calls):
        prefix = f"{label}.call{index}"
        review.equal(prefix+".independent_input_positions", call["input_positions"], batch*(call["stop"]-call["start"]))
        review.equal(prefix+".state_continuation", call["state_before"], previous_position)
        review.equal(prefix+".state_after", call["state_after"], call["stop"])
        review.equal(prefix+".settle_output_slots", call["output_slots"], batch*(call["stop"]-call["start"]))
        review.check(prefix+".positive_seconds", math.isfinite(call["seconds"]) and call["seconds"] > 0)
        review.close(prefix+".ready_boundary", call["ready_seconds"]-previous_ready, call["seconds"])
        review.check(prefix+".forward_inside_call", 0 < call["forward_readout_sync_seconds"] <= call["seconds"])
        previous_ready, previous_position = call["ready_seconds"], call["state_after"]
    for name, phase, drop in (("prefill", "prefill", 0), ("decode", "decode", 0),
                              ("stable_decode", "decode", 1), ("chunked_stream", "stream", 0)):
        selected = [r for r in calls if r["phase"] == phase][drop:]
        summary = row[name]
        durations = [r["seconds"] for r in selected]
        elapsed = sum(durations)
        denominator = sum(batch*(r["stop"]-r["start"]) for r in selected)
        review.equal(label+"."+name+".calls", summary["calls"], len(selected))
        review.equal(label+"."+name+".input_positions", summary["input_positions"], denominator)
        review.close(label+"."+name+".seconds", summary["seconds"], elapsed)
        if selected:
            review.close(label+"."+name+".throughput", summary["positions_per_second"], denominator/elapsed)
            review.close(label+"."+name+".median", summary["median_call_seconds"], statistics.median(durations))
            review.close(label+"."+name+".p95", summary["p95_call_seconds"], sorted(durations)[math.ceil(.95*len(durations))-1])
        else:
            for key in ("positions_per_second", "median_call_seconds", "p95_call_seconds"):
                review.equal(label+"."+name+"."+key, summary[key], None)
    prompt_ready = next(r["ready_seconds"] for r in calls if r["stop"] == schedule["prefill"])
    review.close(label+".prompt_ingested_seconds", row["prompt_ingested_seconds"], prompt_ready)
    review.close(label+".settle_prompt_output_ready_seconds", row["prompt_output_ready_seconds"], prompt_ready)
    first_decode = next((r["seconds"] for r in calls if r["phase"] == "decode"), None)
    review.equal(label+".first_incremental_call_seconds", row["first_incremental_call_seconds"], first_decode)
    flags = diagnostic["output_available_by_call"]
    review.equal(label+".availability_schedule_length", len(flags), len(calls))
    first = next((r["ready_seconds"] for r, available in zip(calls, flags) if available), None)
    review.equal(label+".first_backbone_output_availability", row["first_any_backbone_output_seconds"], first)
    review.close(label+".input_window_seconds", row["input_window_seconds"], calls[-1]["ready_seconds"])
    review.equal(label+".settle_no_drain", row["drain_seconds"], 0)
    review.equal(label+".state_dtypes", row["state_memory"]["dtypes"], ["torch.float32", "torch.int64"])
    for device, memory in row["allocator"].items():
        for kind in ("allocated", "reserved"):
            peak = memory[f"peak_{kind}_bytes"]
            current = memory[f"{kind}_bytes"]
            budget = args[f"{kind}_budget_gib"]*2**30
            review.check(f"{label}.{device}.{kind}_budget", peak <= budget, peak_bytes=peak, budget_bytes=budget)
            review.check(f"{label}.{device}.{kind}_current_le_peak", 0 < current <= peak)
    return dict(first_backbone_output_seconds=first, prompt_output_ready_seconds=prompt_ready,
        prefill_positions_per_second=row["prefill"]["positions_per_second"],
        stable_decode_positions_per_second=row["stable_decode"]["positions_per_second"],
        complete_positions_per_second=row["complete_positions_per_second"],
        independent_complete_sequences_per_second=batch/row["complete_seconds"],
        stable_decode_input_positions=batch*max(0, schedule["decode_steps"]-1),
        request_input_positions=batch*positions, allocator=row["allocator"], state_memory=row["state_memory"])


def summarize_requests(rows):
    if not rows:
        return dict(samples=0)
    keys = ("first_backbone_output_seconds", "prompt_output_ready_seconds", "prefill_positions_per_second",
            "stable_decode_positions_per_second", "complete_positions_per_second", "independent_complete_sequences_per_second")
    return dict(samples=len(rows), **{key: statistics.median(r[key] for r in rows) for key in keys},
        allocator_peak_bytes={kind: max(m[f"peak_{kind}_bytes"] for r in rows for m in r["allocator"].values())
                              for kind in ("allocated", "reserved")})


def review_point(item):
    data, identity = load(item["path"])
    result = dict(label=item["label"], series=item["series"], file=identity,
                  raw_state=data.get("state", identity["state"]), scope=item.get("scope"))
    if data.get("state") != "completed":
        result.update(state="failed" if data.get("state") == "failed" else "pending",
                      last_stage=data.get("last_stage"), error=data.get("error"), source=data.get("source"))
        return result, data
    review = Review()
    args, schedule, subject = data["arguments"], data["schedule"], data["subjects"]["tide"]
    config, diagnostic = subject["spec"], subject["diagnostics"]
    length = schedule["prefill"]+schedule["decode_steps"]+schedule["stream_positions"]
    width = config.get("input_dim") or config["d"]
    reconstructed = independent_input(args["batch"], length, width, args["seed"]+1)
    review.equal("subject_completed", subject["state"], "completed")
    review.equal("only_tide_subject", list(data["subjects"]), ["tide"])
    review.equal("settle_timing", config["topology"]["timing"], "settle")
    review.equal("input_shape", subject["input_shape"], [args["batch"], length, width])
    review.equal("independent_cpu_input_sha256", subject["input_sha256"], reconstructed["sha256"])
    review.equal("independent_samples_not_broadcast", reconstructed["distinct_sample_count"], args["batch"])
    review.equal("shared_prefix_width", subject["shared_input_prefix"]["width"], width)
    review.equal("shared_prefix_sha256", subject["shared_input_prefix"]["sha256"], reconstructed["sha256"])
    review.equal("parameter_identity_finite", subject["initial_parameters"]["finite"], True)
    review.equal("declared_fp32", args["precision"], "fp32")
    review.equal("precision_name", data["precision"]["name"], "fp32")
    for key in ("parameters", "input", "matrix_arithmetic", "normalization_mass_allocation", "kv_keys_values", "kv_summary_and_ssd_state"):
        review.equal("precision_policy."+key, data["precision"][key], "float32")
    review.equal("precision_policy.logical_time_indices", data["precision"]["logical_time_indices"], "int64")
    for key in ("finite_outputs", "finite_private_state", "private_state_detached"):
        review.equal("diagnostic."+key, diagnostic[key], True)
    for key in ("readout_dtypes", "returned_value_dtypes", "returned_strength_dtypes"):
        review.equal("actual_diagnostic."+key, diagnostic[key], ["torch.float32"])
    review.equal("actual_state_dtypes", diagnostic["state_memory"]["dtypes"], ["torch.float32", "torch.int64"])
    review.equal("diagnostic_final_state_position", diagnostic["final_state_position"], length)
    review.equal("observed_samples", len(subject["observations"]), args["samples"])
    child = item.get("child")
    if child:
        for key in ("batch", "warmup", "samples"):
            review.equal("child."+key, args[key], child[key])
        expected_warmup, expected_samples = (2, 5) if child["formal"] else (1, 2)
        review.equal("policy_warmup", args["warmup"], expected_warmup)
        review.equal("policy_samples", args["samples"], expected_samples)
        review.equal("child_result_sha256", identity["sha256"], child["result_sha256"])
        review.equal("child_completed", child["outcome"], "completed")
    else:
        expected_warmup, expected_samples = (1, 3) if schedule["decode_steps"] == 128 else (2, 5)
        review.equal("supplementary_warmup", args["warmup"], expected_warmup)
        review.equal("supplementary_samples", args["samples"], expected_samples)
    review.equal("schedule_prompt", schedule["prefill"], args["prefill"])
    review.equal("schedule_chunk", schedule["prefill_chunk"], args["prefill_chunk"])
    review.equal("schedule_decode", schedule["decode_steps"], args["decode"])
    review.equal("cold_requested", args["cold"], True)
    review.equal("cold_first_subject", subject["cold"]["preceding_subjects_in_process"], [])
    normal = [review_request(row, f"normal{i}", data, diagnostic, review) for i, row in enumerate(subject["observations"])]
    cold = review_request(subject["cold"], "cold", data, diagnostic, review)
    reference = data["matching_reference"]
    review.equal("audit_parameter_identity", reference["tide_parameters"], subject["initial_parameters"])
    review.equal("audit_input_identity", reference["tide_input_sha256"], reconstructed["sha256"])
    review.equal("audit_diagnostic_identity", reference["diagnostics"], diagnostic)
    review.equal("audit_config_identity", reference["config"], config)
    review.equal("audit_input_positions", reference["audit"]["input_positions"], args["batch"]*length)
    review.equal("audit_parameter_count", reference["audit"]["parameters"]["unique_parameters"], subject["parameters"])
    result.update(review.result(), source=data["source"], physical_devices=data["physical_devices"],
        resource_identity=data["resource_identity"], arguments=args, config=config,
        parameters=subject["parameters"], initial_parameter_sha256=subject["initial_parameters"]["sha256"],
        input_sha256=subject["input_sha256"], independent_input=reconstructed,
        diagnostic={key: diagnostic[key] for key in ("finite_outputs", "finite_private_state", "private_state_detached",
                    "readout_dtypes", "returned_value_dtypes", "returned_strength_dtypes", "state_memory",
                    "output_support_sha256", "original_output_times_sha256", "output_available_by_call")},
        normal=summarize_requests(normal), cold=cold,
        boundaries={key: data[key] for key in ("request_boundary", "allocator_boundary", "cold_boundary")})
    return result, data


def work_summary(data):
    audit = data["matching_reference"]["audit"]
    dispatch = audit["node_matrix_dispatch"]
    matrices = audit["node_matrices"]
    return dict(work=audit["work"], query_packing=audit["query_packing"],
        effective_touched={key: value for key, value in audit["effective_touched"].items() if key != "by_sample_time"},
        node_matrices=matrices, matrix_dispatch=dispatch,
        dispatch_totals=dict(base_mm_calls=sum(r["calls"] for r in dispatch if r["base_operator"] == "mm"),
            base_bmm_calls=sum(r["calls"] for r in dispatch if r["base_operator"] == "bmm"),
            repeated_base_weight_rows=sum(r["repeated_base_weight_rows"] for r in dispatch),
            node_matrix_multiply_adds=sum(r["dense_multiply_adds"] for r in matrices.values()),
            ffn_multiply_adds=sum(r["dense_multiply_adds"] for name, r in matrices.items() if ".ffn." in name)))


def shared_comparison(points, raw):
    original = next((r for r in points if r["label"] == "memory-shared/common-b8"), None)
    alternative = next((r for r in points if "region_batch_size1" in r["label"]), None)
    if original is None or alternative is None or original["state"] != "passed" or alternative["state"] != "passed":
        return dict(state="pending", region4_state=None if original is None else original["state"],
                    region1_state=None if alternative is None else alternative["state"])
    review = Review()
    for key in ("parameters", "initial_parameter_sha256", "input_sha256"):
        review.equal(key, alternative[key], original[key])
    left, right = dict(original["config"]), dict(alternative["config"])
    review.equal("region4", left.pop("region_batch_size"), 4)
    review.equal("region1", right.pop("region_batch_size"), 1)
    review.equal("same_config_except_region_batch_size", left, right)
    a, b = original["diagnostic"], alternative["diagnostic"]
    first, second = work_summary(raw[original["label"]]), work_summary(raw[alternative["label"]])
    differences = {key: dict(region4=value, region1=second["work"].get(key),
                    delta=second["work"].get(key, 0)-value) for key, value in first["work"].items()}
    return dict(**review.result(), source_files=[original["file"], alternative["file"]],
        parameter_sha256=original["initial_parameter_sha256"], input_sha256=original["input_sha256"],
        physical_devices=dict(region4=original["physical_devices"], region1=alternative["physical_devices"]),
        output_support_equal=a["output_support_sha256"] == b["output_support_sha256"],
        output_support_sha256=dict(region4=a["output_support_sha256"], region1=b["output_support_sha256"]),
        original_times_equal=a["original_output_times_sha256"] == b["original_output_times_sha256"],
        availability_flags_equal=a["output_available_by_call"] == b["output_available_by_call"],
        actual_work_equal=first["work"] == second["work"], actual_work_differences=differences,
        region4=first, region1=second,
        normal=dict(region4=original["normal"], region1=alternative["normal"]),
        strict_numerical_equivalence="not established by these performance records; full output values and VJPs were not retained for paired comparison",
        attribution="Same parameters/input and execution grouping change. Support hashes and actual work are compared, not assumed equal. Any activity difference remains part of the measured speed comparison; do not label it strict equal-trajectory acceleration.")


def main():
    entries, suites, suites_review = [], [], Review()
    for candidate in CANDIDATES:
        data, identity = load(ROOT/"artifacts"/f"memory-infer-{candidate}-v1/results/suite.json")
        suites_review.equal(candidate+".completed", data.get("state"), "completed")
        measured = [r for r in data.get("children", []) if r.get("outcome") == "budget_refused"]
        suites_review.equal(candidate+".measured_budget_refusals", len(measured), 0)
        predictions = [dict(opportunity=d.get("measurement_opportunity"), selected_batch=d.get("selected_batch"), **p)
                       for d in data.get("decisions", []) for p in d.get("predictions", []) if p["state"] == "projected_budget_refusal"]
        suites.append(dict(candidate=candidate, file=identity, state=data.get("state"),
            measured_budget_refusals=measured, predicted_budget_refusals=predictions,
            best_formal_measured=data.get("best_formal_measured"), best_common_or_screen=data.get("best_common_or_screen")))
        for child in data.get("children", []):
            entries.append(dict(label=candidate+"/"+child["identity"], series="main", path=child["output"], child=child))
    suites_review.equal("main_measured_points", len(entries), 28)
    extras, extra_id = load(HERE/"extra-inference.json")
    for extra in extras:
        entries.append(dict(**extra, series="supplementary"))
    points, raw = [], {}
    for entry in entries:
        try:
            reviewed, data = review_point(entry)
            points.append(reviewed)
            raw[entry["label"]] = data
        except Exception as error:
            points.append(dict(label=entry["label"], series=entry["series"], state="review_error", error=repr(error), path=entry["path"]))
    groups = {}
    for point in points:
        if point["state"] != "passed":
            continue
        key = json.dumps(dict(config=point["config"], seed=point["arguments"]["seed"],
                              source=point["source"]["tide_content_sha256"]), sort_keys=True)
        groups.setdefault(key, []).append(point)
    identity_review, identities = Review(), []
    for key, members in groups.items():
        parameter_hashes = sorted({m["initial_parameter_sha256"] for m in members})
        parameter_counts = sorted({m["parameters"] for m in members})
        name = hashlib.sha256(key.encode()).hexdigest()
        identity_review.equal(name+".same_initial_parameters_across_B_and_workload", len(parameter_hashes), 1)
        identity_review.equal(name+".same_constructor_parameter_count", len(parameter_counts), 1)
        identities.append(dict(configuration_identity=name, points=[m["label"] for m in members],
            batches=sorted({m["arguments"]["batch"] for m in members}), parameter_counts=parameter_counts,
            initial_parameter_sha256=parameter_hashes))
    comparison = shared_comparison(points, raw)
    failures = [dict(label=p["label"], state=p["state"], failures=p.get("failures"), error=p.get("error"))
                for p in points if p["state"] in {"failed", "review_error"}]
    for label, review in (("suites", suites_review.result()), ("parameter_identity", identity_review.result()), ("shared_region_comparison", comparison)):
        if review["state"] == "failed":
            failures.append(dict(label=label, failures=review["failures"]))
    pending = [p["label"] for p in points if p["state"] == "pending"]
    result = dict(schema="tide-memory-inference-independent-review-v1", observed_at=datetime.now(timezone.utc).isoformat(),
        state="failed" if failures else ("pending" if pending else "passed"), failures=failures, pending=pending,
        reviewer=dict(script=str(Path(__file__).relative_to(ROOT)), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            torch=torch.__version__, executable=sys.executable, threads=torch.get_num_threads(),
            backend_autoload=os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD"), torch_npu_imported="torch_npu" in sys.modules,
            cuda_initialized=torch.cuda.is_initialized(), scope="read-only recorded evidence and independent CPU input reconstruction; no model or device replay"),
        observed_states=dict(Counter(p["state"] for p in points)), points=points, suites=suites, suite_review=suites_review.result(),
        identity_review=dict(**identity_review.result(), groups=identities), shared_region_batch_comparison=comparison,
        unique_reconstructed_inputs=list(INPUT_CACHE.values()), inputs=list(INPUTS.values()),
        units=dict(latency="seconds", throughput="independent input positions/s; complete sequence throughput = B/complete request seconds", memory="bytes"),
        limitations=["Finite initial weights and full-schedule diagnostic outputs/private state are checked. Normal and cold samples contain finite timing/allocation metadata, but their output tensors were not separately checked in timed regions.",
            "Actual readout, returned values/strengths and retained floating state are FP32; state includes int64. FP32 operator arithmetic is the recorded policy, not a new per-operator trace in this audit.",
            "First backbone output maps independent support-witness flags to timed ready boundaries. It excludes import, construction and transfer, has no task head and is not full language-model TTFT.",
            "Stable decode excludes the first incremental call: denominator B*(decode_steps-1). Complete independent sequence throughput is B/complete_seconds, not a token or padding denominator.",
            "Same configuration and seed require the same initial parameter identity across B. Input generation uses full batch/length shapes: equal seed alone does not imply identical trajectory when schedule length or batch changes.",
            "Cold and normal peaks are checked separately against48/56GiB; intrusive diagnostic peaks and unrecorded warmup peaks are outside these checks. No claim about physical capacity maxima or global best batch.",
            "Source/card/raw file identities remain attached to each point. Frozen source and descendant terminal verification belong to the separate final release audit."])
    atomic(HERE/"inference-review.json", result)
    print(json.dumps(dict(state=result["state"], points=len(points), states=result["observed_states"], pending=pending,
        checks=sum(p.get("checks", 0) for p in points)+suites_review.checks+identity_review.checks+comparison.get("checks", 0),
        failures=failures, shared_comparison={k:comparison.get(k) for k in ("state", "output_support_equal", "actual_work_equal")})))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
