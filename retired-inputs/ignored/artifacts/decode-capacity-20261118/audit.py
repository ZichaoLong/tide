"""Independent finite decode-capacity audit; no Torch import unless --inputs.

Run after, or during, the manually managed series. Incomplete runs are reported
without qualification. Expected sampled-budget refusals are distinct from other
execution failures and from inconsistent evidence. --inputs rebuilds CPU inputs
only, one (B,L,d,seed) tensor at a time; it never builds a model or uses an NPU.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
TARGETS = {"9b": 8778358224, "17b": 16707220176}
RUNNING = {"prepared", "starting", "running", "queued", "waiting", "submitted"}
SCHEDULE = dict(prefill=128, prefill_chunk=32, decode_steps=16,
                stream_positions=0, stream_chunk=128)
BUDGET_ERROR = re.compile(r"^MemoryError: .* observed (allocated|reserved) peak exceeds .* sampled budget$")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def relative(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_record(path):
    """Parse and fingerprint the same bytes, including a concurrently read snapshot."""
    if not path.exists():
        return None, None
    data = path.read_bytes()
    identity = dict(path=relative(path), bytes=len(data), sha256=digest(data))
    return json.loads(data), identity


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".audit.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def check(condition, errors, label):
    if not condition:
        errors.append(label)


def close(left, right):
    return (isinstance(left, (int, float)) and math.isfinite(left)
            and math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-12))


def tree_identity(path):
    files = sorted(path.rglob("*.py"))
    if not files:
        raise ValueError("missing frozen Python source: " + str(path))
    sha = hashlib.sha256()
    for file in files:
        sha.update(str(file.relative_to(path)).encode()); sha.update(file.read_bytes())
    return dict(path=str(path), python_files=len(files), sha256=sha.hexdigest())


def observation_check(observed, batch, devices, errors, label, tf=None, *, diagnostic=None):
    calls = observed["calls"]
    check(len(calls) == 20, errors, label + ": call count")
    starts = [0, 32, 64, 96, *range(128, 144)]
    stops = [32, 64, 96, 128, *range(129, 145)]
    check([c["phase"] for c in calls] == ["prefill"] * 4 + ["decode"] * 16,
          errors, label + ": phases")
    check([c["start"] for c in calls] == starts == [c["state_before"] for c in calls],
          errors, label + ": starts/continued state")
    check([c["stop"] for c in calls] == stops == [c["state_after"] for c in calls],
          errors, label + ": stops/continued state")
    for index, call in enumerate(calls):
        check(call["input_positions"] == batch * (call["stop"] - call["start"]),
              errors, label + f": call {index} input denominator")
        check(isinstance(call["seconds"], (int, float)) and math.isfinite(call["seconds"])
              and call["seconds"] > 0, errors, label + f": call {index} finite positive time")
        check(close(call["ready_seconds"], sum(c["seconds"] for c in calls[:index + 1])),
              errors, label + f": call {index} ready boundary")
    stable_calls = [c for c in calls if c["phase"] == "decode"][1:]
    seconds = sum(c["seconds"] for c in stable_calls)
    stable = observed["stable_decode"]
    check(stable["calls"] == len(stable_calls) == 15, errors, label + ": stable count")
    check(stable["input_positions"] == batch * 15, errors, label + ": stable denominator")
    check(close(stable["seconds"], seconds), errors, label + ": stable elapsed")
    check(seconds > 0 and close(stable["positions_per_second"], batch * 15 / seconds),
          errors, label + ": stable throughput")
    # Rebuild every published phase statistic from the elementary call ledger,
    # including p95/median fields used by the separate summary script.
    for field, phase, omit_first in (("prefill", "prefill", False),
                                     ("decode", "decode", False),
                                     ("stable_decode", "decode", True),
                                     ("chunked_stream", "stream", False)):
        selected = [call for call in calls if call["phase"] == phase]
        if omit_first:
            selected = selected[1:]
        durations = [call["seconds"] for call in selected]
        elapsed = sum(durations)
        positions = sum(call["input_positions"] for call in selected)
        expected = dict(calls=len(selected), seconds=elapsed, input_positions=positions,
                        positions_per_second=positions / elapsed if elapsed else None,
                        median_call_seconds=statistics.median(durations) if durations else None,
                        p95_call_seconds=sorted(durations)[math.ceil(.95 * len(durations)) - 1] if durations else None)
        for key, value in expected.items():
            actual = observed[field][key]
            check(actual is None if value is None else close(actual, value),
                  errors, label + f": {field} {key}")
    prompt = next((call["ready_seconds"] for call in calls if call["stop"] == SCHEDULE["prefill"]), None)
    first_decode = next((call["seconds"] for call in calls if call["phase"] == "decode"), None)
    # This finite search admits Settle Tide and causal Transformers only; the
    # Tide config check below rejects a Timed model rather than guessing its
    # graph depth/clock from an unrelated latency field.
    check(prompt is not None and close(observed["prompt_ingested_seconds"], prompt),
          errors, label + ": prompt ingested boundary")
    check(prompt is not None and close(observed["prompt_output_ready_seconds"], prompt),
          errors, label + ": Settle/Transformer prompt output-ready boundary")
    check(first_decode is not None and close(observed["first_incremental_call_seconds"], first_decode),
          errors, label + ": first incremental call")
    if diagnostic is not None:
        flags = diagnostic["output_available_by_call"]
        check(len(flags) == len(calls) and all(type(flag) is bool for flag in flags),
              errors, label + ": diagnostic availability flags")
        first_any = next((call["ready_seconds"] for call, available in zip(calls, flags) if available), None)
        actual = observed["first_any_backbone_output_seconds"]
        check(actual is None if first_any is None else close(actual, first_any),
              errors, label + ": first backbone output boundary")
    check(close(observed["input_window_seconds"], calls[-1]["ready_seconds"]),
          errors, label + ": complete input window boundary")
    check(observed["input_positions"] == batch * 144 and observed["final_state_position"] == 144,
          errors, label + ": complete denominator/final position")
    check(observed["drain_seconds"] == 0, errors, label + ": unexpected drain")
    total = observed["complete_seconds"]
    check(math.isfinite(total) and total >= sum(c["seconds"] for c in calls),
          errors, label + ": complete timing boundary")
    check(total > 0 and close(observed["complete_positions_per_second"], batch * 144 / total),
          errors, label + ": complete throughput")
    check(set(observed["synchronized_devices"]) == devices,
          errors, label + ": participating synchronized devices")
    check(set(observed["allocator"]) == devices, errors, label + ": allocator devices")
    check(observed["diagnostic_checks_in_timing"] is False, errors, label + ": timed diagnostics")
    for device, memory in observed["allocator"].items():
        check(all(isinstance(v, int) and v >= 0 for v in memory.values()),
              errors, label + f": {device} nonnegative memory")
        check(memory["peak_allocated_bytes"] >= memory["allocated_bytes"]
              and memory["peak_reserved_bytes"] >= memory["reserved_bytes"]
              and memory["peak_reserved_bytes"] >= memory["peak_allocated_bytes"],
              errors, label + f": {device} peak ordering")
    if tf:
        expected = 2 * batch * tf["layers"] * min(144, tf["window"] - 1) * tf["d"] * 4
        memory = observed["state_memory"]
        check(memory["payload_bytes"] == expected == memory["retained_storage_bytes"],
              errors, label + ": FP32 KV bytes/retained storage")
        check(memory["dtypes"] == ["torch.float32"] and memory["tensor_count"] == 2 * tf["layers"],
              errors, label + ": FP32 KV dtype/count")


def audit_run(submitted):
    name = submitted["name"]
    if Path(name).name != name:
        raise ValueError("submission name must be a single path component")
    folder = ROOT / "artifacts" / name
    files, records = {}, {}
    errors = []
    for filename in ("inference.json", "reference.json", "status.json", "launch.json", "queue.json", "environment.json"):
        try:
            records[filename], identity = read_record(folder / filename)
            if identity:
                files[filename] = identity
        except (OSError, ValueError) as exc:
            errors.append(filename + ": " + str(exc))
    raw = records.get("inference.json") or {}
    outer = records.get("status.json") or {}
    series = "/".join(str(submitted[k]) for k in ("scale", "kind", "cards"))
    row = dict(name=name, series=series, submission=submitted, files=files,
               outer_state=outer.get("state"), result_state=raw.get("state"),
               last_stage=raw.get("last_stage"), source=raw.get("source"),
               error=raw.get("error"), audit_errors=errors, eligible=False,
               classification="incomplete", observations_checked=0)
    outer_state = outer.get("state")
    terminal = outer_state is not None and outer_state not in RUNNING
    if not terminal and submitted.get("state") != "launch_failed":
        return row
    if not raw:
        row["classification"] = "launch_failed" if submitted.get("state") == "launch_failed" else "execution_failed_without_result"
        check(outer_state != "passed", errors, "passed outer job has no inference record")
        return row
    finished = raw.get("state") in ("completed", "failed")
    row["classification"] = "execution_failed" if finished else "interrupted_execution"
    check(outer_state != "passed" or raw.get("state") == "completed", errors,
          "passed outer job has incomplete/failed result")
    try:
        source = raw["source"]
        check(source["source_commit"] == source["harness_commit"] == submitted["source"] == outer.get("source"),
              errors, "submitted/outer/runtime source identity")
        launch = records.get("launch.json") or {}
        check(launch.get("source") == submitted["source"], errors, "launch/runtime source identity")
        if launch.get("source_path"):
            frozen = Path(launch["source_path"]).resolve()
            check(Path(source["tide_path"]).resolve() == frozen / "src" / "tide"
                  and Path(source["harness_path"]).resolve() == frozen / "benchmarks",
                  errors, "runtime packages differ from launched frozen source")
        row["source_trees"] = {}
        for kind in ("tide", "harness"):
            identity = tree_identity(Path(source[kind + "_path"]))
            row["source_trees"][kind] = identity
            check(identity["sha256"] == source[kind + "_content_sha256"], errors, kind + " frozen source digest")
        workload_path = Path(source["harness_path"]) / "workloads.py"
        data = workload_path.read_bytes()
        row["input_generator"] = dict(path=str(workload_path), bytes=len(data), sha256=digest(data))
        args = raw["arguments"]
        batch, cards = submitted["batch"], submitted["cards"]
        check(args["batch"] == batch, errors, "submitted/runtime batch")
        check(raw["schedule"] == SCHEDULE, errors, "fixed request schedule")
        check(args["precision"] == "fp32" and raw["precision"]["name"] == "fp32", errors, "FP32 policy")
        expected_samples, expected_warmup = (5, 2) if submitted["confirm"] else (2, 1)
        check((args["samples"], args["warmup"]) == (expected_samples, expected_warmup), errors, "screen/confirmation sampling")
        check(args["allocated_budget_gib"] == 48 and args["reserved_budget_gib"] == 56,
              errors, "fixed per-card sampled budgets")
        check(args["seed"] == 751 and args["threads"] == 4, errors, "seed/threads")
        devices = set(args["devices"].split(","))
        check(len(devices) == cards and devices == {f"npu:{i}" for i in range(cards)}, errors, "requested device set")
        physical = str(raw.get("physical_devices", "")).split(",")
        check(len(physical) == cards and len(set(physical)) == cards and all(v.isdigit() for v in physical),
              errors, "physical device identity")
        row["physical_devices"] = raw.get("physical_devices")
        queue = records.get("queue.json") or {}
        check(queue.get("job_id") == name and queue.get("npu_count") == cards,
              errors, "queue admission job/device count")
        # Preserve ordering: physical_ids[i] is the card behind logical npu:i.
        check(queue.get("physical_ids") == [int(v) for v in physical] if all(v.isdigit() for v in physical) else False,
              errors, "runtime/queue physical device mapping")
        check(queue.get("logical_devices") == [f"npu:{i}" for i in range(cards)],
              errors, "queue logical device mapping")
        environment = records.get("environment.json") or {}
        captured = environment.get("environment", {})
        launch_env = launch.get("environment", {})
        origins = environment.get("module_origins", {})
        checked_environment = []
        if origins.get("tide"):
            check(Path(origins["tide"]).resolve().parent == Path(source["tide_path"]).resolve(),
                  errors, "captured Tide import origin differs from runtime source")
            checked_environment.append("tide module origin")
        if origins.get("benchmarks"):
            check(Path(origins["benchmarks"]).resolve().parent == Path(source["harness_path"]).resolve(),
                  errors, "captured benchmark import origin differs from runtime source")
            checked_environment.append("benchmarks module origin")
        for key, expected in (("TIDE_SOURCE_COMMIT", source["source_commit"]),
                              ("TIDE_HARNESS_COMMIT", source["harness_commit"]),
                              ("TIDE_RESOURCE_IDENTITY", raw.get("resource_identity"))):
            for owner, values in (("launch", launch_env), ("captured", captured)):
                if key in values:
                    check(values[key] == expected, errors, f"{owner} {key} differs from runtime")
                    checked_environment.append(owner + " " + key)
        # The capture may precede queue admission and does not promise to record
        # visibility variables. Compare recorded launch/capture values only;
        # physical-card identity is checked against the admitted queue above.
        for key in ("TORCH_DEVICE_BACKEND_AUTOLOAD", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            if key in launch_env and key in captured:
                check(captured[key] == launch_env[key], errors, "captured/launch " + key)
                checked_environment.append("captured/launch " + key)
        if "torch_threads" in raw:
            check(raw["torch_threads"] == args["threads"], errors, "actual Torch thread count")
            checked_environment.append("actual Torch thread count")
        if "cpu_affinity" in raw:
            expected_affinity = list(range(80 + 4 * submitted["slot"], 84 + 4 * submitted["slot"]))
            check(raw["cpu_affinity"] == expected_affinity, errors, "actual slot CPU affinity")
            checked_environment.append("actual slot CPU affinity")
        row["resource_crosscheck"] = dict(queue_physical_ids=queue.get("physical_ids"),
            queue_logical_devices=queue.get("logical_devices"), environment_fields_checked=checked_environment,
            environment_scope="only recorded correlatable fields; absent visibility/toolchain/runtime fields are not inferred")
        is_tide = submitted["kind"] == "tide"
        subject = raw.get("subjects", {}).get("tide", {}) if is_tide else raw
        observations = subject.get("observations", [])
        row["observations_checked"] = len(observations)
        if observations or raw.get("state") == "completed":
            check(all(k in subject for k in ("initial_parameters", "parameters", "input_sha256", "input_shape",
                                             "spec", "placement", "diagnostics")), errors, "missing measured-run identity fields")
        check(len(observations) <= expected_samples, errors, "too many normal samples")
        if raw.get("state") == "completed":
            check(len(observations) == expected_samples and subject.get("state") == "completed", errors, "completed sample count/state")
        # A construction failure need not have any model/input/diagnostic data.
        if "initial_parameters" in subject:
            row["initial_parameters"] = subject["initial_parameters"]
            check(subject["initial_parameters"]["finite"] is True, errors, "finite initial parameters")
            check(re.fullmatch(r"[0-9a-f]{64}", subject["initial_parameters"]["sha256"]) is not None,
                  errors, "initial parameter SHA256 format")
        if "parameters" in subject:
            row["parameters"] = subject["parameters"]
        if "spec" not in subject:
            check(not observations and raw.get("state") != "completed", errors, "missing model spec")
            return row
        spec = subject["spec"]
        row["model_definition"] = raw["config"] if is_tide else {k: v for k, v in spec.items() if k != "placement"}
        row["model_definition_sha256"] = canonical(row["model_definition"])
        row["placement"] = subject["placement"]
        placement = row["placement"]
        check(set(placement["devices"]) == devices, errors, "placed device set")
        if cards > 1:
            check(placement["transport"] == "host_stage", errors, "cross-device transport")
            actual = placement["actual_devices"]
            check(set(actual) == devices and all(v["parameters"] > 0 for v in actual.values()), errors, "every admitted device owns parameters")
            check(sum(v["parameters"] for v in actual.values()) == subject["parameters"], errors, "placement parameter total")
        target = TARGETS[submitted["scale"]]
        if is_tide:
            check(subject["parameters"] == target and args.get("cold") is False, errors, "Tide target/no-cold")
            check(raw["config"]["topology"]["timing"] == "settle", errors,
                  "fixed search requires Settle prompt-readiness semantics")
            if cards > 1:
                check(placement["region_concurrency"] == "per_device", errors, "Tide region concurrency")
        else:
            layers = int(submitted["kind"][2:])
            expected_parameters = layers * (4 * spec["d"] ** 2 + 3 * spec["d"] * spec["ffn_width"] + 2 * spec["d"]) + spec["d"]
            check(spec["layers"] == layers and subject["parameters"] == expected_parameters, errors, "real Transformer parameter formula")
            check(spec["window"] == 128 and spec["chunk"] == 256 and spec["backend"] == "auto"
                  and spec["contiguous_qkv"] is False, errors, "Transformer fixed execution settings")
            check(raw["comparison"]["match"]["targets"]["parameters"] == target, errors, "Transformer target parameter identity")
            if cards > 1:
                check(placement["layer_partitions"] == [i * cards // layers for i in range(layers)], errors, "Transformer contiguous layer partitions")
        if "input_sha256" in subject:
            width = (raw["config"].get("input_dim") or raw["config"]["d"]) if is_tide else spec["d"]
            row["input"] = dict(batch=batch, length=144, width=width, seed=args["seed"] + 1,
                                sha256=subject["input_sha256"], shape=subject["input_shape"],
                                shared_input_prefix=subject.get("shared_input_prefix"), torch=raw.get("torch"))
            check(subject["input_shape"] == [batch, 144, width], errors, "input shape")
        diagnostic = subject.get("diagnostics")
        if observations or raw.get("state") == "completed":
            check(diagnostic is not None, errors, "missing diagnostic")
        if diagnostic:
            row["diagnostics"] = diagnostic
            check(all(diagnostic[k] is True for k in ("finite_outputs", "finite_private_state", "private_state_detached")), errors, "finite detached diagnostic")
            check(diagnostic["final_state_position"] == 144 and len(diagnostic["output_available_by_call"]) == 20,
                  errors, "diagnostic schedule")
            check(diagnostic["readout_dtypes"] == ["torch.float32"], errors, "FP32 readout dtype")
        reference = raw.get("matching_reference")
        if reference:
            check(canonical({k: v for k, v in reference.items() if k != "reference_sha256"}) == reference["reference_sha256"], errors, "Tide reference digest")
            if records.get("reference.json"):
                check(records["reference.json"] == reference, errors, "separate Tide reference identity")
            audit = reference["audit"]
            row["activity"] = {k: audit[k] for k in ("effective_touched", "work", "query_packing", "retention")}
            check(audit["input_positions"] == batch * 144, errors, "activity denominator")
        breaches = []
        for index, observed in enumerate(observations):
            observation_check(observed, batch, devices, errors, f"sample {index}", None if is_tide else spec,
                              diagnostic=diagnostic)
            for device, memory in observed["allocator"].items():
                for kind, limit in (("allocated", 48), ("reserved", 56)):
                    if memory[f"peak_{kind}_bytes"] > limit * 2**30:
                        breaches.append(dict(sample=index, device=device, kind=kind,
                                             observed_bytes=memory[f"peak_{kind}_bytes"], limit_bytes=limit * 2**30))
        row["budget_breaches"] = breaches
        if observations:
            rates = [o["stable_decode"]["positions_per_second"] for o in observations]
            row["metrics"] = dict(samples=len(observations), sample_rates=rates,
                median_stable_decode_positions_per_second=statistics.median(rates),
                median_mean_step_ms=1000 * statistics.median(o["stable_decode"]["seconds"] / 15 for o in observations),
                peak_bytes_by_device={d: {k: max(o["allocator"][d][f"peak_{k}_bytes"] for o in observations)
                                         for k in ("allocated", "reserved")} for d in sorted(devices)})
        if raw.get("state") == "completed" and outer_state == "passed":
            row["classification"] = "completed_within_budget" if not breaches else "completed_over_budget"
            check(not breaches, errors, "completed run exceeded sampled budget without refusal")
        elif BUDGET_ERROR.fullmatch(str(raw.get("error", ""))):
            row["classification"] = "sampled_budget_refusal"
            check(bool(breaches), errors, "sampled-budget MemoryError without a measured breach")
        elif "MemoryError" in str(raw.get("error", "")):
            row["classification"] = "other_memory_error"
    except (KeyError, TypeError, ValueError, OSError, ZeroDivisionError) as exc:
        errors.append(f"malformed/incomplete terminal record: {type(exc).__name__}: {exc}")
    return row


def identity_checks(rows):
    """Compare model/weights within a series and across card-count placements."""
    groups = defaultdict(list)
    for row in rows:
        if "model_definition_sha256" in row and "initial_parameters" in row:
            groups[row["series"]].append(row)
            submitted = row["submission"]
            groups["model/" + submitted["scale"] + "/" + submitted["kind"]].append(row)
    reports = {}
    for key, members in groups.items():
        first = members[0]
        fields = ("parameters", "model_definition_sha256", "initial_parameters")
        same = all(all(r[f] == first[f] for f in fields) for r in members)
        reports[key] = dict(anchor=first["name"], members=[r["name"] for r in members], consistent=same)
        if not same:
            for row in members:
                row["audit_errors"].append("model/parameter identity differs in " + key)
    inputs = defaultdict(list)
    for row in rows:
        if row.get("input"):
            x = row["input"]
            inputs[(x["batch"], x["length"], x["width"], x["seed"])].append(row)
    for members in inputs.values():
        if len({r["input"]["sha256"] for r in members}) != 1:
            for row in members:
                row["audit_errors"].append("same-shaped seeded inputs have different raw hashes")
    return reports


def rebuild_inputs(rows, threads, max_gib):
    # Disable third-party device autoload before importing Torch. No device API,
    # model construction, placement, or benchmark execution appears in this path.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    torch.set_num_threads(threads)
    groups = defaultdict(list)
    for row in rows:
        if row.get("input"):
            x = row["input"]
            groups[(row["input_generator"]["sha256"], x["batch"], x["length"], x["width"], x["seed"])].append(row)
    reports = []
    prefix_hashes = {}
    for key, members in sorted(groups.items()):
        generator_hash, batch, length, width, seed = key
        report = dict(batch=batch, length=length, width=width, seed=seed,
                      runs=[r["name"] for r in members], generator_sha256=generator_hash,
                      torch=str(torch.__version__), state="not_checked")
        reports.append(report)
        # paired_input keeps per-feature pieces until stack completes; account
        # for two full tensors, plus the independently generated narrow witness.
        estimated = 2 * batch * length * width * 4 + 2 * batch * length * 768 * 4
        report["estimated_input_working_bytes"] = estimated
        if estimated > max_gib * 2**30:
            report["reason"] = "CPU input reconstruction exceeds explicit working-memory budget"
            continue
        errors = []
        for row in members:
            check(row["input"]["torch"] == str(torch.__version__), errors, "CPU reconstruction Torch version differs")
        path = Path(members[0]["input_generator"]["path"])
        check(digest(path.read_bytes()) == generator_hash, errors, "input generator changed since snapshot")
        if errors:
            report.update(state="failed", errors=errors)
            for row in members:
                row["audit_errors"].extend(errors)
            continue
        spec = importlib.util.spec_from_file_location("capacity_input_generator", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        value = narrow = None
        try:
            value = module.paired_input(batch, length, width, seed)
            check(value.device.type == "cpu" and value.dtype == torch.float32, errors, "reconstruction must stay CPU FP32")
            # NumPy's memoryview avoids an additional full-tensor bytes copy.
            whole = hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest()
            # A width prefix is strided: hash one sample at a time, in the same
            # contiguous B/L/feature order as workloads.tensor_hash.
            prefix = hashlib.sha256()
            for sample in value:
                piece = sample[:, :768].contiguous()
                prefix.update(memoryview(piece.numpy()).cast("B"))
                del piece
            del sample
            prefix = prefix.hexdigest()
            del value; value = None; gc.collect()
            prefix_key = (generator_hash, batch, length, seed)
            if prefix_key not in prefix_hashes:
                narrow = module.paired_input(batch, length, 768, seed)
                prefix_hashes[prefix_key] = hashlib.sha256(memoryview(narrow.numpy()).cast("B")).hexdigest()
                del narrow; narrow = None; gc.collect()
            check(width >= 768 and prefix == prefix_hashes[prefix_key], errors, "common 768-feature prefix")
            for row in members:
                check(whole == row["input"]["sha256"], errors, row["name"] + ": complete input hash")
                old_prefix = row["input"].get("shared_input_prefix")
                if old_prefix:
                    check(old_prefix["width"] == 768 and old_prefix["sha256"] == prefix,
                          errors, row["name"] + ": recorded prefix hash")
            report.update(state="failed" if errors else "passed", full_sha256=whole,
                          common_prefix_width=768, common_prefix_sha256=prefix,
                          expected_prefix_sha256=prefix_hashes[prefix_key], errors=errors)
        except Exception as exc:
            errors.append(f"CPU input reconstruction: {type(exc).__name__}: {exc}")
            report.update(state="failed", errors=errors)
        finally:
            del value, narrow, module; gc.collect()
        for row in members:
            row["audit_errors"].extend(errors)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions", type=Path, default=BASE / "submissions.json")
    parser.add_argument("--output", type=Path, default=BASE / "audit.json")
    parser.add_argument("--inputs", action="store_true")
    parser.add_argument("--input-threads", type=int, default=2)
    parser.add_argument("--input-max-gib", type=float, default=8)
    args = parser.parse_args(argv)
    if args.input_threads < 1 or not math.isfinite(args.input_max_gib) or args.input_max_gib <= 0:
        parser.error("CPU input thread/memory limits must be positive and finite")
    submissions, manifest = read_record(args.submissions)
    if not isinstance(submissions, list):
        parser.error("submissions must be a JSON list")
    rows = []
    for submitted in submissions:
        try:
            rows.append(audit_run(submitted))
        except Exception as exc:
            rows.append(dict(name=submitted.get("name"), classification="audit_read_failure", eligible=False,
                             audit_errors=[f"{type(exc).__name__}: {exc}"]))
    series = identity_checks(rows)
    input_reports = rebuild_inputs(rows, args.input_threads, args.input_max_gib) if args.inputs else []
    input_states = {name: group["state"] for group in input_reports for name in group["runs"]}
    for row in rows:
        row["input_verification"] = input_states.get(row["name"], "not_checked" if args.inputs else "not_requested")
        row["eligible"] = (row["classification"] == "completed_within_budget" and not row["audit_errors"]
                           and row["input_verification"] in ("passed", "not_requested"))
    counts = Counter(r["classification"] for r in rows)
    errors = sum(len(r["audit_errors"]) for r in rows)
    report = dict(schema="tide-decode-capacity-audit-1", recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        auditor=dict(path=relative(Path(__file__).resolve()), sha256=digest(Path(__file__).read_bytes()),
                     python=sys.version, torch_imported="torch" in sys.modules),
        submissions=manifest, rules=dict(schedule=SCHEDULE, allocated_budget_gib=48, reserved_budget_gib=56,
            throughput="median of per-normal-sample 15*B / stable seconds; screens and confirmations retain their own sample counts",
            latency_checks="phase medians/p95 and first incremental/prompt boundaries rebuilt from calls; first backbone output uses the separate diagnostic availability flags; Settle Tide/causal Transformer scope",
            memory="per-device maximum of normal-request allocator peaks; no cross-device sum; not a hard allocator cap",
            resource_checks="ordered runtime/queue physical and logical mapping; launch source/resource identity and captured import/environment fields only where recorded",
            incomplete="snapshot only; no qualification or input reconstruction",
            input_scope="exact same-B/L/seed common width prefix; no assertion of trajectory equality across batch sizes",
            input_working_memory_limit_gib=args.input_max_gib if args.inputs else None),
        series=series, runs=rows, inputs=dict(requested=args.inputs, groups=input_reports,
            all_checked=bool(args.inputs) and bool(input_reports) and all(r["state"] == "passed" for r in input_reports)),
        summary=dict(runs=len(rows), classes=dict(counts), eligible=sum(r["eligible"] for r in rows),
                     all_terminal=bool(rows) and counts["incomplete"] == 0,
                     validation_errors=errors, audit_state="failed" if errors else "passed",
                     execution_failures=[r["name"] for r in rows if r["classification"] in
                         {"execution_failed", "interrupted_execution", "other_memory_error", "launch_failed", "execution_failed_without_result"}]))
    atomic(args.output, report)
    print(json.dumps(dict(output=str(args.output), **report["summary"])))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
