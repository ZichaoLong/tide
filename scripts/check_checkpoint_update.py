"""Recover one existing FP32 checkpoint and compare an actual further update.

The preserved checkpoint is read directly and never copied or overwritten. A new
reference owner starts from its parameters, optimizer and private state, with
explicitly empty gradients and a newly declared RNG seed. Its next forward/update
is compared to a fresh interpreter loading the same checkpoint. This does not
recreate an original process's unrecorded gradients/RNG or repair earlier failures.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import traceback

import torch
import tide

from benchmarks.baselines import parameter_count
from benchmarks.execution_candidates import configure_device, rss_record
from benchmarks.inference import allocator_stats, synchronize_all
from benchmarks.programme import run_child
from benchmarks.workloads import tensor_hash
from scripts.check_execution_candidates import (Artifacts, COMPLETED, TOLERANCE,
    extra_anchor, file_sha256, implementation_identity, parent_memory_before_child,
    worker as existing_restore_worker)
from scripts.check_training_paths import observe, screen, training_data
from scripts.job import atomic, enable_subreaper
from tide import ModelConfig, TideModel, load_checkpoint, placement_adamw


KNOWN_STATES = COMPLETED | {"nonfinite", "exact_contract_failed", "structural_failed",
    "metadata_failed", "support_failed", "execution_failed"}
DEVICE_BUDGET_BYTES = 56 << 30


def allocator_peak_budget(statistics, scope):
    """Sampled allocator limits; CPU's empty device dictionary is a no-op."""
    violations = []
    for device, row in statistics.items():
        for kind in ("allocated", "reserved"):
            observed = row["peak_"+kind+"_bytes"]
            if observed > DEVICE_BUDGET_BYTES:
                violations.append(dict(scope=scope, device=device, kind=kind,
                    observed_bytes=observed, limit_bytes=DEVICE_BUDGET_BYTES))
    return dict(limit_bytes=DEVICE_BUDGET_BYTES, devices_observed=sorted(statistics),
                within_budget=not violations, violations=violations,
                scope="sampled process allocator peak, not a hard allocation cap; empty CPU allocator is a no-op")


def recovery_memory_budget(parent, child_statistics=None):
    """Keep parent and child checks separate, then bound simultaneous ownership."""
    before, held = parent["before_cache_release"], parent["after_cache_release"]
    checked = allocator_peak_budget(before, "parent_cumulative_peak")
    report = dict(parent=checked, parent_after_cache_release=held, child=None,
                  conservative_overlap={}, violations=list(checked["violations"]))
    if child_statistics is not None:
        child = report["child"] = allocator_peak_budget(child_statistics, "child_cumulative_peak")
        report["violations"].extend(child["violations"])
        if set(held) != set(child_statistics):
            raise ValueError("parent and child allocator device identities differ")
        for device in held:
            row = {}
            for kind in ("allocated", "reserved"):
                parent_bytes = held[device][kind+"_bytes"]
                child_peak = child_statistics[device]["peak_"+kind+"_bytes"]
                bound = parent_bytes+child_peak
                row[kind] = dict(parent_held_bytes=parent_bytes, child_peak_bytes=child_peak,
                                 conservative_bound_bytes=bound, limit_bytes=DEVICE_BUDGET_BYTES)
                if bound > DEVICE_BUDGET_BYTES:
                    report["violations"].append(dict(scope="conservative_parent_child_overlap",
                        device=device, kind=kind, observed_bytes=bound, limit_bytes=DEVICE_BUDGET_BYTES))
            report["conservative_overlap"][device] = row
    report.update(within_budget=not report["violations"],
        interpretation="parent held allocation/reservation after synchronized cache release plus the child's cumulative peak; parent performs no device computation while waiting; this is a conservative process-allocator overlap bound, not a measured whole-device peak or opaque runtime-memory accounting")
    return report


def validate_limits(args):
    if not 1 <= args.threads <= 32:
        raise ValueError("threads must be 1..32")
    if not 1 <= args.max_parameters <= 200_000_000:
        raise ValueError("max-parameters must be 1..200000000; no 9B training")
    if not math.isfinite(args.artifact_budget_gib) or not 0 < args.artifact_budget_gib <= 10:
        raise ValueError("artifact-budget-gib must be finite, positive and <=10")
    if not math.isfinite(args.restore_timeout_seconds) or not 1 <= args.restore_timeout_seconds <= 7200:
        raise ValueError("restore-timeout-seconds must be finite and 1..7200")
    if not 1 <= args.chunk <= 128 or not 0 <= args.seed < 2**63:
        raise ValueError("chunk must be 1..128 and seed a nonnegative signed-int64 value")
    if str(args.device).split(":", 1)[0] not in {"cpu", "npu"}:
        raise ValueError("this finite recovery supports CPU/NPU FP32 only")


def checkpoint_metadata(path, maximum):
    """Inspect storage metadata and a meta constructor before real allocation."""
    payload = torch.load(path, map_location="meta", mmap=True, weights_only=False)
    if payload.get("format") != "tide-instances-1" or payload.get("placement") is not None:
        raise ValueError("this recovery entry requires an ordinary single-device public checkpoint")
    config = ModelConfig.from_dict(payload["config"])
    with torch.device("meta"):
        model = TideModel(config)
    count = parameter_count(model)
    if count > maximum:
        raise ValueError("checkpoint exceeds the declared parameter bound before real weight allocation")
    if not payload.get("optimizer") or not payload["optimizer"].get("state"):
        raise ValueError("checkpoint requires nonempty saved optimizer slots")
    expected = model.state_dict()
    actual = payload["parameters"]
    if actual.keys() != expected.keys() or any(actual[key].shape != expected[key].shape for key in expected):
        raise ValueError("checkpoint parameter/buffer geometry differs from its constructor")
    if any(value.is_floating_point() and value.dtype != torch.float32 for value in actual.values()):
        raise ValueError("checkpoint floating parameters must already be FP32")
    return dict(config=config.to_dict(), parameters=count, format=payload["format"])


def retained_input(anchor_path, config, identity):
    """Read the final retained input; no model parameter/state is reconstructed."""
    anchor = torch.load(anchor_path, map_location="cpu", mmap=True, weights_only=False)
    if ModelConfig.from_dict(anchor["config"]).to_dict() != config:
        raise ValueError("input anchor configuration differs from the checkpoint")
    recorded_source = anchor.get("source", {})
    if recorded_source.get("tide_content_sha256") != identity["tide_content_sha256"]:
        raise ValueError("input anchor production package differs from this frozen implementation")
    inputs = anchor.get("inputs")
    if not isinstance(inputs, (tuple, list)) or not inputs:
        raise ValueError("input anchor requires a nonempty retained input sequence")
    value = inputs[-1]
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.ndim != 3 or
            not 1 <= value.shape[0] <= 16 or not 1 <= value.shape[1] <= 128 or
            value.shape[2] != (config["input_dim"] or config["d"])):
        raise ValueError("retained final input must be FP32 B1..16/L1..128 with the checkpoint input width")
    value = value.clone()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("retained input must be finite")
    return value, dict(index=len(inputs)-1, count=len(inputs), source=recorded_source,
                       sha256=tensor_hash(value), shape=list(value.shape), dtype=str(value.dtype))


def initialize_reference_rng(optimizer, device, seed):
    """Declare a new comparison anchor, never an original gradient/RNG resume."""
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    if device.type == "npu":
        torch.npu.manual_seed_all(seed)


def restore_worker(bundle_path, output, *, device, threads):
    """Existing complete recovery checks plus this process's allocator boundary."""
    header = torch.load(bundle_path, map_location="cpu", mmap=True, weights_only=False)
    if str(header["options"]["device"]) != str(device):
        raise ValueError("worker device differs from the immutable bundle")
    del header
    device = configure_device(device, threads)
    synchronize_all([device])
    allocator_stats([device], reset=True)
    before = allocator_stats([device])
    try:
        record = existing_restore_worker(bundle_path, output, threads)
    except Exception as error:
        record = dict(state="execution_failed", error=repr(error), pid=os.getpid(),
                      input_bundle_sha256=file_sha256(bundle_path))
        traceback.print_exc()
    finally:
        synchronize_all([device])
        after = allocator_stats([device])
    record["process_memory"] = dict(before_checkpoint_load=before, after_update_and_checks=after,
        rss=rss_record(), scope="fresh recovery process only; allocator peaks reset before checkpoint load and include load/forward/backward/update/checks; parent remains a separate process; not ordinary performance memory")
    record["process_memory"]["budget"] = allocator_peak_budget(after, "child_cumulative_peak")
    if not record["process_memory"]["budget"]["within_budget"]:
        record.update(numerical_state=record["state"], state="execution_failed",
                      failure_kind="sampled_device_memory_budget_refusal")
    atomic(output, record)
    return record


def run(args):
    validate_limits(args)
    torch.set_num_threads(args.threads)
    checkpoint, input_anchor = args.checkpoint.resolve(), args.input_anchor.resolve()
    inspected = checkpoint_metadata(checkpoint, args.max_parameters)
    identity = implementation_identity()
    value, input_record = retained_input(input_anchor, inspected["config"], identity)
    if args.chunk > value.shape[1]:
        raise ValueError("chunk must not exceed the retained input length")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = Artifacts(directory, args.artifact_budget_gib)
    record = dict(schema="tide-existing-checkpoint-update-v1", state="running", qualified=False,
        source=identity, precision="fp32", tolerance=TOLERANCE, inspected_checkpoint=inspected,
        arguments={key: str(part) if isinstance(part, Path) else part for key, part in vars(args).items()},
        checkpoint=dict(path=str(checkpoint), sha256=file_sha256(checkpoint), bytes=checkpoint.stat().st_size,
                        policy="existing public checkpoint read directly; never copied or overwritten"),
        input_anchor=dict(path=str(input_anchor), sha256=file_sha256(input_anchor), bytes=input_anchor.stat().st_size),
        input=input_record, artifacts=artifacts.records, artifact_budget_bytes=artifacts.limit,
        cpu_affinity=sorted(os.sched_getaffinity(0)), physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified"),
        reference_anchor=dict(gradients="all None", seed=args.seed,
            meaning="new reference owner initialized from the preserved checkpoint, with newly declared empty gradients and RNG; not a claim of continuing the old process's unrecorded gradients/RNG"),
        boundary="one new reference next forward/update at the saved parameter/optimizer/private-state point; one actual further update after loading that same public checkpoint in a distinct interpreter",
        original_evidence="earlier numerical failures and artifact-budget refusal remain unchanged; this local recovery check cannot qualify an unrun candidate or repair cross-mode failures",
        objective="existing check_training_paths signed-cosine backbone VJP across chunks; no task head or model-quality claim",
        last_stage="construction")
    def persist():
        record["retained_artifact_bytes"] = artifacts.used()
        atomic(directory/"result.json", record)
    persist()
    try:
        device = configure_device(args.device, args.threads)
        allocator_stats([device], reset=True)
        record["reference_process_before_load"] = allocator_stats([device])
        model, state, slots = load_checkpoint(checkpoint, device=device)
        optimizer = placement_adamw(model, lr=1e-4, foreach=False)
        optimizer.load_state_dict(slots)
        del slots
        initialize_reference_rng(optimizer, device, args.seed)
        extra = extra_anchor(model, device)
        if any(gradient is not None for gradient in extra["gradients"].values()):
            raise AssertionError("new reference gradient anchor is not empty")
        saved = training_data(model, optimizer, state)
        saved_screen = screen(saved)
        del saved
        record["reference_anchor"].update(saved_data=saved_screen, gradient_rng=screen(extra),
            state_position=state.position, deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        record["last_stage"] = "reference_next_forward"
        persist()
        prediction, temporary_state, _ = observe(model, optimizer, state, value, "fp32", [device], args.chunk, grad=False)
        del temporary_state
        record["reference_forward"] = screen(prediction)
        record["last_stage"] = "reference_actual_update"
        persist()
        updated, temporary_state, work = observe(model, optimizer, state, value, "fp32", [device], args.chunk, update=True)
        del temporary_state
        record["reference_update"] = dict(summary=screen(updated), optimizer_update=updated["optimizer_update"], work=work)
        bundle = dict(checkpoint=str(checkpoint), checkpoint_sha256=record["checkpoint"]["sha256"],
            saved_screen=saved_screen, extra=extra, input=value, next_forward=prediction, updated=updated,
            options=dict(device=str(device), chunk=args.chunk, deterministic=torch.are_deterministic_algorithms_enabled()),
            config=model.config.to_dict(), source=identity, artifact_root=str(directory),
            artifact_budget_gib=args.artifact_budget_gib)
        bundle_path = directory/"restore-input.pt"
        bundle_record = artifacts.save(bundle_path, bundle, "new reference next-update witness and declared gradient/RNG sidecar")
        del bundle, extra, prediction, updated
        gc.collect()
        record["parent_memory_before_child"] = parent_memory_before_child(device)
        record["memory_budget"] = recovery_memory_budget(record["parent_memory_before_child"])
        record["reference_process_rss"] = rss_record()
        persist()
        if not record["memory_budget"]["within_budget"]:
            record["failure_kind"] = "sampled_device_memory_budget_refusal"
            raise MemoryError("parent allocator peak exceeded the 56 GiB sampled budget before recovery")
        record["last_stage"] = "fresh_process_recovery"
        output = directory/"restore.json"
        command = [sys.executable, "-u", "-m", "scripts.check_checkpoint_update", "--resume-bundle", str(bundle_path),
            "--resume-output", str(output), "--device", str(device), "--threads", str(args.threads)]
        child = record["child"] = dict(command=command, log=str(directory/"restore.log"), output=str(output),
            timeout_seconds=args.restore_timeout_seconds, state="planned")
        persist()
        # Dedicated serial supervisor: retain descendants even if a compiler
        # starts a new session. The outer durable service supplies RSS limits.
        enable_subreaper()
        run_child(child, persist, supervisor_pid=os.getpid())
        if not output.exists():
            raise RuntimeError("fresh-process recovery produced no result; inspect restore.log")
        restored = json.loads(output.read_text())
        record["restore"] = restored
        if (restored.get("input_bundle_sha256") != bundle_record["sha256"] or
                type(restored.get("pid")) is not int or restored["pid"] == os.getpid() or restored["pid"] <= 0):
            raise RuntimeError("recovery result does not identify this bundle and a distinct interpreter")
        if restored.get("state") not in KNOWN_STATES:
            raise RuntimeError("fresh-process recovery has an unknown result state")
        expected_exit = 0 if restored["state"] in COMPLETED else 2
        if child.get("exit_code") != expected_exit or child["state"] not in {"completed", "failed"}:
            raise RuntimeError("recovery process termination disagrees with its numerical record")
        artifacts.records.extend(restored.get("failed_artifacts", []))
        record["memory_budget"] = recovery_memory_budget(record["parent_memory_before_child"],
            restored["process_memory"]["after_update_and_checks"])
        persist()
        if not record["memory_budget"]["within_budget"]:
            record["failure_kind"] = "sampled_device_memory_budget_refusal"
            raise MemoryError("child or conservative parent+child allocator overlap exceeded the 56 GiB sampled budget")
        record.update(state=restored["state"], qualified=restored["state"] == "within_threshold",
                      last_stage="complete", original_process_pid=os.getpid())
        persist()
        if record["qualified"]:
            artifacts.retire(bundle_path)
        persist()
        return record
    except Exception as error:
        record.update(state="execution_failed", qualified=False, error=repr(error))
        persist()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input-anchor", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--max-parameters", type=int, default=200_000_000)
    parser.add_argument("--artifact-budget-gib", type=float, default=8.)
    parser.add_argument("--restore-timeout-seconds", type=float, default=1800.)
    parser.add_argument("--resume-bundle", type=Path)
    parser.add_argument("--resume-output", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_limits(args)
    except ValueError as error:
        parser.error(str(error))
    if Path(tide.__file__).resolve().parent != Path(__file__).resolve().parents[1]/"src"/"tide":
        parser.error("Tide must resolve from this frozen checkout")
    try:
        if args.resume_bundle is not None:
            if args.resume_output is None or args.resume_output.exists():
                parser.error("worker requires a new output path")
            record = restore_worker(args.resume_bundle, args.resume_output, device=args.device, threads=args.threads)
        else:
            if args.checkpoint is None or args.input_anchor is None or args.output_dir is None:
                parser.error("checkpoint, input-anchor and new output-dir are required")
            record = run(args)
    except Exception:
        traceback.print_exc()
        return 2
    print(json.dumps(dict(state=record["state"], qualified=record["state"] == "within_threshold")))
    return 0 if record["state"] in COMPLETED else 2


if __name__ == "__main__":
    raise SystemExit(main())
