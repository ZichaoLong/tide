"""Bounded FP32 same-source W/P qualification with real fresh-process recovery.

The first real update supplies one common parameter/AdamW/private-state/gradient/
RNG anchor. Every execution candidate starts there, continues three updates and
recovers its public checkpoint in a distinct interpreter before a further update.
These intrusive observations are never performance samples.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch
import tide

from benchmarks.baselines import parameter_count
from benchmarks.costs import _rng
from benchmarks.execution_candidates import VARIANTS, configure_device, resolved_config, variant_options
from benchmarks.inference import allocator_stats, source_identity, synchronize_all
from benchmarks.training_breakdown import copy_tree
from benchmarks.workloads import paired_input, tensor_hash
from scripts.check_memory_candidates import aggregate, comparison_summary
from scripts.check_training_paths import (build_candidate, compare, leaves, observe,
    optimizer_step_count, screen, training_data)
from scripts.job import atomic
from scripts.qualify_precision import THRESHOLDS, atomic_tensor
from tide import (ModelConfig, TideModel, canonical_state_dict, load_checkpoint,
                  placement_adamw, save_checkpoint)


TOLERANCE = THRESHOLDS["fp32"]
COMPLETED = {"within_threshold", "finite_deviation"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def implementation_identity():
    result = source_identity()
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return dict(result, scripts_content_sha256=digest.hexdigest())


class Artifacts:
    """Finite byte budget; successful replay payloads may be retired explicitly."""
    def __init__(self, directory, budget_gib):
        self.directory = Path(directory)
        self.limit = int(budget_gib*2**30)
        self.records = []

    def used(self):
        return sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file())

    def reserve(self, payload):
        # Payload bytes conservatively count repeated tensor aliases; the 1 MiB
        # margin covers serialization headers, names and non-tensor metadata.
        estimated = sum(value.numel()*value.element_size() for _, value in leaves(payload)
                        if isinstance(value, torch.Tensor))+(1 << 20)
        if self.used()+estimated > self.limit:
            raise MemoryError("qualification artifact byte budget would be exceeded")

    def record(self, path, role):
        path = Path(path)
        row = dict(path=str(path.relative_to(self.directory)), bytes=path.stat().st_size,
                   sha256=file_sha256(path), role=role, retained=True)
        self.records.append(row)
        if self.used() > self.limit:
            raise MemoryError("qualification artifact byte budget exceeded")
        return row

    def save(self, path, payload, role):
        self.reserve(payload)
        atomic_tensor(path, payload)
        return self.record(path, role)

    def retire(self, path):
        path = Path(path)
        relative = str(path.relative_to(self.directory))
        row = next(row for row in self.records if row["path"] == relative)
        if not row["retained"] or file_sha256(path) != row["sha256"]:
            raise RuntimeError("only the identified successful intermediate may be retired")
        path.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        row.update(retained=False, retirement="successful comparison recorded; deterministic intermediate is reproducible from retained common anchor/checkpoint and frozen source")


def extra_anchor(model, device):
    return dict(gradients=canonical_state_dict(model, gradients=True), rng=copy_tree(_rng(device), "cpu"))


def restore_extra(model, extra, device):
    if set(extra["gradients"]) != {name for name, _ in model.named_parameters()}:
        raise ValueError("gradient anchor parameter names differ")
    for name, parameter in model.named_parameters():
        gradient = extra["gradients"][name]
        parameter.grad = None if gradient is None else gradient.to(parameter.device, copy=True)
    torch.set_rng_state(extra["rng"]["cpu"])
    if device.type == "npu":
        torch.npu.set_rng_state(extra["rng"]["npu"], device)


def classified(checks, *, exact=()):
    result = comparison_summary(checks)
    if any(not check["comparison"]["exact"] for check in exact):
        result["state"] = aggregate([result["state"], "exact_contract_failed"])
    return result


def parent_memory_before_child(device):
    """Release only inactive NPU allocator cache; all live owners stay resident."""
    synchronize_all([device])
    before = allocator_stats([device])
    if device.type == "npu":
        torch.npu.empty_cache()
    synchronize_all([device])
    after = allocator_stats([device])
    return dict(before_cache_release=before, after_cache_release=after,
        cache_release_requested=device.type == "npu",
        scope="parent-process device allocator immediately before spawning recovery; peaks are cumulative and not reset; empty_cache releases only inactive cached blocks; model, gradients, optimizer and private state remain resident; not normal-training memory")


def worker(bundle_path, output, threads):
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    options = bundle["options"]
    device = configure_device(options["device"], threads)
    torch.use_deterministic_algorithms(options["deterministic"])
    checkpoint = Path(bundle["checkpoint"])
    if file_sha256(checkpoint) != bundle["checkpoint_sha256"]:
        raise RuntimeError("checkpoint differs from its recorded identity")
    model, state, slots = load_checkpoint(checkpoint, device=device)
    optimizer = placement_adamw(model, lr=1e-4, foreach=False)
    optimizer.load_state_dict(slots)
    del slots
    restore_extra(model, bundle["extra"], device)
    loaded_screen = screen(training_data(model, optimizer, state))
    extra_check = compare(extra_anchor(model, device), bundle["extra"], TOLERANCE)
    checks = {"gradient_rng_anchor": extra_check}
    predicted, _, _ = observe(model, optimizer, state, bundle["input"], "fp32", [device], options["chunk"], grad=False)
    checks["next_forward"] = compare(predicted, bundle["next_forward"], TOLERANCE)
    del predicted
    updated, _, work = observe(model, optimizer, state, bundle["input"], "fp32", [device], options["chunk"], update=True)
    checks["actual_update"] = compare(updated, bundle["updated"], TOLERANCE)
    summary = classified(checks, exact=[extra_check, checks["next_forward"]])
    identity = implementation_identity()
    record = dict(state=summary["state"], summary=summary, checks=checks, pid=os.getpid(),
        input_bundle_sha256=file_sha256(bundle_path), checkpoint_sha256=bundle["checkpoint_sha256"],
        source=identity, source_exact=identity == bundle["source"],
        config_exact=model.config.to_dict() == bundle["config"],
        saved_data_exact=loaded_screen["sha256"] == bundle["saved_screen"]["sha256"],
        saved_screen=loaded_screen, expected_saved_screen=bundle["saved_screen"],
        actual_update=updated["optimizer_update"]["checked"], optimizer_update=updated["optimizer_update"],
        work=work, tolerance=TOLERANCE,
        checkpoint_scope="public checkpoint restores parameters/optimizer/private state/config; declared sidecar restores preexisting gradients and RNG before a genuine further update")
    if not all(record[key] for key in ("saved_data_exact", "source_exact", "config_exact", "actual_update")):
        record["state"] = aggregate([record["state"], "exact_contract_failed"])
    if record["state"] != "within_threshold":
        artifacts = Artifacts(bundle["artifact_root"], bundle["artifact_budget_gib"])
        artifacts.save(output.parent/"restore-failed-actual.pt", updated,
                       "retained actual failed fresh-process update; expected update remains in restore-input.pt")
        record["failed_artifacts"] = artifacts.records
    atomic(output, record)
    return record


def check_restore(model, optimizer, state, value, directory, artifacts, *, device, chunk, timeout):
    checkpoint = directory/"checkpoint.pt"
    saved = training_data(model, optimizer, state)
    artifacts.reserve(saved)
    saved_screen = screen(saved)
    del saved
    save_checkpoint(checkpoint, model, state=state, optimizer=optimizer)
    checkpoint_record = artifacts.record(checkpoint, "retained actual recovery anchor")
    extra = extra_anchor(model, device)
    prediction, _, _ = observe(model, optimizer, state, value, "fp32", [device], chunk, grad=False)
    updated, _, work = observe(model, optimizer, state, value, "fp32", [device], chunk, update=True)
    bundle_path = directory/"restore-input.pt"
    bundle = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_record["sha256"],
        saved_screen=saved_screen, extra=extra, input=value, next_forward=prediction, updated=updated,
        options=dict(device=str(device), chunk=chunk, deterministic=torch.are_deterministic_algorithms_enabled()),
        config=model.config.to_dict(), source=implementation_identity(),
        artifact_root=str(artifacts.directory.resolve()), artifact_budget_gib=artifacts.limit/2**30)
    bundle_record = artifacts.save(bundle_path, bundle, "fresh-process expected update and gradient/RNG sidecar")
    del bundle, extra, prediction, updated
    gc.collect()
    parent_memory = parent_memory_before_child(device)
    # Keep this boundary even if the fresh interpreter times out or fails before
    # producing its own result. No surviving training owner is offloaded.
    atomic(directory/"parent-memory-before-child.json", parent_memory)
    output = directory/"restore.json"
    command = [sys.executable, "-m", "scripts.check_execution_candidates", "--resume-bundle", str(bundle_path.resolve()),
               "--resume-output", str(output.resolve()), "--threads", str(torch.get_num_threads())]
    environment = dict(os.environ)
    package_parent = str(Path(tide.__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = package_parent+(os.pathsep+environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    with (directory/"restore.log").open("x") as log:
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=environment,
            stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        log.flush(); os.fsync(log.fileno())
    if not output.exists():
        raise RuntimeError("fresh-process recovery produced no record; inspect restore.log")
    result = json.loads(output.read_text())
    if (result.get("input_bundle_sha256") != bundle_record["sha256"] or
            type(result.get("pid")) is not int or result["pid"] <= 0 or result["pid"] == os.getpid()):
        raise RuntimeError("recovery did not identify this bundle and a distinct interpreter")
    if result.get("state") not in COMPLETED | {"nonfinite", "exact_contract_failed", "structural_failed", "metadata_failed", "support_failed"}:
        raise RuntimeError("recovery did not complete its checks: "+str(result.get("error", result.get("state"))))
    if completed.returncode != (0 if result["state"] in COMPLETED else 2):
        raise RuntimeError("fresh-process exit code disagrees with its result")
    result.update(new_process=True, original_process_pid=os.getpid(), command=command,
                  exit_code=completed.returncode, original_update_work=work,
                  parent_memory_before_child=parent_memory,
                  memory_scope="parent and fresh child may simultaneously retain model/optimizer on the same device; bounded to <=200M, never a 9B checkpoint; this process-overlap peak is qualification overhead, not normal training memory")
    artifacts.records.extend(result.get("failed_artifacts", []))
    # Retain the checkpoint regardless; full expected payload survives every
    # failed tolerance/discrete check for bounded follow-up investigation.
    if result["state"] == "within_threshold":
        atomic(output, result)  # fsync successful result and bundle SHA first
        artifacts.retire(bundle_path)
    return result


def run(config, *, directory, candidates=("baseline", "packing"), device="cpu", batch=2,
        length=8, chunk=4, steps=3, seed=1701, artifact_budget_gib=20., restore_timeout=1800.,
        progress=lambda record: None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("qualification directory must be new and empty")
    if not candidates or candidates[0] != "baseline" or len(set(candidates)) != len(candidates):
        raise ValueError("baseline must be first and candidate names unique")
    if steps != 3 or not (1 <= batch <= 16 and 1 <= length <= 128 and 1 <= chunk <= length):
        raise ValueError("this qualification requires three updates and bounded B1..16/L1..128/chunk<=L")
    if not math.isfinite(artifact_budget_gib) or not 0 < artifact_budget_gib <= 64:
        raise ValueError("artifact budget must be positive and <=64 GiB")
    device = torch.device(device)
    if device.type not in {"cpu", "npu"}:
        raise ValueError("CPU/NPU FP32 only")
    variants = {name: variant_options(name) for name in candidates}
    base = resolved_config(config, "baseline")
    with torch.device("meta"):
        count = parameter_count(TideModel(base))
    if count > 200_000_000:
        raise ValueError("qualification is bounded to 200M parameters; no 9B training/checkpoint")
    artifacts = Artifacts(directory, artifact_budget_gib)
    inputs = [paired_input(batch, length, config.input_dim or config.d, seed+1+i) for i in range(steps+2)]
    record = dict(schema="tide-execution-qualification-v1", state="running", source=implementation_identity(),
        precision="fp32", tolerance=TOLERANCE, base_config=base.to_dict(), resolved_candidates=variants,
        parameters=count, device=str(device), threads=torch.get_num_threads(),
        cpu_affinity=sorted(os.sched_getaffinity(0)), physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified"),
        input_hashes=[tensor_hash(value) for value in inputs],
        workload=dict(batch=batch, length=length, chunk=chunk, continuing_updates=steps, seed=seed),
        artifacts=artifacts.records, artifact_budget_bytes=artifacts.limit, candidates=[],
        boundary="same frozen source and semantic config; one common real-update anchor; inference, repeated graph-enabled VJP, three continuing AdamW updates; public checkpoint into a distinct interpreter and one actual further update",
        observation="intrusive complete CPU tensor comparisons, original int64 times, support and finite checks; never throughput timing",
        objective="sum of signed cosine readout VJPs per chunk from check_training_paths; independent of the mean-square normal performance objective",
        exact_start="parameters, AdamW, complete private state, prior gradients and RNG",
        failure_policy="original FP32 thresholds unchanged; finite deviations and support/time failures remain separate; no automatic rounding attribution")
    def persist():
        record["retained_artifact_bytes"] = artifacts.used()
        atomic(directory/"result.json", record)
        progress(record)
    persist()
    try:
        torch.manual_seed(seed)
        model = TideModel(base).to(device)
        optimizer = placement_adamw(model, lr=1e-4, foreach=False)
        warm, state, work = observe(model, optimizer, None, inputs[0], "fp32", [device], chunk, update=True)
        anchor, extra = training_data(model, optimizer, state), extra_anchor(model, device)
        counts = {optimizer_step_count(slot, name) for name, slot in anchor["optimizer"]["state"].items()}
        if not anchor["optimizer"]["state"] or counts != {1}:
            raise RuntimeError("common anchor needs exactly one real connected AdamW update")
        artifacts.save(directory/"anchor.pt", dict(training=anchor, extra=extra, inputs=inputs,
            config=base.to_dict(), source=record["source"]), "retained common real-update anchor")
        record.update(anchor=screen(anchor), extra_anchor=screen(extra), warmup=screen(warm),
                      anchor_work=work, anchor_optimizer_steps=sorted(counts))
        del model, optimizer, state, warm
        reference_paths = []
        for name in candidates:
            case_dir = directory/name; case_dir.mkdir()
            model, optimizer, state = build_candidate(base, name, anchor, [device], "fp32", "auto", overrides=variants[name])
            restore_extra(model, extra, device)
            case = dict(name=name, config=model.config.to_dict(), state="running", checks={}, observations={}, updates=[])
            record["candidates"].append(case)
            case["checks"]["fixed_start"] = compare(training_data(model, optimizer, state), anchor, TOLERANCE)
            case["checks"]["gradient_rng_start"] = compare(extra_anchor(model, device), extra, TOLERANCE)
            if not all(row["comparison"]["exact"] for row in case["checks"].values()):
                raise RuntimeError("execution candidate did not start at the exact common anchor")
            own_paths = []
            def comparison_to_file(observation, path):
                expected = torch.load(path, map_location="cpu", weights_only=False)
                checked = compare(observation, expected, TOLERANCE)
                del expected
                return checked
            for phase in ("inference", "vjp1", "vjp2"):
                observed, _, work = observe(model, optimizer, state, inputs[1], "fp32", [device], chunk,
                                            grad=phase != "inference")
                row = case["observations"][phase] = dict(summary=screen(observed), work=work)
                if name != "baseline":
                    reference = directory/"baseline"/("vjp1.pt" if phase == "vjp2" else phase+".pt")
                    case["checks"][phase+"_vs_baseline"] = comparison_to_file(observed, reference)
                if phase == "vjp1":
                    inferred = torch.load(case_dir/"inference.pt", map_location="cpu", weights_only=False)
                    case["checks"]["inference_vs_grad_forward"] = compare(observed["forward"], inferred["forward"], TOLERANCE)
                    del inferred
                if phase == "vjp2":
                    case["checks"]["same_owner_vjp_repeat"] = comparison_to_file(observed, case_dir/"vjp1.pt")
                    if classified(case["checks"])["state"] != "within_threshold":
                        artifacts.save(case_dir/"vjp2.pt", observed, "retained failed repeated VJP witness")
                else:
                    path = case_dir/(phase+".pt")
                    artifacts.save(path, observed, "complete forward/VJP comparison witness")
                    (reference_paths if name == "baseline" else own_paths).append(path)
                if phase != "inference":
                    row["none_parameter_gradients"] = [key for key, value in model.named_parameters() if value.grad is None]
                del observed
                persist()
            case["checks"]["parameters_optimizer_state_unchanged_after_vjp"] = compare(training_data(model, optimizer, state), anchor, TOLERANCE)
            for index in range(steps):
                updated, state, work = observe(model, optimizer, state, inputs[index+1], "fp32", [device], chunk, update=True)
                row = dict(index=index, summary=screen(updated), work=work,
                           optimizer_update=updated["optimizer_update"])
                path = case_dir/f"update-{index}.pt"
                if name == "baseline":
                    artifacts.save(path, updated, "complete baseline continuing-update comparison witness")
                    reference_paths.append(path)
                else:
                    row["vs_baseline"] = comparison_to_file(updated, directory/"baseline"/path.name)
                    if classified([row["vs_baseline"]])["state"] != "within_threshold":
                        artifacts.save(path, updated, "retained failed continuing-update witness")
                case["updates"].append(row)
                del updated
                persist()
            case["restore"] = check_restore(model, optimizer, state, inputs[steps+1], case_dir, artifacts,
                device=device, chunk=chunk, timeout=restore_timeout)
            checks = list(case["checks"].values())+[row["vs_baseline"] for row in case["updates"] if "vs_baseline" in row]
            checks += list(case["restore"]["checks"].values())
            exact = [case["checks"][key] for key in ("fixed_start", "gradient_rng_start", "parameters_optimizer_state_unchanged_after_vjp")]
            case["summary"] = classified(checks, exact=exact)
            case["state"] = aggregate([case["summary"]["state"], case["restore"]["state"]])
            persist()  # durable successful comparisons precede retirement
            if case["state"] == "within_threshold":
                for path in own_paths:
                    artifacts.retire(path)
            del model, optimizer, state
            gc.collect()
            persist()
        record["state"] = aggregate([case["state"] for case in record["candidates"]])
        record["qualified"] = record["state"] == "within_threshold"
        persist()
        if record["state"] == "within_threshold":
            for path in reference_paths:
                artifacts.retire(path)
        persist()
        return record
    except Exception as error:
        record.update(state="execution_failed", qualified=False, error=repr(error))
        persist()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidates", choices=VARIANTS, nargs="+", default=["baseline", "packing"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--artifact-budget-gib", type=float, default=20.)
    parser.add_argument("--restore-timeout-seconds", type=float, default=1800.)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-bundle", type=Path)
    parser.add_argument("--resume-output", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 64:
        parser.error("threads must be 1..64")
    if not math.isfinite(args.restore_timeout_seconds) or not 1 <= args.restore_timeout_seconds <= 7200:
        parser.error("restore timeout must be 1..7200 seconds")
    if Path(tide.__file__).resolve().parent != Path(__file__).resolve().parents[1]/"src"/"tide":
        parser.error("Tide must resolve from this frozen checkout")
    torch.set_num_threads(args.threads)
    if args.resume_bundle is not None:
        if args.resume_output is None or args.resume_output.exists():
            parser.error("fresh recovery requires a new output path")
        try:
            record = worker(args.resume_bundle, args.resume_output, args.threads)
        except Exception as error:
            record = dict(state="execution_failed", error=repr(error), pid=os.getpid())
            atomic(args.resume_output, record)
            traceback.print_exc()
    else:
        if args.config is None or args.output_dir is None:
            parser.error("qualification requires config and a new output directory")
        configure_device(args.device, args.threads)
        config = ModelConfig.from_dict(json.loads(args.config.read_text()))
        try:
            record = run(config, directory=args.output_dir, candidates=args.candidates, device=args.device,
                batch=args.batch, length=args.length, chunk=args.chunk, steps=args.steps, seed=args.seed,
                artifact_budget_gib=args.artifact_budget_gib, restore_timeout=args.restore_timeout_seconds)
        except Exception:
            traceback.print_exc()
            return 2
    print(json.dumps(dict(state=record["state"], qualified=record["state"] == "within_threshold")))
    return 0 if record["state"] in COMPLETED else 2


if __name__ == "__main__":
    raise SystemExit(main())
