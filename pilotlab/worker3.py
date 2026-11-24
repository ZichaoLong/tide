"""Token-bounded third-round training with immutable contracts and stage gates.

This worker never decides to enter another stage or changes a recipe after a
failure.  A target counts only completed optimizer updates.  Checkpoint loading
and numerical trajectory reproducibility are deliberately separate results.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback
import uuid

from .io import append_json, atomic_json, sha256
from .worker import (StreamBatch, atomic_torch_save, canonical_digest, cpu_snapshot,
                     device_memory as base_device_memory, digest_tree, leaves, restore_rng, rng_state,
                     synchronize, tree_map)

SCHEMA = "pilot-r3-checkpoint-v1"
STOP = False
MI = 2**20


class CheckpointCorrupt(ValueError):
    pass


def is_capacity_error(error: Exception, device) -> bool:
    """Recognize allocator OOM only; generic runtime/operator errors stay fatal."""
    import torch
    if isinstance(error, torch.OutOfMemoryError):
        return True
    return (str(device).startswith("npu") and isinstance(error, RuntimeError)
            and re.search(r"\bNPU out of memory\b", str(error), re.IGNORECASE) is not None)


def device_memory(device):
    result = base_device_memory(device)
    if device.type == "npu":
        import torch
        free, total = torch.npu.mem_get_info(device)
        result.update(npu_total_memory_bytes=int(total), npu_free_memory_bytes=int(free),
                      npu_memory_sample_unix=time.time(),
                      npu_device_memory_scope="device-wide mem_get_info; sampled before this worker releases its model/cache")
    return result


def stop_requested(signum, frame):
    global STOP
    STOP = True


def is_formal(task: dict) -> bool:
    return str(task.get("phase", "")).lower().startswith("b") or task.get("phase") in ("formal", "repeat")


def validate_source_identity(task: dict, data: Path) -> dict:
    """Check declared content and runtime metadata without importing a device."""
    identity = task.get("source_identity")
    if not isinstance(identity, dict) or identity.get("schema") != "pilot-r3-source-v1":
        raise ValueError("task requires pilot-r3-source-v1 source_identity")
    files = identity.get("source_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source_identity requires explicit source_files")
    declared = {str(Path(name).resolve()): value for name, value in files.items()}
    package = Path(__file__).resolve().parent
    for name in ("worker3.py", "data3.py", "models3.py", "models.py", "worker.py", "io.py"):
        if str(package / name) not in declared:
            raise ValueError(f"source identity omits executed module: {name}")
    foundation = importlib.util.find_spec("tide")
    if foundation is None or not foundation.submodule_search_locations or len(foundation.submodule_search_locations) != 1:
        raise ValueError("one identifiable Tide source directory is required")
    foundation_root = Path(next(iter(foundation.submodule_search_locations))).resolve()
    for path in foundation_root.rglob("*.py"):
        if str(path.resolve()) not in declared:
            raise ValueError(f"source identity omits foundation module: {path.name}")
    for name, expected in declared.items():
        if sha256(name) != expected:
            raise ValueError(f"source identity mismatch: {name}")
    if identity.get("data_manifest_sha256") != sha256(data / "manifest.json"):
        raise ValueError("data manifest identity mismatch")
    for name, expected in identity.get("runtime_packages", {}).items():
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            observed = None
        if observed != expected:
            raise ValueError(f"runtime package identity mismatch: {name}")
    if identity.get("python_version", sys.version.split()[0]) != sys.version.split()[0]:
        raise ValueError("Python runtime identity mismatch")
    return identity


def validate_gate(task: dict, task_file_sha256: str) -> dict | None:
    if not is_formal(task):
        if task.get("allow_confirmation"):
            raise ValueError("confirmation is not available during stage A or preflight")
        return None
    path = task.get("gate_file") or task.get("stage_b_gate")
    if not path:
        raise ValueError("stage B requires a passed, immutable gate record")
    gate_path = Path(path).resolve()
    gate = json.loads(gate_path.read_text())
    if gate.get("schema") != "pilot-stage-b-gate-v1" or gate.get("status") != "passed":
        raise ValueError("stage B gate has not passed")
    if gate.get("source_identity") != task.get("source_identity"):
        raise ValueError("stage B gate source identity mismatch")
    if task["id"] not in gate.get("allowed_task_ids", []):
        raise ValueError("task is not in the frozen stage B allowed list")
    if gate.get("task_sha256", {}).get(task["id"]) != task_file_sha256:
        raise ValueError("stage B task file hash mismatch")
    for kind in ("manifest", "report"):
        artifact = gate.get(kind + "_path")
        if not artifact or sha256(artifact) != gate.get(kind + "_sha256"):
            raise ValueError(f"stage B gate {kind} hash mismatch")
    if task.get("allow_confirmation"):
        authorization = task.get("confirmation_authorization", {})
        path = authorization.get("manifest_path")
        if (not path or sha256(path) != authorization.get("manifest_sha256")
                or task["id"] not in authorization.get("allowed_task_ids", [])):
            raise ValueError("confirmation requires the frozen formal/repeat list authorization")
    return {"path": str(gate_path), "sha256": sha256(gate_path), "status": "passed"}


def validate_task(task: dict) -> None:
    for name in ("id", "model_id", "phase", "model", "seed", "source_identity"):
        if name not in task:
            raise ValueError(f"missing task field: {name}")
    for name in ("batch_size", "sequence_length", "stream_chunks", "cpu_threads", "schedule_tokens"):
        value = task.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(task.get("target_tokens")) is not int or task["target_tokens"] < 0:
        raise ValueError("target_tokens must be a nonnegative integer")
    quantum = task["batch_size"] * task["sequence_length"]
    if task["target_tokens"] % quantum:
        raise ValueError("target_tokens must be an exact multiple of one optimizer update")
    if task.get("gradient_accumulation", 1) != 1:
        raise ValueError("this stream contract requires one independent batch per update")
    if type(task.get("backward_range_retry", True)) is not bool:
        raise ValueError("backward_range_retry must be boolean")
    if type(task["seed"]) is not int or task["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if task.get("precision", "fp32") not in ("fp32", "bf16"):
        raise ValueError("only declared FP32 or BF16 policies are supported")
    for name, default in (("lr", None), ("weight_decay", 0.01), ("grad_clip", 1.0),
                          ("auxiliary_loss_weight", 1.0), ("lr_floor_ratio", 0.1)):
        value = task.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {name}")
    if task.get("lr_floor_ratio", 0.1) > 1 or task.get("grad_clip", 1.0) <= 0:
        raise ValueError("invalid clipping/schedule bounds")
    if type(task.get("warmup_tokens", 65536)) is not int or not 0 <= task.get("warmup_tokens", 65536) <= task["schedule_tokens"]:
        raise ValueError("warmup_tokens is outside the schedule")
    if task.get("eval_tokens", 16384) % task["sequence_length"]:
        raise ValueError("eval_tokens must be a multiple of sequence_length")


def contract(task: dict, device: str) -> dict:
    return {"schema": "pilot-r3-trajectory-contract-v1", "model_id": task["model_id"],
            "model": copy.deepcopy(task["model"]), "init_recipe": task.get("init_recipe", "default"),
            "seed": task["seed"], "cpu_threads": task["cpu_threads"],
            "backend": str(device).split(":")[0], "precision": task.get("precision", "fp32"),
            "deterministic": task.get("deterministic", False),
            "batch_size": task["batch_size"], "sequence_length": task["sequence_length"],
            "stream_chunks": task["stream_chunks"], "lr": task["lr"],
            "schedule_tokens": task["schedule_tokens"], "warmup_tokens": task.get("warmup_tokens", 65536),
            "lr_floor_ratio": task.get("lr_floor_ratio", 0.1),
            "weight_decay": task.get("weight_decay", 0.01), "grad_clip": task.get("grad_clip", 1.0),
            "auxiliary_loss_weight": task.get("auxiliary_loss_weight", 1.0),
            "optimizer": "AdamW(betas=(0.9,0.999),eps=1e-8,foreach=False)",
            "state_policy": "document-bounded stream, detach each completed update"}


def learning_rate(task: dict, tokens_after_update: int) -> float:
    warmup = task.get("warmup_tokens", 65536)
    if warmup and tokens_after_update < warmup:
        return task["lr"] * tokens_after_update / warmup
    progress = min(1.0, max(0.0, (tokens_after_update - warmup) / max(1, task["schedule_tokens"] - warmup)))
    floor = task.get("lr_floor_ratio", 0.1)
    return task["lr"] * (floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2)


def initialize(device: str, task: dict):
    import random
    import numpy as np
    import torch
    torch.set_num_threads(task["cpu_threads"])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    if device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(device)
    elif device != "cpu":
        raise ValueError("worker requires cpu or npu:N")
    torch.use_deterministic_algorithms(bool(task.get("deterministic", False)))
    torch.manual_seed(task["seed"])
    random.seed(task["seed"])
    np.random.seed(task["seed"] % 2**32)
    return torch.device(device)


def optimizer_for(model, task):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=task["lr"], weight_decay=task.get("weight_decay", 0.01),
                             betas=(0.9, 0.999), eps=1e-8, foreach=False)


def tensor_stats(tensor) -> dict:
    import torch
    tensor = tensor.detach()
    count = tensor.numel()
    if not count:
        return {"elements": 0, "finite": True, "max_abs": 0.0, "rms": 0.0}
    if not tensor.is_floating_point():
        return {"elements": count, "dtype": str(tensor.dtype), "finite": True}
    values = tensor.float()
    finite = bool(torch.isfinite(values).all())
    if not finite:
        return {"elements": count, "finite": False, "max_abs": None, "rms": None}
    maximum = float(values.abs().max())
    rms = float((values / maximum).square().mean().sqrt()) * maximum if maximum else 0.0
    return {"elements": count, "finite": True, "max_abs": maximum, "rms": rms}


def state_stats(state) -> dict:
    import torch
    tensors = [(name, value) for name, value in leaves(state) if isinstance(value, torch.Tensor)]
    records = {name: tensor_stats(value) for name, value in tensors[:64]}
    return {"tensor_count": len(tensors), "sampled_tensor_count": len(records),
            "sampling": "first 64 tensor leaves; not a complete-state finite proof", "tensors": records,
            "observed_finite": all(record["finite"] for record in records.values())}


class ActivationSample:
    def __init__(self, model):
        self.records, self.handles = {}, []
        for name, module in model.named_modules():
            if len(self.handles) >= 128:
                break
            if any(True for _ in module.parameters(recurse=False)):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def record(module, inputs, output):
            import torch
            value = output if isinstance(output, torch.Tensor) else next(
                (leaf for _, leaf in leaves(output) if isinstance(leaf, torch.Tensor)), None)
            if value is not None:
                observed = tensor_stats(value)
                prior = self.records.get(name)
                observed["calls"] = (prior or {}).get("calls", 0) + 1
                if prior is not None:
                    observed["finite"] &= prior["finite"]
                    if observed["finite"]:
                        observed["max_abs"] = max(observed.get("max_abs", 0), prior.get("max_abs", 0))
                self.records[name] = observed
        return record

    def close(self):
        for handle in self.handles:
            handle.remove()


def train_update(model, optimizer, inputs, targets, state, device, task, tokens, *, sample=False):
    import torch
    from .models import detach_state
    from tide.precision import precision_context
    try:
        from tide.scaled_arithmetic import GradientRangeError
        range_errors = (GradientRangeError,)
    except ImportError:
        # Older frozen instances have no typed retryable error.
        range_errors = ()
    model.train()
    after = tokens + targets.numel()
    lr = learning_rate(task, after)
    for group in optimizer.param_groups:
        group["lr"] = lr
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()} if sample else {}
    divisors = (1, 8, 16) if range_errors and task.get("backward_range_retry", True) else (1,)
    # Each supported model takes its recurrent state as a functional input.
    # The stream has already delivered this batch. Replay the same RNG and
    # observations; no optimizer update takes place until backward succeeds.
    saved_rng = rng_state(device) if len(divisors) > 1 else None
    counters = getattr(getattr(model, "core", None), "_pool_counts", None)
    saved_counts = vars(counters).copy() if counters is not None else {}
    for divisor in divisors:
        optimizer.zero_grad(set_to_none=True)
        activation = ActivationSample(model) if sample else None
        try:
            with precision_context(device, task.get("precision", "fp32")):
                output = model(inputs.to(device), state, record_work=sample)
                nll = torch.nn.functional.cross_entropy(output.logits.float().flatten(0, 1), targets.to(device).flatten())
                auxiliary = getattr(output, "auxiliary_loss", None)
                objective = nll if auxiliary is None else nll + task.get("auxiliary_loss_weight", 1.0) * auxiliary
            if not bool(torch.isfinite(objective)):
                raise ArithmeticError("nonfinite training objective")
            try:
                (objective if divisor == 1 else objective / divisor).backward()
            except range_errors:
                optimizer.zero_grad(set_to_none=True)
                if saved_rng is not None:
                    restore_rng(saved_rng, device)
                    if counters is not None:
                        vars(counters).update(saved_counts)
                if divisor == divisors[-1]:
                    raise
                del output, objective, auxiliary, nll
                continue
            break
        finally:
            if activation is not None:
                activation.close()
    if divisor != 1:
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(divisor)
    gradients = ({name: None if p.grad is None else tensor_stats(p.grad)
                  for name, p in model.named_parameters()} if sample else {})
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), task.get("grad_clip", 1.0),
                                        error_if_nonfinite=True, foreach=False)
    optimizer.step()
    health = None
    if sample:
        updates = {}
        for name, parameter in model.named_parameters():
            current = tensor_stats(parameter)
            update = tensor_stats(parameter.detach() - before[name])
            initial = tensor_stats(before[name])
            update["relative_rms"] = (update["rms"] / initial["rms"]
                                       if update.get("rms") is not None and initial.get("rms") else None)
            update["parameter_finite"] = current["finite"]
            updates[name] = update
        health = {"tokens": after, "activations": activation.records, "gradients_before_clip": gradients,
                  "updates": updates, "logits": tensor_stats(output.logits), "state": state_stats(output.state),
                  "grad_norm_before_clip": float(norm), "clip_threshold": task.get("grad_clip", 1.0)}
        health["observed_finite"] = (health["logits"]["finite"] and health["state"]["observed_finite"]
            and all(record["finite"] for record in activation.records.values())
            and all(record is None or record["finite"] for record in gradients.values())
            and all(record["finite"] and record["parameter_finite"] for record in updates.values()))
        if not health["observed_finite"]:
            raise ArithmeticError("nonfinite sampled activation, parameter, update, or private state")
    return {"loss": float(nll.detach()), "objective": float(objective.detach()), "grad_norm": float(norm),
            "backward_divisor": divisor,
            "lr": lr, "state": detach_state(output.state), "health": health, "work": output.work}


def evaluate(model, store, task, device, deadline, *, split="selection", tokens=None):
    import numpy as np
    import torch
    from tide.precision import precision_context
    count = task.get("eval_tokens", 16384) if tokens is None else tokens
    if count <= 0:
        return {"status": "not_requested", "tokens": 0, "split": split}
    length = task["sequence_length"]
    if count % length:
        raise ValueError("evaluation exposure must be a whole number of sequences")
    saved_rng = rng_state(device)
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(17729 if split == "selection" else 92917)
    done, loss_sum = 0, 0.0
    start = time.monotonic()
    try:
        with torch.no_grad(), precision_context(device, task.get("precision", "fp32")):
            while done < count:
                if STOP or time.time() >= deadline:
                    break
                batch = min(task["batch_size"], 4, (count - done) // length)
                sample = store.sample_batch(split, batch, length, rng)
                inputs = torch.from_numpy(sample[:, :-1].copy()).to(device)
                targets = torch.from_numpy(sample[:, 1:].copy()).to(device)
                logits = model(inputs).logits.float()
                value = float(torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten()))
                if not math.isfinite(value):
                    raise ArithmeticError("nonfinite evaluation NLL")
                done += targets.numel()
                loss_sum += value * targets.numel()
        synchronize(device)
    finally:
        restore_rng(saved_rng, device)
        model.train(was_training)
    result = {"status": "completed" if done == count else "budget_exhausted", "tokens": done,
              "requested_tokens": count, "split": split, "seconds": time.monotonic() - start,
              "state_policy": "fixed article windows from cold state; training-state activity measured separately"}
    if done:
        result["nll"] = loss_sum / done
    return result


def _state(payload):
    return {key: payload[key] for key in ("model", "optimizer", "private_state", "stream", "rng", "tokens", "steps")}


def checkpoint_payload(model, optimizer, state, stream, task, device, tokens, steps, initialization):
    payload = {"schema": SCHEMA, "source_identity": copy.deepcopy(task["source_identity"]),
               "contract": contract(task, str(device)), "task_id": task["id"],
               "model": cpu_snapshot(model.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
               "private_state": cpu_snapshot(state), "stream": copy.deepcopy(stream.state_dict()),
               "rng": rng_state(device), "tokens": tokens, "steps": steps,
               "initialization": copy.deepcopy(initialization),
               "boundary": "after a complete optimizer update, detached private state"}
    payload["state_sha256"] = digest_tree(_state(payload))
    return payload


def save_checkpoint(output: Path, payload: dict) -> dict:
    directory = output / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    index_path = output / "checkpoint.json"
    old = json.loads(index_path.read_text()) if index_path.exists() else {}
    filename = f"step-{payload['steps']:09d}-{uuid.uuid4().hex[:10]}.pt"
    path = directory / filename
    atomic_torch_save(path, payload)
    current = {"path": str(path.resolve()), "sha256": sha256(path), "tokens": payload["tokens"],
               "steps": payload["steps"], "state_sha256": payload["state_sha256"], "size_bytes": path.stat().st_size}
    atomic_json(index_path, {"schema": "pilot-r3-checkpoint-generations-v1", "current": current,
                             "previous": old.get("current")})
    previous = old.get("previous")
    if previous:
        retired = Path(previous["path"]).resolve()
        if retired.parent == directory.resolve() and retired != path.resolve():
            retired.unlink(missing_ok=True)
    return current


def _read_checkpoint(path: Path, expected_sha=None):
    import torch
    if expected_sha and sha256(path) != expected_sha:
        raise CheckpointCorrupt("checkpoint file SHA256 mismatch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise CheckpointCorrupt(f"checkpoint decoding failed: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise CheckpointCorrupt("unsupported checkpoint payload")
    try:
        observed = digest_tree(_state(payload))
    except (KeyError, TypeError) as error:
        raise CheckpointCorrupt("checkpoint state is incomplete") from error
    if observed != payload.get("state_sha256"):
        raise CheckpointCorrupt("checkpoint state digest mismatch")
    return payload


def validate_checkpoint(payload, task, device):
    if payload["source_identity"] != task["source_identity"]:
        raise ValueError("checkpoint source identity mismatch")
    if payload["contract"] != contract(task, str(device)):
        raise ValueError("checkpoint trajectory contract mismatch")
    quantum = task["batch_size"] * task["sequence_length"]
    if (type(payload["tokens"]) is not int or type(payload["steps"]) is not int
            or payload["tokens"] != payload["steps"] * quantum or payload["tokens"] > task["target_tokens"]):
        raise ValueError("checkpoint exposure does not fit the requested cumulative target")
    block, cursor = payload["stream"]["block"], payload["stream"]["cursor"]
    if payload["steps"]:
        if (block is None or tuple(block.shape) != (task["batch_size"], task["sequence_length"] * task["stream_chunks"] + 1)
                or not isinstance(cursor, int) or not 0 < cursor <= task["sequence_length"] * task["stream_chunks"]
                or cursor % task["sequence_length"]):
            raise ValueError("checkpoint stream state violates the trajectory contract")


def load_checkpoint(reference, task, device, expected_sha=None):
    path = Path(reference).resolve()
    if path.is_dir():
        path = path / "checkpoint.json"
    candidates = []
    if path.suffix == ".json":
        index = json.loads(path.read_text())
        candidates = [record for record in (index.get("current"), index.get("previous")) if record]
    else:
        candidates = [{"path": str(path), "sha256": expected_sha}]
        index_path = path.parent.parent / "checkpoint.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if expected_sha is None:
                for generation in (index.get("current"), index.get("previous")):
                    if generation and Path(generation["path"]).resolve() == path:
                        candidates[0]["sha256"] = generation["sha256"]
                        break
            previous = index.get("previous")
            if previous and previous["path"] != str(path):
                candidates.append(previous)
    rejected = []
    for record in candidates:
        try:
            payload = _read_checkpoint(Path(record["path"]), record.get("sha256"))
        except (CheckpointCorrupt, OSError) as error:
            rejected.append({"path": record["path"], "reason": str(error)})
            continue
        validate_checkpoint(payload, task, device)
        return payload, {"path": record["path"], "sha256": sha256(record["path"]),
                         "fallback_used": bool(rejected), "rejected_generations": rejected}
    raise CheckpointCorrupt("no intact checkpoint generation: " + json.dumps(rejected))


def restore_owners(payload, model, optimizer, store, task, device):
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    state = tree_map(payload["private_state"], lambda tensor: tensor.to(device))
    stream = StreamBatch(store, task, payload["stream"])
    restore_rng(payload["rng"], device)
    actual = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "private_state": state,
              "stream": stream.state_dict(), "rng": rng_state(device),
              "tokens": payload["tokens"], "steps": payload["steps"]}
    if digest_tree(actual) != payload["state_sha256"]:
        raise ValueError("loaded model/optimizer/private/RNG/stream state is not exact")
    return state, stream


def verify_saved_checkpoint(record, task, store, device):
    """New host owners verify state decoding; this is not a trajectory test."""
    import torch
    from .models3 import build_model
    payload = _read_checkpoint(Path(record["path"]), record["sha256"])
    validate_checkpoint(payload, task, device)
    saved_rng = rng_state(device)
    try:
        model = build_model(task["model"], store.vocab_size, init_recipe=task.get("init_recipe", "default"))
        optimizer = optimizer_for(model, task)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "private_state": payload["private_state"], "stream": payload["stream"],
                 "rng": payload["rng"], "tokens": payload["tokens"], "steps": payload["steps"]}
        if digest_tree(state) != payload["state_sha256"]:
            raise ValueError("fresh host model/optimizer owners did not restore exactly")
    finally:
        restore_rng(saved_rng, device)
    return {"status": "exact", "scope": "file/state digest and fresh host model/optimizer owners; not numerical trajectory",
            "tokens": payload["tokens"], "steps": payload["steps"], "state_sha256": payload["state_sha256"]}


def _heartbeat(output, task, tokens, steps, state, **extra):
    atomic_json(output / "heartbeat.json", {"schema": "pilot-r3-worker-heartbeat-v1", "task_id": task["id"],
        "phase": task["phase"], "pid": os.getpid(), "updated_unix": time.time(), "status": state,
        "tokens": tokens, "steps": steps, "target_tokens": task["target_tokens"], **extra})


def execute(args):
    import torch
    from .data3 import DocumentStore3
    from .models3 import build_model
    from .worker import causality_check
    global STOP
    STOP = False
    output, data = Path(args.output).resolve(), Path(args.data).resolve()
    task_path = Path(args.task).resolve()
    task = json.loads(task_path.read_text())
    task_sha = sha256(task_path)
    start = time.time()
    output.mkdir(parents=True, exist_ok=True)
    old_task_path = output / "task.json"
    if old_task_path.exists() and json.loads(old_task_path.read_text()) != task:
        raise ValueError("output already belongs to a different immutable task")
    old_result = output / "result.json"
    if old_result.exists():
        previous = json.loads(old_result.read_text())
        if previous.get("status") == "completed":
            raise FileExistsError("completed task must not be executed again")
    if not old_task_path.exists():
        atomic_json(old_task_path, task)
    result = {"schema": "pilot-r3-worker-result-v1", "id": task.get("id"), "model_id": task.get("model_id"),
              "phase": task.get("phase"), "status": "running", "task_sha256": task_sha,
              "source_identity": task.get("source_identity"), "started_unix": start,
              "training": {"tokens": 0, "steps": 0, "seconds": 0.0, "segment_tokens": 0, "segment_steps": 0},
              "health": {"eligible": False, "blocking_reasons": ["not_evaluated"]},
              "recovery": {"checkpoint_load": {"status": "not_tested"},
                           "numeric_trajectory": {"status": "not_tested", "atol": 1e-5, "rtol": 1e-4}},
              "initialization": {"status": "not_measured"}, "work": [], "checkpoint": None}
    atomic_json(old_result, result)
    model = optimizer = state = stream = device = store = None
    tokens = steps = from_tokens = from_steps = 0
    train_seconds = 0.0
    evaluated_tokens = 0
    health_samples = []
    last_inputs = last_targets = None
    update_in_progress = False
    try:
        validate_task(task)
        validate_source_identity(task, data)
        result["stage_gate"] = validate_gate(task, task_sha)
        device = initialize(args.device, task)
        store = DocumentStore3(data, allow_confirmation=bool(task.get("allow_confirmation", False)))
        model = build_model(task["model"], store.vocab_size, init_recipe=task.get("init_recipe", "default")).to(device)
        optimizer = optimizer_for(model, task)
        reference = getattr(args, "resume_checkpoint", None) or task.get("resume_checkpoint")
        if reference:
            # A scheduler retry names this task's own newer checkpoint.  The
            # frozen task may still name an earlier rung and its different SHA.
            expected_sha = None if getattr(args, "resume_checkpoint", None) else task.get("resume_checkpoint_sha256")
            payload, recovered = load_checkpoint(reference, task, device, expected_sha)
            state, stream = restore_owners(payload, model, optimizer, store, task, device)
            tokens, steps = payload["tokens"], payload["steps"]
            result["initialization"] = payload["initialization"]
            result["resume"] = {**recovered, "from_tokens": tokens, "from_steps": steps,
                                 "runtime_override": bool(getattr(args, "resume_checkpoint", None)),
                                 "state_reused": state is not None, "source_task_id": payload["task_id"],
                                 "state_sha256": payload["state_sha256"],
                                 "contract_sha256": canonical_digest(payload["contract"]),
                                 "checkpoint_load": {"status": "exact",
                                     "scope": "actual device model/optimizer/private/RNG/stream before continuation"}}
            result["recovery"]["checkpoint_load"] = {"status": "exact", "scope": "actual device owners, RNG and data stream",
                                                        "state_sha256": payload["state_sha256"]}
            del payload
        else:
            stream = StreamBatch(store, task)
            result["initialization"] = {"status": "constructed_single_seed", "seed": task["seed"],
                "recipe": task.get("init_recipe", "default"), "parameter_sha256": digest_tree(model.state_dict()),
                "resolved_recipe": getattr(model, "initialization_recipe", task.get("init_recipe", "default")),
                "multi_seed_qualification": "controller must aggregate prescribed independent seeds"}
        from_tokens, from_steps = tokens, steps
        result["device"] = str(device)
        result["physical_visibility"] = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        result["parameters"] = sum(p.numel() for p in model.parameters())
        result["readout_policy"] = model.readout_policy
        _heartbeat(output, task, tokens, steps, "initial_checks")
        reserve = float(task.get("closing_reserve_seconds", 120))
        training_deadline = args.deadline_unix - reserve
        if time.time() < training_deadline and not STOP:
            saved_rng = rng_state(device)
            try:
                result["causality"] = causality_check(model, store, task, device)
            finally:
                restore_rng(saved_rng, device)
            result["validation_initial"] = evaluate(model, store, task, device, training_deadline)
            evaluated_tokens += result["validation_initial"]["tokens"]
            if result["validation_initial"]["status"] != "completed":
                training_deadline = time.time()
        sample_interval = int(task.get("health_interval_tokens", 262144))
        eval_interval = int(task.get("eval_interval_tokens", 1048576))
        checkpoint_seconds = float(task.get("checkpoint_interval_seconds", 120))
        last_saved = time.monotonic()
        next_eval = ((tokens // eval_interval) + 1) * eval_interval if eval_interval > 0 else math.inf
        next_sample = ((tokens // sample_interval) + 1) * sample_interval if sample_interval > 0 else math.inf
        if device.type == "npu":
            torch.npu.reset_peak_memory_stats(device)
        while tokens < task["target_tokens"] and not STOP and time.time() < training_deadline:
            synchronize(device)
            tick = time.monotonic()
            inputs, targets, reset = stream.next()
            last_inputs, last_targets = inputs, targets
            if reset:
                state = None
            sample = (steps == from_steps or tokens + targets.numel() >= next_sample
                      or tokens + targets.numel() == task["target_tokens"])
            before_rng = digest_tree(rng_state(device)) if task.get("record_step_fingerprints", False) else None
            update_in_progress = True
            update = train_update(model, optimizer, inputs, targets, state, device, task, tokens, sample=sample)
            synchronize(device)
            elapsed = time.monotonic() - tick
            state = update["state"]
            steps += 1
            tokens += targets.numel()
            train_seconds += elapsed
            update_in_progress = False
            row = {"step": steps, "tokens": tokens, "loss": update["loss"], "objective": update["objective"],
                   "grad_norm": update["grad_norm"], "lr": update["lr"], "state_reset": reset,
                   "seconds": elapsed, "sampled_diagnostics": sample, "time_unix": time.time()}
            if update["backward_divisor"] != 1:
                row["backward_divisor"] = update["backward_divisor"]
            if before_rng is not None:
                row.update(input_sha256=digest_tree((inputs, targets)), rng_before_sha256=before_rng,
                           rng_after_sha256=digest_tree(rng_state(device)))
            append_json(output / "steps.jsonl", row)
            if sample:
                health_samples.append(update["health"])
                append_json(output / "health.jsonl", update["health"])
                if update["work"] is not None:
                    observation = {"tokens": tokens, "step": steps, "work": update["work"]}
                    result["work"].append(observation)
                    append_json(output / "work.jsonl", observation)
                next_sample = tokens + max(sample_interval, 1)
            _heartbeat(output, task, tokens, steps, "training", loss=update["loss"], seconds=train_seconds)
            if (time.monotonic() - last_saved >= checkpoint_seconds or steps == from_steps + 1
                    or task.get("checkpoint_interval_steps", 0) and steps % task["checkpoint_interval_steps"] == 0):
                saved = checkpoint_payload(model, optimizer, state, stream, task, device, tokens, steps, result["initialization"])
                result["checkpoint"] = save_checkpoint(output, saved)
                del saved
                last_saved = time.monotonic()
            if tokens >= next_eval and tokens < task["target_tokens"] and not STOP:
                validation = evaluate(model, store, task, device, training_deadline)
                evaluated_tokens += validation["tokens"]
                append_json(output / "validation.jsonl", {"tokens": tokens, **validation})
                next_eval += eval_interval
                if validation["status"] != "completed":
                    break
        _heartbeat(output, task, tokens, steps, "saving_checkpoint")
        payload = checkpoint_payload(model, optimizer, state, stream, task, device, tokens, steps, result["initialization"])
        result["checkpoint"] = save_checkpoint(output, payload)
        del payload
        if time.time() < args.deadline_unix and not STOP:
            result["recovery"]["checkpoint_load"] = verify_saved_checkpoint(result["checkpoint"], task, store, device)
        reached = tokens == task["target_tokens"]
        result["validation_final"] = evaluate(model, store, task, device, args.deadline_unix)
        evaluated_tokens += result["validation_final"]["tokens"]
        if task.get("allow_confirmation") and reached and not STOP:
            result["confirmation_final"] = evaluate(model, store, task, device, args.deadline_unix,
                split="confirmation", tokens=task.get("confirmation_tokens", 131072))
            evaluated_tokens += result["confirmation_final"]["tokens"]
        final_eval = result["validation_final"]["status"] == "completed"
        confirmation_ok = not task.get("allow_confirmation") or result.get("confirmation_final", {}).get("status") == "completed"
        blocking = []
        if not reached:
            blocking.append("declared_training_exposure_not_reached")
        if not final_eval or not confirmation_ok:
            blocking.append("evaluation_incomplete")
        if result.get("causality", {}).get("within_tolerance") is not True:
            blocking.append("causality_not_verified")
        if not health_samples and tokens > from_tokens:
            blocking.append("training_health_not_sampled")
        if result["recovery"]["checkpoint_load"]["status"] != "exact":
            blocking.append("checkpoint_load_not_verified")
        core_gradients = [record for observation in health_samples
                          for name, record in observation["gradients_before_clip"].items()
                          if not name.startswith(("embedding.", "output_norm.")) and record is not None]
        core_signal = any(record.get("rms", 0) > 0 for record in core_gradients if record["finite"])
        if task["lr"] > 0 and tokens > from_tokens and not core_signal:
            blocking.append("no_nonzero_backbone_gradient_in_prescribed_samples")
        result["health"] = {"eligible": not blocking, "blocking_reasons": blocking,
                            "sample_count": len(health_samples),
                            "nonzero_backbone_gradient_observed": core_signal,
                            "scope": "observed single-trajectory diagnostics; prescribed seed groups assessed by controller"}
        health_failure = "no_nonzero_backbone_gradient_in_prescribed_samples" in blocking
        result["status"] = ("failed" if health_failure else "completed"
                            if not blocking and not STOP else "budget_exhausted")
        result["stop_reason"] = ("health_check_failed" if health_failure else "target_reached"
                                 if result["status"] == "completed" else "signal" if STOP else "deadline_or_incomplete_checks")
    except Exception as error:
        result["status"] = "failed"
        if is_capacity_error(error, device):
            result["failure_class"] = "capacity_error"
        result["stop_reason"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        result["health"] = {"eligible": False, "blocking_reasons": [str(error)], "sample_count": len(health_samples)}
        if model is not None and optimizer is not None and task.get("capture_failure", True):
            try:
                failure_path = output / "failure.pt"
                atomic_torch_save(failure_path, {"schema": "pilot-r3-failure-capture-v1", "task": task,
                    "error": str(error), "traceback": result["traceback"], "completed_tokens": tokens,
                    "completed_steps": steps, "update_in_progress": update_in_progress,
                    "model": cpu_snapshot(model.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
                    "private_state": cpu_snapshot(state), "inputs": last_inputs, "targets": last_targets,
                    "resumable": False})
                result["failure_capture"] = {"path": str(failure_path), "sha256": sha256(failure_path)}
            except Exception as capture_error:
                result["failure_capture_error"] = str(capture_error)
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        result["training"] = {"tokens": tokens, "steps": steps, "seconds": train_seconds,
            "from_tokens": from_tokens, "from_steps": from_steps, "segment_tokens": tokens - from_tokens,
            "segment_steps": steps - from_steps, "target_tokens": task.get("target_tokens"),
            "tokens_per_second": (tokens - from_tokens) / train_seconds if train_seconds else None,
            "timing_scope": "sampling, transfers, forward/backward, clipping, optimizer, synchronization and sampled diagnostics; excludes eval/checkpoint"}
        result["finished_unix"] = time.time()
        result["evaluation"] = {"tokens": evaluated_tokens,
                                "scope": "completed selection/confirmation loss evaluation; excludes causality probes"}
        result["wall_seconds"] = result["finished_unix"] - start
        if device is not None:
            try:
                result["device_memory"] = device_memory(device)
            except Exception as memory_error:
                result["device_memory"] = {"observation_error": str(memory_error)}
        atomic_json(old_result, result)
        if all(name in task for name in ("id", "phase", "target_tokens")):
            _heartbeat(output, task, tokens, steps, result["status"], stop_reason=result.get("stop_reason"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    result = execute(args)
    return 0 if result["status"] == "completed" else 2 if result["status"] == "budget_exhausted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
