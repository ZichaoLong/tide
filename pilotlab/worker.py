"""Bounded language-model training, inference and fresh-process continuation."""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields, is_dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

from .io import atomic_json, append_json, sha256

STOP = False
CHECKPOINT_SCHEMA = "pilot-application-checkpoint-v2"
BUDGET_DEFAULTS = {"max_steps": 10000, "train_seconds": 420, "wall_seconds": 900,
                   "closing_reserve_seconds": 180}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def metadata_checkpoint_identity(data):
    """Content identity only: no torch/vendor imports or device initialization.

    Paths are relative to the active packages, so copying frozen inputs does
    not change their identity. This identifies Python execution source, not a
    native-binary build or a claim of cross-runtime numerical portability.
    """
    experiment = Path(__file__).resolve().parent
    experiment_files = {name: sha256(experiment / name)
                        for name in ("worker.py", "models.py", "data.py", "io.py")}
    spec = importlib.util.find_spec("tide")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("the active tide Python package cannot be identified")
    locations = list(spec.submodule_search_locations)
    if len(locations) != 1:
        raise ValueError("foundation identity requires one resolved tide package directory")
    foundation = Path(locations[0]).resolve()
    foundation_files = {str(path.relative_to(foundation)): sha256(path)
                        for path in sorted(foundation.rglob("*.py"))}
    if not foundation_files:
        raise ValueError("the active tide package has no Python execution source")
    packages = {}
    for name in ("torch", "torch-npu", "numpy", "tide"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    foundation_distributions = {name: importlib.metadata.version(name)
                               for name in importlib.metadata.packages_distributions().get("tide", [])}
    if packages["tide"] is None and len(foundation_distributions) == 1:
        packages["tide"] = next(iter(foundation_distributions.values()))
    return {"schema": "pilot-checkpoint-identity-v1",
            "experiment_sha256": canonical_digest(experiment_files),
            "foundation_sha256": canonical_digest(foundation_files),
            "data_manifest_sha256": sha256(Path(data) / "manifest.json"),
            "runtime_packages": packages, "python_version": sys.version.split()[0],
            "foundation_distributions": foundation_distributions,
            "experiment_files": experiment_files, "foundation_files": foundation_files}


def continuation_contract(task, device):
    """All application settings that affect the trajectory or its measurement.

    Research labels, task IDs, dependencies, placement and artifact estimates
    are scheduler metadata. Physical NPU index may be remapped; backend and
    thread count stay fixed. Budgets are checked separately with an explicit
    field whitelist on the continuation task.
    """
    return {"schema": "pilot-continuation-contract-v1", "model": copy.deepcopy(task["model"]),
            "seed": task["seed"], "cpu_threads": task["cpu_threads"],
            "backend": str(device).split(":", 1)[0], "precision": task.get("precision", "fp32"),
            "batch_size": task["batch_size"], "sequence_length": task["sequence_length"],
            "stream_chunks": task.get("stream_chunks", 4),
            "optimizer": {"class": "AdamW", "lr": task["lr"],
                          "weight_decay": task.get("weight_decay", 0.01),
                          "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False, "foreach": False},
            "warmup_steps": task.get("warmup_steps", 10), "lr_after_warmup": "constant",
            "grad_clip": task.get("grad_clip", 1.0), "detach_after_update": True,
            "state_reset": "sampled document-block boundary",
            "eval_batches": task.get("eval_batches", 2), "decode_steps": task.get("decode_steps", 8),
            "measurement_warmup_steps": task.get("measurement_warmup_steps", 3),
            "resume_check": task.get("resume_check", True),
            "resume_verify_steps": task.get("resume_verify_steps", 1),
            "checkpoint_interval_steps": task.get("checkpoint_interval_steps", 0)}


def runtime_budgets(task):
    return {key: task.get(key, default) for key, default in BUDGET_DEFAULTS.items()}


def stop_requested(signum, frame):
    global STOP
    STOP = True


def tree_map(value, fn):
    import torch
    if isinstance(value, torch.Tensor):
        return fn(value)
    if is_dataclass(value):
        return type(value)(**{f.name: tree_map(getattr(value, f.name), fn) for f in fields(value)})
    if isinstance(value, dict):
        return {k: tree_map(v, fn) for k, v in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(tree_map(v, fn) for v in value))
    if isinstance(value, tuple):
        return tuple(tree_map(v, fn) for v in value)
    if isinstance(value, list):
        return [tree_map(v, fn) for v in value]
    return value


def leaves(value, prefix=""):
    import torch
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif is_dataclass(value):
        for f in fields(value):
            yield from leaves(getattr(value, f.name), prefix + "/" + f.name)
    elif isinstance(value, dict):
        for k in sorted(value, key=str):
            yield from leaves(value[k], prefix + "/" + str(k))
    elif isinstance(value, (list, tuple)):
        for k, item in enumerate(value):
            yield from leaves(item, prefix + "/" + str(k))
    else:
        yield prefix, value


def digest_tree(value):
    import numpy as np
    import torch
    h = hashlib.sha256()
    for name, item in leaves(value):
        h.update(name.encode())
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            h.update(str((item.dtype, tuple(item.shape))).encode())
            h.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(str((item.dtype, item.shape)).encode())
            h.update(item.tobytes())
        else:
            h.update(repr(item).encode())
    return h.hexdigest()


def compare_tree(expected, actual, *, atol=1e-5, rtol=1e-4):
    import torch
    left, right = dict(leaves(expected)), dict(leaves(actual))
    result = {"atol": atol, "rtol": rtol, "fields": len(left), "exact": True,
              "within_tolerance": True, "structure_equal": set(left) == set(right),
              "integer_equal": True, "finite": True, "max_abs": 0.0, "failed_fields": []}
    if not result["structure_equal"]:
        result["exact"] = result["within_tolerance"] = False
        result["failed_fields"].append("field_keys")
    for name in left.keys() & right.keys():
        a, b = left[name], right[name]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            a, b = a.detach().cpu(), b.detach().cpu()
            if a.dtype != b.dtype or a.shape != b.shape:
                result["structure_equal"] = result["within_tolerance"] = result["exact"] = False
                if not a.is_floating_point() or not b.is_floating_point():
                    result["integer_equal"] = False
                result["failed_fields"].append(name + ":shape/dtype")
                continue
            exact = torch.equal(a, b)
            result["exact"] &= exact
            if a.is_floating_point():
                finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                result["finite"] &= finite
                within = finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol))
                if finite and a.numel():
                    result["max_abs"] = max(result["max_abs"], float((a - b).abs().max()))
            else:
                within = exact
                result["integer_equal"] &= exact
        else:
            exact = type(a) is type(b) and a == b
            result["exact"] &= exact
            within = exact
            if not exact:
                result["structure_equal"] = False
                if isinstance(a, (int, bool)) or isinstance(b, (int, bool)):
                    result["integer_equal"] = False
        result["within_tolerance"] &= within
        if not within:
            result["failed_fields"].append(name)
    result["failed_field_count"] = len(result["failed_fields"])
    result["failed_fields"] = result["failed_fields"][:30]
    return result


def atomic_torch_save(path, payload):
    import torch
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def initialize(device, threads, seed):
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(device)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    return torch.device(device)


def synchronize(device):
    import torch
    if device.type == "npu":
        torch.npu.synchronize(device)


def device_memory(device):
    import torch
    out = {"cpu_rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if device.type == "npu":
        out.update(npu_allocated_bytes=torch.npu.memory_allocated(device),
                   npu_reserved_bytes=torch.npu.memory_reserved(device),
                   npu_peak_allocated_bytes=torch.npu.max_memory_allocated(device),
                   npu_peak_reserved_bytes=torch.npu.max_memory_reserved(device))
    return out


def rng_state(device):
    import numpy as np
    import torch
    return {"torch_cpu": torch.get_rng_state(), "python": random.getstate(),
            "numpy": np.random.get_state(),
            "npu": torch.npu.get_rng_state(device).cpu() if device.type == "npu" else None}


def restore_rng(state, device):
    import numpy as np
    import torch
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if device.type == "npu":
        torch.npu.set_rng_state(state["npu"], device)


class StreamBatch:
    """TBPTT streams stay inside a document; reset all rows at block boundaries."""
    def __init__(self, store, task, saved=None):
        import numpy as np
        self.store, self.task = store, task
        self.rng = np.random.default_rng(task["seed"] + 101)
        self.block = None
        self.cursor = 0
        if saved is not None:
            self.rng.bit_generator.state = saved["rng"]
            self.block, self.cursor = saved["block"], saved["cursor"]

    def next(self):
        import torch
        length = self.task["sequence_length"]
        reset = self.block is None or self.cursor + length >= self.block.shape[1]
        if reset:
            total = length * self.task.get("stream_chunks", 4)
            self.block = self.store.sample_batch("train", self.task["batch_size"], total, self.rng)
            self.cursor = 0
        tokens = self.block[:, self.cursor:self.cursor + length + 1].copy()
        self.cursor += length
        return torch.from_numpy(tokens[:, :-1].copy()), torch.from_numpy(tokens[:, 1:].copy()), reset

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state, "block": self.block, "cursor": self.cursor}


def lr_at(task, step):
    return task["lr"] * min(1.0, (step + 1) / max(1, task.get("warmup_steps", 10)))


def train_update(model, optimizer, inputs, targets, state, device, task, step, capture=None):
    import torch
    from .models import detach_state
    from tide.precision import precision_context
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = lr_at(task, step)
    inputs, targets = inputs.to(device), targets.to(device)
    with precision_context(device, task.get("precision", "fp32")):
        out = model(inputs, state)
        loss = torch.nn.functional.cross_entropy(out.logits.float().flatten(0, 1), targets.flatten())
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), task.get("grad_clip", 1.0),
                                             error_if_nonfinite=True, foreach=False)
    if capture is not None:
        capture["gradients"] = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                                for name, p in model.named_parameters()}
    optimizer.step()
    if not bool(torch.isfinite(loss)):
        raise ArithmeticError("nonfinite language loss")
    return float(loss.detach()), float(grad_norm), detach_state(out.state), out.logits.detach()


def optimizer_for(model, task):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=task["lr"],
                             weight_decay=task.get("weight_decay", 0.01), foreach=False)


def evaluate(model, store, task, device, deadline=float("inf")):
    import numpy as np
    import torch
    from tide.precision import precision_context
    model.eval()
    rng = np.random.default_rng(17729)
    values, tokens = [], 0
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad(), precision_context(device, task.get("precision", "fp32")):
        for _ in range(task.get("eval_batches", 2)):
            if STOP or time.time() >= deadline:
                break
            sample = store.sample_batch("validation", min(task["batch_size"], 4), task["sequence_length"], rng)
            x, y = torch.from_numpy(sample[:, :-1].copy()).to(device), torch.from_numpy(sample[:, 1:].copy()).to(device)
            logits = model(x).logits.float()
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
            values.append(float(loss))
            tokens += y.numel()
    synchronize(device)
    if not values:
        return {"status": "budget_exhausted", "tokens": 0, "seconds": time.perf_counter()-start}
    nll = statistics.mean(values)
    if not math.isfinite(nll):
        raise ArithmeticError("nonfinite validation loss")
    return {"nll": nll, "perplexity": math.exp(nll), "tokens": tokens,
            "seconds": time.perf_counter() - start, "split": "pilot_article_holdout"}


def causality_check(model, store, task, device):
    import numpy as np
    import torch
    model.eval()
    seq = store.sample_batch("validation", 1, 16, np.random.default_rng(9001))
    x = torch.from_numpy(seq[:, :-1].copy()).to(device)
    y = x.clone()
    y[:, 8:] = (y[:, 8:] + 29) % store.vocab_size
    with torch.no_grad():
        a, b = model(x).logits[:, :8], model(y).logits[:, :8]
    result = compare_tree(a, b)
    if not result["within_tolerance"]:
        raise ArithmeticError("future-token perturbation changed earlier logits: " + json.dumps(result))
    return result


def inference_measure(model, store, task, device, deadline=float("inf")):
    import numpy as np
    import torch
    from tide.precision import precision_context
    model.eval()
    length, batch = task["sequence_length"], task["batch_size"]
    decode_steps = task.get("decode_steps", 8)
    sample = store.sample_batch("validation", batch, length + decode_steps, np.random.default_rng(9991))
    ids = torch.from_numpy(sample.copy()).to(device)
    prefills, decodes = [], []
    with torch.no_grad(), precision_context(device, task.get("precision", "fp32")):
        for repeat in range(4):
            if STOP or time.time() >= deadline:
                break
            synchronize(device)
            start = time.perf_counter()
            result = model(ids[:, :length])
            synchronize(device)
            elapsed = time.perf_counter() - start
            if repeat:
                prefills.append(elapsed)
            state = result.state
            for pos in range(decode_steps):
                if STOP or time.time() >= deadline:
                    break
                start = time.perf_counter()
                result = model(ids[:, length + pos:length + pos + 1], state)
                synchronize(device)
                elapsed = time.perf_counter() - start
                state = result.state
                if repeat:
                    decodes.append(elapsed)
    if not prefills or not decodes:
        return {"status": "budget_exhausted", "resident_prefill_seconds": prefills,
                "decode_seconds_per_step": decodes}
    return {"resident_prefill_seconds": prefills,
            "prefill_tokens_per_second": batch * length / statistics.mean(prefills),
            "decode_seconds_per_step": decodes,
            "decode_tokens_per_second": batch / statistics.mean(decodes),
            "decode_step_median_seconds": statistics.median(decodes),
            "batch": batch, "prompt_tokens_per_sample": length, "decode_steps_per_sample": decode_steps,
            "measured_prefill_requests": len(prefills), "measured_decode_steps": len(decodes),
            "sampling": "one warmup, up to three measured requests before deadline; teacher-forced real continuation",
            "scope": "resident model including LM head; tokenization and host transfer excluded"}


def cpu_snapshot(value):
    return tree_map(value, lambda tensor: tensor.detach().cpu().clone())


def checkpoint_state(payload):
    return {key: payload[key] for key in
            ("model", "optimizer", "private_state", "stream", "rng", "step")}


def private_history_details(state, stream, task):
    import torch
    tensors = [tensor for _, tensor in leaves(state)
               if isinstance(tensor, torch.Tensor) and tensor.numel()]
    saved = stream.state_dict() if isinstance(stream, StreamBatch) else stream
    block, cursor = saved["block"], saved["cursor"]
    return {"private_state_nonempty": bool(tensors), "private_tensor_count": len(tensors),
            "next_input_reuses_private_state": bool(tensors) and block is not None
                and cursor + task["sequence_length"] < block.shape[1],
            "stream_cursor": cursor}


def continuation_payload(model, optimizer, state, stream, task, device, step, data, identity=None):
    payload = {"schema": CHECKPOINT_SCHEMA, "task": copy.deepcopy(task),
               "data": str(Path(data).resolve()), "step": step,
               "checkpoint_identity": identity or metadata_checkpoint_identity(data),
               "contract": continuation_contract(task, device), "budgets": runtime_budgets(task),
               "model": cpu_snapshot(model.state_dict()),
               "optimizer": cpu_snapshot(optimizer.state_dict()),
               "private_state": cpu_snapshot(state),
               "stream": copy.deepcopy(stream.state_dict()), "rng": rng_state(device),
               "parameter_names": [name for name, _ in model.named_parameters()],
               "boundary": "after completed optimizer update; private state detached"}
    payload["state_sha256"] = digest_tree(checkpoint_state(payload))
    payload["history"] = private_history_details(state, stream, task)
    return payload


def validate_checkpoint(payload, task, data, device, identity=None):
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("strict continuation requires pilot-application-checkpoint-v2")
    observed = identity or metadata_checkpoint_identity(data)
    if payload.get("checkpoint_identity") != observed:
        raise ValueError("checkpoint source/runtime/data-manifest identity mismatch")
    if task.get("checkpoint_identity", observed) != observed:
        raise ValueError("task checkpoint_identity differs from actual source/runtime/data")
    expected = continuation_contract(task, device)
    saved = continuation_contract(payload["task"], payload["contract"]["backend"])
    if payload["contract"] != saved:
        raise ValueError("checkpoint task and saved continuation contract disagree")
    if payload["contract"] != expected:
        changed = [key for key in expected if expected[key] != payload["contract"].get(key)]
        raise ValueError("continuation contract mismatch: " + ", ".join(changed))
    if payload["budgets"] != runtime_budgets(payload["task"]):
        raise ValueError("checkpoint task and saved runtime budgets disagree")
    allowed = task.get("resume_allow_budget_overrides", [])
    if not isinstance(allowed, list) or any(key not in BUDGET_DEFAULTS for key in allowed):
        raise ValueError("resume_allow_budget_overrides must list supported runtime budget fields")
    changed = [key for key, value in runtime_budgets(task).items()
               if value != payload["budgets"][key]]
    unapproved = sorted(set(changed) - set(allowed))
    if unapproved:
        raise ValueError("runtime budget override not declared: " + ", ".join(unapproved))
    if type(payload["step"]) is not int or payload["step"] < 1:
        raise ValueError("checkpoint must represent at least one completed update")
    if digest_tree(checkpoint_state(payload)) != payload.get("state_sha256"):
        raise ValueError("checkpoint application state digest mismatch")
    stream = payload["stream"]
    block, cursor = stream["block"], stream["cursor"]
    length, chunks = task["sequence_length"], task.get("stream_chunks", 4)
    if (block is None or tuple(block.shape) != (task["batch_size"], length * chunks + 1)
            or type(cursor) is not int or cursor <= 0 or cursor > length * chunks or cursor % length):
        raise ValueError("checkpoint stream cursor/block does not match the data-flow contract")
    if not private_history_details(payload["private_state"], stream, task)["private_state_nonempty"]:
        raise ValueError("checkpoint has no private history")
    return {"identity_exact": True, "contract_exact": True, "state_digest_exact": True,
            "budget_overrides": changed}


def restore_application(payload, task, data, device):
    from .data import DocumentStore
    from .models import build_model
    store = DocumentStore(data)
    model = build_model(task["model"], store.vocab_size).to(device)
    if [name for name, _ in model.named_parameters()] != payload["parameter_names"]:
        raise ValueError("optimizer parameter identity/order changed")
    model.load_state_dict(payload["model"], strict=True)
    optimizer = optimizer_for(model, task)
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
    state = tree_map(payload["private_state"], lambda tensor: tensor.to(device).clone())
    stream = StreamBatch(store, task, copy.deepcopy(payload["stream"]))
    restored_digest = digest_tree({"model": model.state_dict(), "state": state,
                                  "optimizer": optimizer.state_dict()})
    expected_digest = digest_tree({"model": payload["model"], "state": payload["private_state"],
                                  "optimizer": payload["optimizer"]})
    if restored_digest != expected_digest:
        raise ValueError("restored model/optimizer/private state is not exact")
    restore_rng(payload["rng"], device)
    return store, model, optimizer, state, stream, restored_digest


def continuation_update(model, optimizer, state, stream, task, device, step):
    rng_before = digest_tree(rng_state(device))
    x, y, reset = stream.next()
    if reset:
        state = None
    capture = {}
    loss, norm, state, logits = train_update(model, optimizer, x, y, state, device, task, step, capture)
    synchronize(device)
    record = {"model": cpu_snapshot(model.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
              "private_state": cpu_snapshot(state), "logits": cpu_snapshot(logits),
              "gradients": capture["gradients"], "next_input_digest": digest_tree((x, y)),
              "rng_before_digest": rng_before, "rng_after_digest": digest_tree(rng_state(device)),
              "stream_after_digest": digest_tree(stream.state_dict()), "loss": loss, "grad_norm": norm,
              "step": step + 1, "state_reset": reset, "continued_private_state": not reset}
    return state, record


def resumed_update(payload, device):
    """Single-update convenience API; the verifier uses the same multi-step path."""
    task, data = payload["task"], payload["data"]
    validate_checkpoint(payload, task, data, device)
    _, model, optimizer, state, stream, restored_digest = restore_application(payload, task, data, device)
    _, actual = continuation_update(model, optimizer, state, stream, task, device, payload["step"])
    actual["restored_digest"] = restored_digest
    return actual


def compare_update(expected, actual):
    comparisons = {key: compare_tree(expected[key], actual[key]) for key in
                   ("model", "optimizer", "private_state", "logits", "gradients")}
    checks = {"next_input_exact": expected["next_input_digest"] == actual["next_input_digest"],
              "rng_before_exact": expected["rng_before_digest"] == actual["rng_before_digest"],
              "rng_after_exact": expected["rng_after_digest"] == actual["rng_after_digest"],
              "stream_after_exact": expected["stream_after_digest"] == actual["stream_after_digest"],
              "step_exact": expected["step"] == actual["step"],
              "state_reset_exact": expected["state_reset"] == actual["state_reset"]}
    passed = all(checks.values()) and all(value["within_tolerance"] for value in comparisons.values())
    return {"step": actual["step"], "within_tolerance": passed, **checks,
            "continued_private_state": actual["continued_private_state"], "comparisons": comparisons,
            "reference_loss": expected["loss"], "restored_loss": actual["loss"],
            "reference_grad_norm": expected["grad_norm"], "restored_grad_norm": actual["grad_norm"]}


def aggregate_comparisons(records):
    result = {}
    for key in ("model", "optimizer", "private_state", "logits", "gradients"):
        values = [record["comparisons"][key] for record in records]
        result[key] = {"atol": 1e-5, "rtol": 1e-4,
                       **{field: all(value[field] for value in values) for field in
                          ("exact", "within_tolerance", "structure_equal", "integer_equal", "finite")},
                       "max_abs": max((value["max_abs"] for value in values), default=0.0),
                       "fields": sum(value["fields"] for value in values),
                       "failed_field_count": sum(value["failed_field_count"] for value in values),
                       "failed_fields": [f"step{record['step']}:{name}" for record in records
                                         for name in record["comparisons"][key]["failed_fields"]][:30]}
    return result


def resume_child(args):
    import torch
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    result = {"status": "running", "started_unix": start, "pid": os.getpid(), "parent_pid": os.getppid()}
    try:
        payload = torch.load(args.resume_verify, map_location="cpu", weights_only=False)
        task, data = payload["task"], args.data or payload["data"]
        result["validation"] = validate_checkpoint(payload, task, data, args.device)
        device = initialize(args.device, task["cpu_threads"], task["seed"])
        _, model, optimizer, state, stream, restored_digest = restore_application(payload, task, data, device)
        count = args.resume_steps if args.resume_steps is not None else task.get("resume_verify_steps", 1)
        if type(count) is not int or count < 1:
            raise ValueError("resume_verify_steps must be a positive integer")
        comparisons, completed = [], 0
        for index in range(count):
            if STOP or time.time() >= args.deadline_unix:
                break
            state, actual = continuation_update(model, optimizer, state, stream, task, device,
                                                payload["step"] + index)
            completed += 1
            actual["restored_digest"] = restored_digest
            if args.resume_reference:
                expected = torch.load(Path(args.resume_reference) / f"{index:04d}.pt",
                                      map_location="cpu", weights_only=False)
                comparison = compare_update(expected, actual)
                comparisons.append(comparison)
                if not comparison["within_tolerance"]:
                    atomic_torch_save(output / f"mismatch-{index:04d}.pt", actual)
            else:
                atomic_torch_save(output / ("actual.pt" if count == 1 else f"actual-{index:04d}.pt"), actual)
        result.update(status="completed" if completed == count else "budget_exhausted",
                      step=payload["step"] + completed, verified_steps=completed, requested_steps=count,
                      restored_digest=restored_digest, per_step=comparisons, memory=device_memory(device))
    except Exception as exc:
        result.update(status="failed", error={"type": type(exc).__name__, "message": str(exc),
                                              "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        result["ended_unix"] = time.time()
        atomic_json(output / "result.json", result)
    return result


def save_checkpoint(output, payload):
    import torch
    if not all(bool(torch.isfinite(tensor).all()) for _, tensor in leaves(checkpoint_state(payload))
               if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()):
        raise ArithmeticError("nonfinite parameter/optimizer/private state at checkpoint")
    tick = time.perf_counter()
    checkpoint = output / "checkpoint.pt"
    atomic_torch_save(checkpoint, payload)
    record = {"path": str(checkpoint.resolve()), "bytes": checkpoint.stat().st_size,
              "save_seconds": time.perf_counter() - tick, "sha256": sha256(checkpoint),
              "step": payload["step"], "identity": payload["checkpoint_identity"], **payload["history"]}
    atomic_json(output / "checkpoint.json", record)
    return record


def verify_continuation(payload, model, optimizer, state, stream, task, device, output, deadline):
    import torch
    tick = time.perf_counter()
    count = task.get("resume_verify_steps", 1)
    if type(count) is not int or count < 1:
        raise ValueError("resume_verify_steps must be a positive integer")
    reference = output / "reference-updates"
    reference.mkdir(exist_ok=True)
    expected_start_digest = digest_tree({"model": payload["model"], "state": payload["private_state"],
                                        "optimizer": payload["optimizer"]})
    # Keep the original owner and its real model/optimizer/private history. Only
    # diagnostic evaluation's RNG effects are removed at the saved boundary.
    restore_rng(payload["rng"], device)
    for index in range(count):
        if STOP or time.time() >= deadline - 15:
            return {"status": "budget_exhausted", "reference_steps": index, "requested_steps": count}
        state, expected = continuation_update(model, optimizer, state, stream, task, device,
                                              payload["step"] + index)
        atomic_torch_save(reference / f"{index:04d}.pt", expected)
        del expected
    return {"reference": reference, "start_digest": expected_start_digest, "started": tick,
            "count": count}


def finish_verification(context, payload, task, device, output, deadline):
    checkpoint, child_out = output / "checkpoint.pt", output / "fresh-process"
    command = [sys.executable, "-m", "pilotlab.worker", "--resume-verify", str(checkpoint),
               "--resume-reference", str(context["reference"]), "--resume-steps", str(context["count"]),
               "--data", payload["data"], "--output", str(child_out), "--device", str(device),
               "--deadline-unix", str(deadline - 10)]
    with (output / "restore.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=max(1, deadline - time.time() - 5))
    if completed.returncode != 0 or not (child_out / "result.json").is_file():
        return {"status": "failed", "returncode": completed.returncode,
                "seconds": time.perf_counter() - context["started"]}
    actual = json.loads((child_out / "result.json").read_text())
    per_step = actual.get("per_step", [])
    if actual["status"] != "completed" or len(per_step) != context["count"]:
        return {"status": actual["status"], "requested_steps": context["count"],
                "verified_steps": len(per_step), "fresh_pid_result": actual}
    saved_exact = actual["restored_digest"] == context["start_digest"]
    ok = saved_exact and all(value["within_tolerance"] for value in per_step)
    result = {"status": "within_tolerance" if ok else "comparison_failed", "saved_data_exact": saved_exact,
              **{key: all(value[key] for value in per_step) for key in
                 ("next_input_exact", "rng_before_exact", "rng_after_exact", "stream_after_exact")},
              "continued_private_state": per_step[0]["continued_private_state"],
              "private_history": payload["history"], "comparisons": aggregate_comparisons(per_step),
              "per_step": per_step, "verified_steps": len(per_step), "requested_steps": context["count"],
              "reference_loss": per_step[-1]["reference_loss"], "restored_loss": per_step[-1]["restored_loss"],
              "reference_grad_norm": per_step[-1]["reference_grad_norm"],
              "restored_grad_norm": per_step[-1]["restored_grad_norm"],
              "from_step": payload["step"], "to_step": actual["step"],
              "seconds": time.perf_counter() - context["started"],
              "fresh_pid_result": {key: value for key, value in actual.items() if key != "per_step"}}
    if ok:
        for path in context["reference"].glob("*.pt"):
            path.unlink()
        context["reference"].rmdir()
    return result


def execute(args):
    task = json.loads(Path(args.task).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start_wall = time.time()
    args.deadline_unix = min(args.deadline_unix, start_wall + task.get("wall_seconds", 900))
    atomic_json(output / "worker.json", {"pid": os.getpid(), "started_unix": start_wall,
                "task_id": task["id"], "affinity": sorted(os.sched_getaffinity(0))})
    result = {"schema": "pilot-task-result-v1", "task_id": task["id"], "task": task,
              "started_unix": start_wall, "status": "running", "recovery": {"status": "not_attempted"}}
    try:
        import numpy as np
        import torch
        from .data import DocumentStore
        from .models import build_model
        identity = metadata_checkpoint_identity(args.data)
        if task.get("checkpoint_identity", identity) != identity:
            raise ValueError("task checkpoint_identity differs from actual source/runtime/data")
        if task.get("backend", args.device.split(":", 1)[0]) != args.device.split(":", 1)[0]:
            raise ValueError("task backend differs from explicit execution device")
        checkpoint_every = task.get("checkpoint_interval_steps", 0)
        if type(checkpoint_every) is not int or checkpoint_every < 0:
            raise ValueError("checkpoint_interval_steps must be a nonnegative integer")
        resume_payload = None
        if task.get("resume_from"):
            resume_path = Path(task["resume_from"])
            checkpoint_sha256 = sha256(resume_path)
            if task.get("resume_checkpoint_sha256", checkpoint_sha256) != checkpoint_sha256:
                raise ValueError("resume checkpoint file sha256 mismatch")
            resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            validation = validate_checkpoint(resume_payload, task, args.data, args.device, identity)
            result["continuation"] = {"source": str(resume_path.resolve()), "sha256": checkpoint_sha256,
                                       **validation, "from_step": resume_payload["step"]}
        device = initialize(args.device, task["cpu_threads"], task["seed"])
        if resume_payload is not None:
            store, model, optimizer, state, stream, restored_digest = restore_application(
                resume_payload, task, args.data, device)
            start_step = resume_payload["step"]
            result["continuation"]["restored_digest"] = restored_digest
        else:
            store = DocumentStore(args.data)
            model = build_model(task["model"], store.vocab_size).to(device)
            optimizer = optimizer_for(model, task)
            stream, state, start_step = StreamBatch(store, task), None, 0
        training_rng = rng_state(device)
        result.update(device=str(device), physical_visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                      torch_version=torch.__version__, data_manifest_sha256=identity["data_manifest_sha256"],
                      checkpoint_identity=identity, cpu_threads=torch.get_num_threads(),
                      precision=task.get("precision", "fp32"))
        synchronize(device)
        result.update(parameters=sum(p.numel() for p in model.parameters()), readout_policy=model.readout_policy,
                      initialization_seconds=time.time() - start_wall)
        initial_hashes = {name: digest_tree(parameter) for name, parameter in model.named_parameters()}
        result["causality"] = causality_check(model, store, task, device)
        result["validation_initial"] = evaluate(model, store, task, device, args.deadline_unix)
        probe = torch.from_numpy(store.sample_batch("validation", 1, task["sequence_length"],
                                                   np.random.default_rng(66))[:, :-1].copy()).to(device)
        with torch.no_grad():
            result["work_initial"] = model(probe, record_work=True).work
        restore_rng(training_rng, device)
        times, losses, steps = [], [], 0
        if device.type == "npu":
            torch.npu.reset_peak_memory_stats(device)
        training_start = time.monotonic()
        reserve = task.get("closing_reserve_seconds", 180)
        training_end = min(time.time() + task.get("train_seconds", 420), args.deadline_unix - reserve)
        max_steps = task.get("max_steps", 10000)
        if type(max_steps) is not int or max_steps < 1 or reserve < 0:
            raise ValueError("max_steps must be positive and closing reserve nonnegative")
        while (not STOP and steps < max_steps and time.time() < args.deadline_unix - reserve
               and (time.time() < training_end or steps > 0
                    and not private_history_details(state, stream, task)["next_input_reuses_private_state"])):
            synchronize(device)
            tick = time.perf_counter()
            rng_before = digest_tree(rng_state(device))
            x, y, reset = stream.next()
            if reset:
                state = None
            input_digest = digest_tree((x, y))
            loss, norm, state, _ = train_update(model, optimizer, x, y, state, device, task, start_step + steps)
            synchronize(device)
            elapsed = time.perf_counter() - tick
            times.append(elapsed)
            losses.append(loss)
            steps += 1
            global_step = start_step + steps
            append_json(output / "steps.jsonl", {"step": global_step, "segment_step": steps,
                        "loss": loss, "grad_norm": norm, "lr": optimizer.param_groups[0]["lr"],
                        "seconds": elapsed, "tokens": y.numel(), "state_reset": reset,
                        "input_sha256": input_digest, "rng_before_sha256": rng_before,
                        "rng_after_sha256": digest_tree(rng_state(device)), "time_unix": time.time()})
            if checkpoint_every and steps % checkpoint_every == 0:
                payload = continuation_payload(model, optimizer, state, stream, task, device,
                                               global_step, args.data, identity)
                result["checkpoint"] = save_checkpoint(output, payload)
                del payload
            if steps % 10 == 0:
                print(json.dumps({"id": task["id"], "step": global_step, "loss": loss, "seconds": elapsed}), flush=True)
        stop_reason = ("signal" if STOP else "max_steps" if steps >= max_steps else
                       "deadline" if time.time() >= args.deadline_unix - reserve else "train_seconds")
        measured = times[min(task.get("measurement_warmup_steps", 3), max(0, len(times) - 1)):]
        result["training"] = {"steps": steps, "from_step": start_step, "to_step": start_step + steps,
               "global_steps": start_step + steps,
               "tokens": steps * task["batch_size"] * task["sequence_length"], "stop_reason": stop_reason,
               "total_seconds": time.monotonic() - training_start,
               "measured_steps": len(measured), "step_seconds": measured,
               "tokens_per_second": task["batch_size"] * task["sequence_length"] / statistics.mean(measured) if measured else None,
               "step_median_seconds": statistics.median(measured) if measured else None,
               "step_p90_seconds": float(np.quantile(measured, .9)) if measured else None,
               "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
               "loss_last10_mean": statistics.mean(losses[-10:]) if losses else None,
               "scope": "data sampling/transfer, input and RNG digest, forward CE, backward, clipping, AdamW and synchronization; checkpoint/JSON logging excluded",
               "state_policy": "document-contiguous TBPTT; detach at updates; reset at sampled document-block boundaries"}
        result["training_memory"] = device_memory(device)
        if start_step + steps == 0:
            result["status"] = "budget_exhausted"
            return result
        payload = continuation_payload(model, optimizer, state, stream, task, device,
                                       start_step + steps, args.data, identity)
        changed = [name for name, parameter in model.named_parameters() if digest_tree(parameter) != initial_hashes[name]]
        result["parameter_updates"] = {"changed_tensor_count": len(changed), "total_tensor_count": len(initial_hashes),
                    "changed_by_component": {prefix: sum(name.startswith(prefix) for name in changed)
                                              for prefix in ("core.", "blocks.", "embedding.", "output_norm.")},
                    "none_gradient_parameters": [name for name, parameter in model.named_parameters() if parameter.grad is None]}
        result["checkpoint"] = save_checkpoint(output, payload)
        if result.get("continuation"):
            result["continuation"].update(to_step=payload["step"], additional_steps=steps,
                                           first_input_reused_private_history=bool(steps) and
                                           resume_payload["history"]["next_input_reuses_private_state"])
        if not STOP and time.time() < args.deadline_unix - min(reserve, 150):
            result["validation_final"] = evaluate(model, store, task, device, args.deadline_unix - min(reserve, 100))
            model.eval()
            with torch.no_grad():
                result["work_final"] = model(probe, record_work=True).work
            result["inference"] = inference_measure(model, store, task, device,
                                                     args.deadline_unix - min(reserve, 100))
        if not STOP and task.get("resume_check", True) and time.time() < args.deadline_unix - min(reserve, 110):
            context = verify_continuation(payload, model, optimizer, state, stream, task,
                                          device, output, args.deadline_unix)
            del optimizer, model, state, probe
            if device.type == "npu":
                torch.npu.empty_cache()
            if "reference" in context:
                result["recovery"] = finish_verification(context, payload, task, device, output, args.deadline_unix)
            else:
                result["recovery"] = context
        if result["recovery"]["status"] == "failed":
            result["status"] = "failed"
        else:
            result["status"] = "budget_exhausted" if STOP or stop_reason == "deadline" or not steps else "completed"
        result["memory_final"] = device_memory(device)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        traceback.print_exc()
    finally:
        result["ended_unix"] = time.time()
        result["wall_seconds"] = time.time() - start_wall
        atomic_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--resume-verify")
    parser.add_argument("--resume-reference")
    parser.add_argument("--resume-steps", type=int)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    if args.resume_verify:
        result = resume_child(args)
    else:
        result = execute(args)
    return 0 if result["status"] in {"completed", "budget_exhausted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
