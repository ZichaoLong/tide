"""Bounded same-precision training-path checks, not a performance benchmark.

Every path starts from one real baseline update's parameters, private state and
AdamW slots. Repeats precede mutation; continued updates and restored updates are
separate comparisons. Existing tolerances and observed failures are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
import tide

from tide import (ModelConfig, TideModel, canonical_state, canonical_state_dict,
                  detach_state, load_checkpoint, placement_adamw, placement_report,
                  save_checkpoint)
from tide.placement import move_tensors, place_state
from tide.precision import precision_context, precision_policy
from benchmarks.workloads import paired_input, tensor_hash
from scripts.check_placement import _objective, _place, _sync
from scripts.job import atomic
from scripts.qualify_precision import THRESHOLDS, atomic_tensor, comparison, cpu_tree


RESTORE_TOLERANCE = {"atol": 1e-6, "rtol": 2e-3}
VARIANTS = {
    "baseline": {"region_batch_size": 1, "checkpoint_regions": False},
    "batch4": {"region_batch_size": 4, "checkpoint_regions": False},
    "recompute": {"region_batch_size": 1, "checkpoint_regions": True},
    "combined": {"region_batch_size": 4, "checkpoint_regions": True},
    "placement_serial": {"region_batch_size": 1, "checkpoint_regions": False},
    "placement_per_device": {"region_batch_size": 1, "checkpoint_regions": False},
}


def package_identity():
    root = Path(tide.__file__).resolve().parent
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(root.rglob("*.py"))}
    return dict(root=str(root), files=hashes,
                sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())


def resolve_variants(extra=None):
    variants = {name: dict(options) for name, options in VARIANTS.items()}
    allowed = {"region_batch_size", "checkpoint_regions", "checkpoint_reception",
               "ssd_summary_backend", "ssd_scan_backend", "local_attention_backend", "reception_qkv_backend",
               "reception_mask_padding", "parameter_indexing"}
    if extra is not None and not isinstance(extra, dict):
        raise ValueError("candidate overrides must be a name-to-options JSON object")
    for name, options in (extra or {}).items():
        if (not isinstance(name, str) or not name or name in variants or name.startswith("placement_") or
                any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in name)):
            raise ValueError("new candidate names must be unique lowercase identifiers; built-in meanings cannot be replaced")
        if not isinstance(options, dict) or not set(options) <= allowed:
            raise ValueError("candidate overrides must contain only declared execution-option fields")
        variants[name] = dict(VARIANTS["baseline"], **options)
    return variants


def leaves(value, prefix="root"):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif is_dataclass(value):
        for field in fields(value):
            yield from leaves(getattr(value, field.name), prefix+"."+field.name)
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from leaves(value[key], prefix+"."+str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from leaves(item, prefix+"."+str(index))
    else:
        yield prefix, value


def support_field(name):
    field = name.rsplit(".", 1)[-1]
    return field == "strengths" or ".state." in name and field in {"weight", "short_weight", "long_weight"}


def screen(value):
    """Compact hashes/counts before retaining any first-difference values."""
    all_hash, discrete, support = (hashlib.sha256() for _ in range(3))
    dtypes, counts, missing, nonfinite = Counter(), Counter(), [], []
    support_rows = []
    def structure(item):
        if isinstance(item, torch.Tensor):
            return "tensor"
        if is_dataclass(item):
            return (type(item).__name__, [(field.name, structure(getattr(item, field.name))) for field in fields(item)])
        if isinstance(item, dict):
            return (type(item).__name__, [(str(key), structure(item[key])) for key in sorted(item, key=str)])
        if isinstance(item, (list, tuple)):
            return (type(item).__name__, [structure(part) for part in item])
        return type(item).__name__
    layout = repr(structure(value)).encode()
    all_hash.update(layout); discrete.update(layout)
    for name, item in leaves(value):
        all_hash.update(name.encode())
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            identity = str((tuple(item.shape), str(item.dtype))).encode()
            digest = tensor_hash(item.reshape(-1))
            all_hash.update(identity); all_hash.update(digest.encode())
            dtypes[str(item.dtype)] += item.numel(); counts["tensors"] += 1
            if not item.is_floating_point():
                discrete.update(name.encode()+identity+digest.encode())
            elif not bool(torch.isfinite(item).all()):
                nonfinite.append(name)
            if support_field(name):
                mask = item > 0
                sign_hash = tensor_hash(mask.reshape(-1))
                support.update(name.encode()+str(tuple(item.shape)).encode()+sign_hash.encode())
                support_rows.append(dict(field=name, positive=int(mask.sum()), elements=item.numel(), sha256=sign_hash))
        else:
            encoded = repr((type(item).__name__, item)).encode()
            all_hash.update(encoded); discrete.update(name.encode()+encoded)
            if item is None:
                missing.append(name)
    return dict(sha256=all_hash.hexdigest(), discrete_sha256=discrete.hexdigest(),
                support_sha256=support.hexdigest(), support=support_rows, none_fields=missing,
                tensor_count=counts["tensors"], dtype_elements=dict(dtypes), nonfinite=nonfinite)


def first_differences(actual, expected, limit=8):
    """First field in deterministic returned-tree order, not a causal trace."""
    left, right = dict(leaves(actual)), dict(leaves(expected))
    result = {}
    if left.keys() != right.keys():
        result["structure"] = dict(missing=sorted(right.keys()-left.keys())[:limit],
                                   extra=sorted(left.keys()-right.keys())[:limit])
    for name in sorted(left.keys() & right.keys()):
        a, b = left[name], right[name]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                result.setdefault("structure", dict(field=name, actual_shape=list(a.shape),
                    expected_shape=list(b.shape), actual_dtype=str(a.dtype), expected_dtype=str(b.dtype)))
                continue
            if torch.equal(a, b):
                continue
            if a.is_floating_point():
                kind = "numeric" if bool(torch.isfinite(a).all() and torch.isfinite(b).all()) else "nonfinite"
            else:
                kind = "time_or_index"
            def selected(mask):
                indices = mask.reshape(-1).nonzero().flatten()[:limit]
                # Strings retain nonfinite values without creating invalid JSON.
                def encode(x):
                    if not x.is_floating_point():
                        return bool(x) if x.dtype == torch.bool else int(x)
                    return float(x) if torch.isfinite(x) else str(float(x))
                return [dict(flat_index=int(i), actual=encode(a.reshape(-1)[i]),
                             expected=encode(b.reshape(-1)[i])) for i in indices]
            result.setdefault(kind, dict(field=name, shape=list(a.shape), dtype=str(a.dtype),
                                         coordinates=selected(a != b)))
            weight_name = name.removesuffix("times")+"weight"
            if (kind == "time_or_index" and result[kind].get("field") == name and
                    name.endswith(".times") and weight_name in left and weight_name in right):
                wa, wb = left[weight_name], right[weight_name]
                if isinstance(wa, torch.Tensor) and isinstance(wb, torch.Tensor) and wa.shape == a.shape and wb.shape == b.shape:
                    result[kind].setdefault("paired_key_weights", [dict(flat_index=int(i),
                        actual=float(wa.reshape(-1)[i]), expected=float(wb.reshape(-1)[i]))
                        for i in (a != b).reshape(-1).nonzero().flatten()[:limit]])
                    result[kind]["interpretation"] = "stored key-time/support witness; causal Attention visibility also requires the actual query time and declared window"
            if support_field(name) and not torch.equal(a > 0, b > 0):
                result.setdefault("support", dict(field=name, shape=list(a.shape),
                    coordinates=selected((a > 0) != (b > 0))))
        elif type(a) is not type(b) or a != b:
            kind = ("none_gradient" if (a is None or b is None) and "gradient" in name else
                    "time_or_index" if name.endswith(".position") else "structure")
            result.setdefault(kind, dict(field=name, actual=repr(a), expected=repr(b)))
    return result


def compare(actual, expected, tolerance):
    a, b = screen(actual), screen(expected)
    checked = comparison(actual, expected, tolerance)
    first = {} if a["sha256"] == b["sha256"] else first_differences(actual, expected)
    if checked["metadata_failures"] and not first:
        first["structure"] = dict(fields=checked["metadata_failures"][:8])
    return dict(screen_actual=a, screen_expected=b, comparison=checked, first_differences=first,
                first_scope="deterministic returned-tree field order; internal causal attribution is separate")


def _parameter_parts(model):
    """Diagnostic mapping from actual Parameters to their original row owners."""
    registry = getattr(model, "_shard_registry", {})
    ordinary = [(name, parameter, None, tuple(parameter.shape))
                for name, parameter in model.named_parameters()
                if not any(name.startswith(prefix+".") for prefix in registry)]
    for prefix, wrapper in registry.items():
        for leaf, shape in wrapper._shapes.items():
            for shard, indices in zip(wrapper.shards, wrapper._maps[leaf]):
                ordinary.append((prefix+"."+leaf, getattr(shard, leaf), indices, shape))
    return ordinary


def optimizer_parameters(model, optimizer):
    """Check the actual update group, independently of retained slot entries."""
    if len(optimizer.param_groups) != 1:
        raise ValueError("this finite diagnostic uses exactly one AdamW group")
    named = dict(model.named_parameters())
    expected = {id(parameter) for parameter in named.values()}
    actual = [id(parameter) for parameter in optimizer.param_groups[0]["params"]]
    if len(actual) != len(set(actual)):
        raise ValueError("optimizer group Parameters must be unique")
    if set(actual) != expected:
        raise ValueError("optimizer group must contain exactly the model Parameters")
    if any(id(parameter) not in expected for parameter in optimizer.state):
        raise ValueError("optimizer slots contain a Parameter outside the model")
    return named


def canonical_optimizer(model, optimizer):
    """Gather optimizer rows explicitly; reject inconsistent shard slot meaning."""
    optimizer_parameters(model, optimizer)
    grouped = {}
    for name, parameter, indices, shape in _parameter_parts(model):
        grouped.setdefault(name, []).append((parameter, indices, shape))
    state = {}
    for name, parts in grouped.items():
        slots = [optimizer.state.get(parameter, {}) for parameter, _, _ in parts]
        if not any(slots):
            continue
        if any(not slot or slot.keys() != slots[0].keys() for slot in slots):
            raise ValueError("inconsistent optimizer slot presence for original Parameter "+name)
        assembled = {}
        for key in slots[0]:
            values = [slot[key] for slot in slots]
            if any(type(value) is not type(values[0]) for value in values):
                raise ValueError("inconsistent optimizer slot type for "+name+"."+key)
            if isinstance(values[0], torch.Tensor):
                if any(value.dtype != values[0].dtype or value.ndim != values[0].ndim for value in values):
                    raise ValueError("inconsistent optimizer slot dtype/shape for "+name+"."+key)
                if key == "step" and values[0].ndim != 0:
                    raise ValueError("optimizer step must be scalar for "+name)
                if values[0].ndim and any(value.shape != parameter.shape
                        for (parameter, _, _), value in zip(parts, values)):
                    raise ValueError("optimizer tensor slot shape differs from Parameter "+name+"."+key)
            if isinstance(values[0], torch.Tensor) and values[0].ndim:
                value = torch.zeros(parts[0][2], dtype=values[0].dtype)
                for (_, indices, _), part in zip(parts, values):
                    if indices is None:
                        value.copy_(part.detach().cpu())
                    else:
                        value.index_copy_(0, torch.tensor(indices), part.detach().cpu())
                assembled[key] = value
            else:
                equal = (all(torch.equal(values[0].cpu(), v.cpu()) for v in values[1:])
                         if isinstance(values[0], torch.Tensor) else all(v == values[0] for v in values[1:]))
                if not equal:
                    raise ValueError("inconsistent optimizer scalar slot for "+name+"."+key)
                assembled[key] = cpu_tree(values[0])
        state[name] = assembled
    group = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}
    return dict(state=state, group=group, parameter_names=sorted(grouped))


def load_canonical_optimizer(model, optimizer, canonical):
    """Install one frozen optimizer point using original parameter-row indices."""
    optimizer_parameters(model, optimizer)
    if "params" in canonical["group"]:
        raise ValueError("canonical optimizer options cannot replace the Parameter group")
    actual_names = {name for name, _, _, _ in _parameter_parts(model)}
    if actual_names != set(canonical["parameter_names"]):
        raise ValueError("canonical optimizer parameter identities differ")
    optimizer.state.clear()
    optimizer.param_groups[0].update(canonical["group"])
    for name, parameter, indices, _ in _parameter_parts(model):
        if name not in canonical["state"]:
            continue
        slots = {}
        for key, value in canonical["state"][name].items():
            if isinstance(value, torch.Tensor):
                if value.ndim:
                    value = value if indices is None else value.index_select(0, torch.tensor(indices))
                    slots[key] = value.clone().to(parameter.device)
                else:
                    slots[key] = value.clone()  # non-capturable AdamW step lives on CPU
            else:
                slots[key] = value
        optimizer.state[parameter] = slots


def training_data(model, optimizer, state):
    return dict(parameters=canonical_state_dict(model), optimizer=canonical_optimizer(model, optimizer),
                state=cpu_tree(canonical_state(model, state)))


def arithmetic(device, precision):
    return nullcontext() if precision == "fp64" else precision_context(device, precision)


def forward(model, value, state, chunk, *, trace=None):
    results = []
    for index, start in enumerate(range(0, value.shape[1], chunk)):
        if trace is not None:
            trace.phase = "chunk"+str(index)
        result = model(value[:, start:start+chunk], state, record_work=True)
        results.append(result); state = result.state
    if not model.settle:
        if trace is not None:
            trace.phase = "flush"
        results.append(model.flush(state, record_work=True))
    loss = sum(_objective(result) for result in results)
    return results, loss


def optimizer_step_count(slot, name):
    if not slot:
        return 0
    if "step" not in slot:
        raise RuntimeError("AdamW slots have no step counter for "+name)
    value = slot["step"]
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise RuntimeError("AdamW step must be scalar for "+name)
        value = value.item()
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value != int(value):
        raise RuntimeError("AdamW step must be a finite nonnegative integer for "+name)
    return int(value)


def checked_optimizer_step(model, optimizer, devices):
    """Require one actual AdamW step per active original Parameter, including its shards."""
    named = optimizer_parameters(model, optimizer)
    parts = _parameter_parts(model)
    active = {name for name, parameter, _, _ in parts if parameter.grad is not None}
    if not active:
        raise RuntimeError("optimizer update requires a connected model Parameter gradient")
    expected_active = {id(parameter) for name, parameter, _, _ in parts if name in active}
    before = {id(parameter): optimizer_step_count(optimizer.state.get(parameter, {}), name)
              for name, parameter in named.items()}
    optimizer.step(); _sync(devices)
    after = {id(parameter): optimizer_step_count(optimizer.state.get(parameter, {}), name)
             for name, parameter in named.items()}
    for name, parameter in named.items():
        key = id(parameter)
        if after[key] != before[key]+int(key in expected_active):
            raise RuntimeError("AdamW step did not advance exactly once for active Parameters or stay unchanged for inactive Parameters: "+name)
    def original_counts(counts):
        result = {}
        for name, parameter, _, _ in parts:
            value = counts[id(parameter)]
            if name in result and result[name] != value:
                raise RuntimeError("inconsistent AdamW step across original Parameter shards: "+name)
            result[name] = value
        return result
    return dict(checked=True, active_original_parameters=sorted(active),
                before=original_counts(before), after=original_counts(after))


def observe(model, optimizer, state, value, precision, devices, chunk, *, grad=True, update=False):
    if update and not grad:
        raise ValueError("an optimizer update requires backward")
    optimizer.zero_grad(set_to_none=True)
    initial_state = screen(cpu_tree(canonical_state(model, state)))["sha256"]
    x = value.detach().clone().to(devices[0]).requires_grad_(grad)
    with torch.set_grad_enabled(grad), arithmetic(devices[0], precision):
        results, loss = forward(model, x, state, chunk)
    if grad:
        loss.backward()
    _sync(devices)
    output = dict(forward=dict(segments=[dict(values=r.values, strengths=r.strengths, times=r.times,
        state=canonical_state(model, r.state)) for r in results]), loss=loss.detach())
    if grad:
        output.update(input_gradient=x.grad, gradients=canonical_state_dict(model, gradients=True))
    output = cpu_tree(output)
    if update:
        output["optimizer_update"] = checked_optimizer_step(model, optimizer, devices)
    following = detach_state(results[-1].state)
    if update:
        output.update(training_data(model, optimizer, following))
    output["mechanism"] = dict(input_connected=not grad or x.grad is not None,
        loss_connected=not grad or loss.requires_grad,
        original_state_unchanged=initial_state == screen(cpu_tree(canonical_state(model, state)))["sha256"])
    if not all(output["mechanism"].values()) or screen(output)["nonfinite"]:
        raise RuntimeError("nonfinite values, disconnected input/loss or mutated incoming state")
    work = [r.work for r in results]
    return output, following, work


def build_candidate(config, name, anchor, devices, precision, transport, *, overrides=None):
    variant = replace(config, **(VARIANTS[name] if overrides is None else overrides))
    model = TideModel(variant)
    if precision == "fp64":
        model = model.double()
    model.load_state_dict(anchor["parameters"])
    if name.startswith("placement_"):
        model = _place(model, devices, transport, "region", region_concurrency=name.removeprefix("placement_"))
        state = place_state(model, cpu_tree(anchor["state"]))
    else:
        model = model.to(devices[0]); state = move_tensors(cpu_tree(anchor["state"]), devices[0])
    optimizer = placement_adamw(model, lr=1e-4, foreach=False)
    load_canonical_optimizer(model, optimizer, anchor["optimizer"])
    return model, optimizer, state


def summarize(checks, exact_checks=()):
    rows = [row["comparison"] for row in checks]
    if any(row["nonfinite"] for row in rows):
        return "nonfinite"
    if any(not row["comparison"]["exact"] for row in exact_checks):
        return "exact_contract_failed"
    if any(row["metadata_failures"] or row["dtype_differences"] for row in rows):
        return "structural_failed"
    if any(row["threshold_failures"] or row["support_differences"] for row in rows):
        return "finite_deviation"
    return "within_threshold"


def restore_worker(bundle_path, output):
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    options = bundle["options"]
    if options.get("deterministic") is not None:
        torch.use_deterministic_algorithms(options["deterministic"])
    tolerance = THRESHOLDS["fp64"] if options["precision"] == "fp64" else RESTORE_TOLERANCE
    kwargs = ({"placement_devices": options["devices"]} if bundle["placed"] else {"device": options["devices"][0]})
    model, state, slots = load_checkpoint(bundle["checkpoint"], **kwargs)
    optimizer = placement_adamw(model, lr=1e-4, foreach=False); optimizer.load_state_dict(slots)
    checks = {"saved_data": compare(training_data(model, optimizer, state), bundle["saved"], tolerance)}
    prediction, _, _ = observe(model, optimizer, state, bundle["input"], options["precision"],
                               options["devices"], options["chunk"], grad=False)
    checks["next_forward"] = compare(prediction, bundle["next_forward"], tolerance)
    updated, _, work = observe(model, optimizer, state, bundle["input"], options["precision"],
                               options["devices"], options["chunk"], update=True)
    checks["actual_update"] = compare(updated, bundle["updated"], tolerance)
    record = dict(state=summarize(list(checks.values()), [checks["saved_data"], checks["next_forward"]]),
        pid=os.getpid(), checks=checks, work=work, precision=options["precision"],
        config_exact=model.config.to_dict() == bundle["config"], actual_update=updated["optimizer_update"]["checked"],
        optimizer_update=updated["optimizer_update"],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), tolerance=tolerance,
        package_identity=package_identity(), harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source=os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        bundle_sha256=hashlib.sha256(Path(bundle_path).read_bytes()).hexdigest(),
        source_identity_scope="source_exact compares Tide Python package files and this entry script; auxiliary harness modules and external toolchains are outside that comparison")
    record["source_exact"] = (record["package_identity"]["sha256"] == bundle["package_identity"]["sha256"] and
                              record["harness_sha256"] == bundle["harness_sha256"])
    if not record["config_exact"] or not record["source_exact"]:
        record["state"] = "exact_contract_failed"
    atomic(output, record)
    return record


def validate_restore_record(record, bundle_sha256, *, exit_code=None, original_pid=None):
    if "checks" not in record:
        raise RuntimeError("restore execution failed: "+str(record.get("error", record)))
    successful = {"within_threshold", "finite_deviation"}
    if record.get("state") not in successful | {"nonfinite", "exact_contract_failed", "structural_failed"}:
        raise RuntimeError("restore execution returned an unknown completion state")
    if record.get("bundle_sha256") != bundle_sha256:
        raise RuntimeError("restore record does not identify the current input bundle")
    if exit_code is not None:
        expected_exit = 0 if record["state"] in successful else 2
        if exit_code != expected_exit:
            raise RuntimeError(f"fresh-process restore exit code {exit_code} disagrees with recorded state {record.get('state')}")
        if type(record.get("pid")) is not int or record["pid"] <= 0 or record["pid"] == original_pid:
            raise RuntimeError("fresh-process restore did not identify a distinct child process")


def check_restore(model, optimizer, state, value, options, directory, *, new_process=True):
    checkpoint = directory/"checkpoint.pt"
    save_checkpoint(checkpoint, model, state=state, optimizer=optimizer)
    saved = training_data(model, optimizer, state)
    prediction, _, _ = observe(model, optimizer, state, value, options["precision"], options["devices"], options["chunk"], grad=False)
    updated, _, work = observe(model, optimizer, state, value, options["precision"], options["devices"], options["chunk"], update=True)
    bundle = dict(checkpoint=str(checkpoint.resolve()), placed=hasattr(model, "_placement_spec"), saved=saved,
        input=value, next_forward=prediction, updated=updated, options=options, config=model.config.to_dict(),
        package_identity=package_identity(), harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = directory/"restore-input.pt"; atomic_tensor(path, bundle)
    bundle_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    output = directory/"restore.json"
    if new_process:
        command = [sys.executable, "-m", "scripts.check_training_paths", "--resume-bundle", str(path.resolve()),
                   "--resume-output", str(output.resolve()), "--threads", str(torch.get_num_threads())]
        environment = dict(os.environ)
        package_parent = str(Path(tide.__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = package_parent+(os.pathsep+environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        with (directory/"restore.log").open("w") as stream:
            stream.write(completed.stdout); stream.flush(); os.fsync(stream.fileno())
        if not output.exists():
            raise RuntimeError("fresh-process restore produced no record: "+completed.stdout[-3000:])
        record = json.loads(output.read_text()); record.update(command=command, exit_code=completed.returncode,
            package_source_prepend=package_parent,
            environment_policy="prepend the actual parent Tide source and preserve existing compiler Python paths")
        validate_restore_record(record, bundle_sha256, exit_code=completed.returncode, original_pid=os.getpid())
    else:
        record = restore_worker(path, output)
        validate_restore_record(record, bundle_sha256)
    record.update(new_process=new_process, original_process_pid=os.getpid(), original_update_work=work,
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    return record


def run(config, *, device, precision, candidates, batch, length, steps, chunk, directory,
        placement_devices=(), transport="auto", new_process=True, witness=True,
        deterministic=None, candidate_overrides=None, witness_elements=2_000_000,
        progress=lambda record: None):
    if type(witness_elements) is not int or not 1 <= witness_elements <= 20_000_000:
        raise ValueError("witness_elements must be 1..20000000")
    if any(directory.iterdir()):
        raise ValueError("training-path output directory must be empty")
    if deterministic is not None:
        torch.use_deterministic_algorithms(deterministic)
    if candidates[0] != "baseline" or len(candidates) != len(set(candidates)):
        raise ValueError("baseline must be first and candidates unique")
    variants = resolve_variants(candidate_overrides)
    if len(candidates) > 8 or any(name not in variants for name in candidates):
        raise ValueError("choose at most eight declared candidates")
    devices = list(placement_devices) or [device]
    if any(name.startswith("placement_") for name in candidates) and len(devices) < 2:
        raise ValueError("placement candidates require at least two declared devices")
    if str(torch.device(devices[0])) != str(torch.device(device)):
        raise ValueError("first placement device must match the reference device")
    torch.manual_seed(1701)
    base = replace(config, **VARIANTS["baseline"])
    model = TideModel(base).to(device)
    if precision == "fp64":
        model = model.double()
    optimizer = placement_adamw(model, lr=1e-4, foreach=False)
    dtype = torch.float64 if precision == "fp64" else torch.float32
    inputs = [paired_input(batch, length, config.input_dim or config.d, 1702+i).to(dtype) for i in range(steps+2)]
    warmup, state, warmup_work = observe(model, optimizer, None, inputs[0], precision, [device], chunk, update=True)
    anchor = training_data(model, optimizer, state)
    anchor_steps = {optimizer_step_count(slot, name) for name, slot in anchor["optimizer"]["state"].items()}
    if not anchor["optimizer"]["state"] or anchor_steps != {1}:
        raise RuntimeError("baseline anchor requires nonempty AdamW slots from exactly one real update")
    atomic_tensor(directory/"anchor.pt", dict(anchor=anchor, inputs=inputs, config=base.to_dict()))
    record = dict(schema="tide-training-paths-v1", state="running", tolerance=THRESHOLDS[precision],
        restore_tolerance=THRESHOLDS["fp64"] if precision == "fp64" else RESTORE_TOLERANCE,
        deterministic_requested=deterministic, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        witness_elements=witness_elements,
        resolved_candidates={name: variants[name] for name in candidates},
        fixed_start=screen(anchor), input_hashes=[tensor_hash(x) for x in inputs],
        anchor_optimizer=dict(parameter_states=len(anchor["optimizer"]["state"]), steps=sorted(anchor_steps)),
        anchor_update_work=warmup_work, warmup_summary=screen(warmup), candidates=[],
        boundary="one real baseline anchor update; same fixed nonempty state/AdamW for each path; no detach between chunks, detach after each complete update/flush",
        observation="intrusive same-precision correctness checks; no performance timing",
        first_divergence_scope="first changed returned-tree field per category; not automatically an implementation error or an internal causal boundary")
    del model, optimizer, state, warmup
    baseline = None; witnessed = False
    for name in candidates:
        case_dir = directory/name; case_dir.mkdir()
        model, optimizer, state = build_candidate(config, name, anchor, devices if name.startswith("placement_") else [device], precision, transport, overrides=variants[name])
        actual_devices = devices if name.startswith("placement_") else [device]
        case = dict(name=name, config=model.config.to_dict(), checks={}, updates=[], observations={}, state="running",
                    placement=placement_report(model) if name.startswith("placement_") else None)
        record["candidates"].append(case)
        case["checks"]["fixed_start"] = compare(training_data(model, optimizer, state), anchor, THRESHOLDS[precision])
        if not case["checks"]["fixed_start"]["comparison"]["exact"]:
            raise RuntimeError("candidate did not start at the frozen parameter/state/optimizer point")
        observed = {}
        for phase in ("inference", "vjp1", "vjp2"):
            observed[phase], _, work = observe(model, optimizer, state, inputs[1], precision, actual_devices, chunk, grad=phase != "inference")
            case["observations"][phase] = dict(summary=screen(observed[phase]), work=work)
            if phase != "inference":
                case["observations"][phase]["actual_none_parameters"] = [key for key, parameter in model.named_parameters() if parameter.grad is None]
            if baseline is not None:
                case["checks"][phase+"_vs_baseline"] = compare(observed[phase], baseline[phase], THRESHOLDS[precision])
            progress(record)
        case["checks"]["same_owner_vjp_repeat"] = compare(observed["vjp2"], observed["vjp1"], THRESHOLDS[precision])
        case["checks"]["inference_vs_grad_forward"] = compare(observed["inference"]["forward"], observed["vjp1"]["forward"], THRESHOLDS[precision])
        case["checks"]["start_unchanged_after_vjp"] = compare(training_data(model, optimizer, state), anchor, THRESHOLDS[precision])
        if witness and not witnessed and baseline is not None and not case["checks"]["vjp1_vs_baseline"]["comparison"]["exact"]:
            case["first_divergence_witness"] = first_path_witness(config, name, anchor, inputs[1], device, precision, chunk, case_dir,
                overrides=variants[name], witness_elements=witness_elements)
            witnessed = True
        observed["updates"] = []
        for index in range(steps):
            updated, state, work = observe(model, optimizer, state, inputs[index+1], precision, actual_devices, chunk, update=True)
            observed["updates"].append(updated)
            row = dict(index=index, summary=screen(updated), work=work)
            if baseline is not None:
                row["vs_baseline"] = compare(updated, baseline["updates"][index], THRESHOLDS[precision])
            case["updates"].append(row); progress(record)
        case["restore"] = check_restore(model, optimizer, state, inputs[steps+1],
            dict(precision=precision, devices=actual_devices, chunk=chunk, deterministic=deterministic), case_dir, new_process=new_process)
        if "checks" not in case["restore"]:
            raise RuntimeError("restore execution failed: "+str(case["restore"].get("error", case["restore"])))
        checks = list(case["checks"].values())+[row["vs_baseline"] for row in case["updates"] if "vs_baseline" in row]
        checks += list(case["restore"]["checks"].values())
        exact = [case["checks"]["fixed_start"], case["checks"]["start_unchanged_after_vjp"],
                 case["restore"]["checks"]["saved_data"], case["restore"]["checks"]["next_forward"]]
        case["state"] = summarize(checks, exact)
        if case["restore"]["state"] not in {"within_threshold", "finite_deviation"}:
            case["state"] = case["restore"]["state"]
        if baseline is None:
            baseline = observed
        del model, optimizer, state, observed
        progress(record)
    states = [case["state"] for case in record["candidates"]]
    record["state"] = next((state for state in ("nonfinite", "exact_contract_failed", "structural_failed", "finite_deviation") if state in states), "within_threshold")
    progress(record)
    return record


def first_path_witness(config, name, anchor, value, device, precision, chunk, directory, *, overrides=None,
                       witness_elements=2_000_000):
    """A bounded serial internal replay only for unplaced forward differences."""
    if type(witness_elements) is not int or not 1 <= witness_elements <= 20_000_000:
        raise ValueError("witness_elements must be 1..20000000")
    if name.startswith("placement_"):
        return dict(state="returned_fields_only", reason="per_device instrumentation changes scheduling; use recorded support/time/None/coordinate witnesses and same-owner repeats; no causal claim")
    from scripts.diagnose_region_batch import Trace, difference, linear_replay, regional_replay, support_witness
    class Limit(Exception):
        pass
    class BoundedTrace(Trace):
        elements = 0
        def charge(self, tensor):
            self.elements += tensor.numel()
            if self.elements > witness_elements:
                raise Limit(f"{witness_elements:,} captured elements per path")
        def put(self, region, stage, tensor, **metadata):
            self.charge(tensor); super().put(region, stage, tensor, **metadata)
        def linear_hook(self, module_name):
            original = super().linear_hook(module_name)
            def hook(module, args, output):
                self.charge(args[0]); original(module, args, output)
            return hook
    models, traces = [], []
    try:
        for candidate in ("baseline", name):
            model, optimizer, state = build_candidate(config, candidate, anchor, [device], precision, "auto",
                overrides=VARIANTS["baseline"] if candidate == "baseline" else overrides)
            trace = BoundedTrace(model)
            with torch.enable_grad(), arithmetic(device, precision), trace.install():
                results, loss = forward(model, value.detach().clone().to(device).requires_grad_(), state, chunk, trace=trace)
            models.append(model); traces.append(trace)
            del optimizer, results, loss
        common = [key for key in traces[0].order if key in traces[1].entries]
        changed = [key for key in common if not torch.equal(traces[0].entries[key]["value"], traces[1].entries[key]["value"])]
        result = dict(state="completed", captured_elements=[trace.elements for trace in traces],
            shared_keys=len(common), changed_keys=len(changed),
            order="baseline serial region traversal within each chunk/flush; first observed internal stage, not automatically its mathematical cause",
            missing_keys=[list(key) for key in traces[0].order if key not in traces[1].entries][:8], facts=[], unproven=[])
        payload = {}
        if changed:
            key = changed[0]
            result["first_internal"] = dict(key=key, difference=difference(traces[1].entries[key]["value"], traces[0].entries[key]["value"]))
            payload["first_internal"] = [trace.entries[key] for trace in traces]
            result["facts"].append("all earlier shared captured tensors were exactly equal")
            if precision != "fp64" and key[2].startswith("linear."):
                result["replay"], payload["replay"] = linear_replay(models, traces, key, precision, device)
            elif precision != "fp64" and key[2] == "regional.score":
                result["replay"], payload["replay"] = regional_replay(models, traces, key, precision, device)
            result["unproven"].append("a replay is explanatory only where its recorded replay_matches_capture checks pass; operator kernels and all downstream errors are not inferred")
        supports = [key for key in common if key[2] == "control" and traces[0].entries[key]["value"].shape == traces[1].entries[key]["value"].shape and not torch.equal(traces[0].entries[key]["value"] > 0, traces[1].entries[key]["value"] > 0)]
        if supports and config.allocation == "quad":
            result["first_support"] = dict(key=supports[0], independent_decimal=support_witness(models[0], traces, supports[0]))
        if not changed:
            result["unproven"].append("these forward captures did not locate a difference; backward/optimizer attribution remains open")
        atomic_tensor(directory/"first-witness.pt", payload)
        return result
    except Limit as error:
        return dict(state="capture_limit", reason=str(error), attribution="not established; returned-field first differences remain recorded")


def configure_devices(devices):
    if any(str(device).startswith("npu") for device in devices):
        import torch_npu  # noqa: F401
        torch.npu.set_device(devices[0])
        torch.npu.matmul.allow_hf32 = False; torch.npu.conv.allow_hf32 = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=tuple(THRESHOLDS), default="fp32")
    parser.add_argument("--candidates", nargs="+", default=["baseline", "recompute"])
    parser.add_argument("--candidate-overrides", type=json.loads, default={},
                        help="JSON object mapping new candidate names to declared execution options; built-ins cannot be replaced")
    parser.add_argument("--placement-devices", nargs="*", default=[])
    parser.add_argument("--transport", choices=("auto", "direct", "host_stage", "synchronized"), default="auto")
    parser.add_argument("--batch", type=int, default=1); parser.add_argument("--length", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2); parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None,
                        help="explicitly enable/disable deterministic algorithms; omitted preserves the default")
    parser.add_argument("--max-parameters", type=int, default=20_000_000)
    parser.add_argument("--restore-mode", choices=("new_process", "same_process"), default="new_process")
    parser.add_argument("--no-internal-witness", action="store_true")
    parser.add_argument("--witness-elements", type=int, default=2_000_000,
                        help="internal capture cap per path, 1..20000000 (default: 2000000)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-bundle", type=Path); parser.add_argument("--resume-output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("threads must be 1..16")
    if not 1 <= args.witness_elements <= 20_000_000:
        parser.error("witness-elements must be 1..20000000")
    expected_package = Path(__file__).resolve().parents[1]/"src"/"tide"
    if Path(tide.__file__).resolve().parent != expected_package:
        parser.error("Tide must resolve inside this frozen checkout; prepend its src to the existing PYTHONPATH")
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    if args.deterministic is not None:
        torch.use_deterministic_algorithms(args.deterministic)
    if args.resume_bundle:
        if args.resume_output is None or args.resume_output.exists():
            parser.error("resume requires a new output path")
        bundle = torch.load(args.resume_bundle, map_location="cpu", weights_only=False)
        configure_devices(bundle["options"]["devices"])
        try:
            record = restore_worker(args.resume_bundle, args.resume_output)
        except Exception as error:
            record = dict(state="execution_failed", error=repr(error), pid=os.getpid(),
                          deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
            atomic(args.resume_output, record)
            raise
        return 0 if record["state"] in {"within_threshold", "finite_deviation"} else 2
    if args.config is None or args.output_dir is None or args.output_dir.exists():
        parser.error("config and a new output directory are required")
    if not (1 <= args.batch <= 8 and 1 <= args.length <= 64 and 1 <= args.steps <= 4 and
            1 <= args.chunk <= args.length and 1 <= args.threads <= 16 and args.max_parameters > 0):
        parser.error("bounded checks require B1..8, L1..64, steps1..4, valid chunk, threads1..16 and positive parameter cap")
    if args.precision == "fp64" and any(not device.startswith("cpu") for device in [args.device]+args.placement_devices):
        parser.error("FP64 is a CPU-only diagnostic")
    config = ModelConfig.from_dict(json.loads(args.config.read_text()))
    try:
        variants = resolve_variants(args.candidate_overrides)
        if any(name not in variants for name in args.candidates):
            raise ValueError("unknown candidate name")
        for name in args.candidates:
            replace(config, **variants[name])  # reject unavailable fields before expensive work
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    with torch.device("meta"):
        parameters = sum(p.numel() for p in TideModel(config).parameters())
    if parameters > args.max_parameters:
        parser.error(f"{parameters} parameters exceeds the explicit {args.max_parameters} diagnostic cap")
    configure_devices(args.placement_devices or [args.device]); args.output_dir.mkdir(parents=True)
    record = dict(source=os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        policy=precision_policy(args.device, args.precision) if args.precision != "fp64" else dict(precision="CPU FP64 diagnostic"),
        torch=str(torch.__version__), package_identity=package_identity(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        deterministic_requested=args.deterministic, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY"),
        state="running")
    def progress(result):
        record["result"] = result; record["state"] = result["state"]
        atomic(args.output_dir/"result.json", record)
    try:
        result = run(config, device=args.device, precision=args.precision, candidates=args.candidates,
            batch=args.batch, length=args.length, steps=args.steps, chunk=args.chunk, directory=args.output_dir,
            placement_devices=args.placement_devices, transport=args.transport,
            new_process=args.restore_mode == "new_process", witness=not args.no_internal_witness,
            deterministic=args.deterministic, candidate_overrides=args.candidate_overrides,
            witness_elements=args.witness_elements, progress=progress)
    except Exception as error:
        record.update(state="execution_failed", error=repr(error)); atomic(args.output_dir/"result.json", record)
        raise
    return 0 if result["state"] in {"within_threshold", "finite_deviation"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
