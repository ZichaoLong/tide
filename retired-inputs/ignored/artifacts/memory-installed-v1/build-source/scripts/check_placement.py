"""Finite HB layer-shard qualification or bounded large capacity execution.

Run actual accelerators only inside an admitted whole-device-group service.
The reference is an ordinary one-device model initialized from the same CPU data.
Large --mode inference/train omit that duplicate capacity requirement.
"""
import argparse
from collections import OrderedDict
import copy
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import tempfile
import time

import torch

from tide import (ModelConfig, TopologyConfig, TideModel, canonical_state_dict, canonical_state, detach_state,
                  load_checkpoint, place_model, placement_adamw, placement_report, save_checkpoint,
                  serial_region_audit)
from tide.placement import move_tensors
from tide.precision import precision_context, precision_policy
from benchmarks.statistics import RuntimeAudit
from benchmarks.continuous import nested_comparison, state_memory, synchronize
from scripts.job import atomic


def _place(model, devices, transport, partition_kind="layer", region_partitions=None, region_concurrency="serial"):
    if partition_kind == "region":
        from tide.placement_ownership import balanced_region_partitions
        regions = (balanced_region_partitions(model, len(devices))
                   if region_partitions is None else region_partitions)
        return place_model(model, devices, region_partitions=regions, transport=transport,
                           region_concurrency=region_concurrency)
    if partition_kind != "layer" or region_partitions is not None:
        raise ValueError("explicit region partitions require partition_kind=region")
    return place_model(model, devices, transport=transport, region_concurrency=region_concurrency)


def _canonical_result(model, result):
    if result is None:
        return None
    from dataclasses import replace
    return replace(result, state=canonical_state(model, result.state))


def _sync(devices):
    for device in dict.fromkeys(map(str, devices)):
        synchronize(device)


def _peaks(devices, *, reset=False):
    result = {}
    for device in dict.fromkeys(map(str, devices)):
        resolved = torch.device(device)
        backend = getattr(torch, resolved.type, None)
        if resolved.type in {"npu", "cuda"}:
            if reset:
                backend.reset_peak_memory_stats(resolved)
            else:
                result[device] = dict(allocated_bytes=backend.max_memory_allocated(resolved),
                                      reserved_bytes=backend.max_memory_reserved(resolved))
    return result


def _objective(result):
    value = result.readout().float()
    # A signed scalar avoids choosing only a nearly constant RMS objective.
    root = torch.cos(torch.arange(value.numel(), device=value.device, dtype=value.dtype)).reshape_as(value)
    return (value*root).mean()


def _full(model, x, state=None, *, record_work=False):
    result = model(x, state, record_work=record_work)
    if model.settle:
        return result, None
    return result, model.flush(result.state, record_work=record_work)


def _named_tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _named_tensors(getattr(value, field.name), prefix+"."+field.name)
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from _named_tensors(item, prefix+"."+str(name))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _named_tensors(item, prefix+"."+str(index))


def _storage_record(items):
    """Actual visible payload and retained storage; never gather device tensors."""
    devices, stores = {}, {}
    for _, tensor in items:
        device, dtype = str(tensor.device), str(tensor.dtype)
        row = devices.setdefault(device, dict(payload_bytes=0, retained_storage_bytes=0,
                                              tensor_count=0, elements=0, dtypes={}))
        typed = row["dtypes"].setdefault(dtype, dict(tensor_count=0, elements=0, payload_bytes=0))
        size = tensor.numel()*tensor.element_size()
        for target in (row, typed):
            target["tensor_count"] += 1
            target["elements"] += tensor.numel()
            target["payload_bytes"] += size
        storage = tensor.untyped_storage()
        stores[device, storage.data_ptr()] = storage.nbytes()
    for (device, _), size in stores.items():
        devices[device]["retained_storage_bytes"] += size
    return dict(by_device=devices, **{key: sum(row[key] for row in devices.values())
                                    for key in ("payload_bytes", "retained_storage_bytes", "tensor_count", "elements")})


def _finite_record(items, *, chunk_elements=1 << 20):
    """Bound temporary masks even for a multi-GiB Parameter or AdamW slot."""
    bad, tensors, elements = [], 0, 0
    for name, tensor in items:
        if not tensor.is_floating_point() and not tensor.is_complex():
            continue
        tensors += 1
        elements += tensor.numel()
        value = tensor.detach().reshape(-1)
        finite = torch.ones((), dtype=torch.bool, device=tensor.device)
        for start in range(0, value.numel(), chunk_elements):
            finite.logical_and_(torch.isfinite(value[start:start+chunk_elements]).all())
        if not bool(finite):
            bad.append(name)
    return dict(finite=not bad, nonfinite_tensors=bad, floating_tensors=tensors,
                floating_elements=elements, chunk_elements=chunk_elements)


def _training_tensors(model, optimizer, state):
    parameters = list(model.named_parameters())
    return dict(parameters=parameters,
                gradients=[(name, value.grad) for name, value in parameters if value.grad is not None],
                optimizer_slots=[(name+"."+str(slot), value)
                                 for name, parameter in parameters
                                 for slot, value in optimizer.state.get(parameter, {}).items()
                                 if isinstance(value, torch.Tensor)],
                buffers=list(model.named_buffers()),
                private_state=list(_named_tensors(state, "state")))


def _parameter_sample(model):
    # A bounded update witness, not a full-parameter equality comparison.
    return {name: parameter.detach().reshape(-1)[:16].cpu().clone()
            for name, parameter in model.named_parameters()}


def _sample_delta(before, after):
    differences = [(after[name]-value).abs() for name, value in before.items()]
    return dict(sampled_parameters=len(before), sampled_elements=sum(value.numel() for value in before.values()),
                changed_elements=sum(int((value > 0).sum()) for value in differences),
                max_abs=max((float(value.max()) for value in differences if value.numel()), default=0.0),
                scope="first up to 16 scalars of every actual Parameter; not a full-parameter trajectory comparison")


def _capacity_update(model, optimizer, x, state, precision, devices, *, mark, label, phase_sync):
    phases = {}
    mark(label+".forward")
    _sync(devices)
    whole_started = started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with precision_context(devices[0], precision):
        result, tail = _full(model, x, state)
        loss = _objective(result)+(0 if tail is None else _objective(tail))
    if phase_sync:
        _sync(devices)
    phases["zero_grad_forward_loss"] = time.perf_counter()-started
    mark(label+".backward")
    started = time.perf_counter()
    loss.backward()
    if phase_sync:
        _sync(devices)
    phases["backward"] = time.perf_counter()-started
    mark(label+".optimizer")
    started = time.perf_counter()
    optimizer.step()
    following = detach_state((result if tail is None else tail).state)
    _sync(devices)
    finished = time.perf_counter()
    phases["optimizer_detach_sync"] = finished-started
    complete = finished-whole_started
    return following, result, tail, loss, dict(complete_seconds=complete, phase_seconds=phases,
                                               between_phase_host_seconds=complete-sum(phases.values()))


def capacity_train(config, devices, precision, batch=1, length=8, steps=2, *, progress=None,
                   transport="auto", foreach=True, phase_sync=False, partition_kind="layer", region_partitions=None,
                   region_concurrency="serial"):
    """One placed model, continued real updates, no duplicate model/checkpoint."""
    if min(batch, length, steps) < 1:
        raise ValueError("positive capacity training batch, length and steps required")
    torch.manual_seed(807)
    record = dict(config=config.to_dict(), precision=precision_policy(devices[0], precision),
                  mode="capacity_train", state="running", seed=807, batch=batch, length=length,
                  requested_updates=steps, updates=[], progress=[], phase_synchronization=phase_sync,
                  comparison="capacity training execution only; no one-device equivalence reference or checkpoint",
                  optimizer=dict(kind="placement_adamw", learning_rate=1e-4, foreach=foreach),
                  continuation="carry detached private state into the next update; Timed flush contributes to loss and advances the resumed clock",
                  timing="complete wall time includes zero_grad, forward/readout/necessary flush/loss, backward, cross-device transport, AdamW, detach, final all-device sync and intervening diagnostic stage record IO; individual host phases exclude stage IO and measure device latency only with phase_sync",
                  timing_exclusions=["CPU construction", "parameter identity hash", "placement and input transfer",
                                     "optimizer construction", "work/operator audits", "finite/storage/update-witness scans"],
                  audit_scope="separate graph-enabled forwards at each update's identical parameters, input and private state; no update or detach of the timed state",
                  memory_scope="timed allocator peaks reset after audit graphs are released; reserved bytes can retain construction/audit caches; process RSS covers the whole process")
    def mark(name):
        record["last_stage"] = name
        record["progress"].append(dict(stage=name, monotonic_seconds=time.monotonic()))
        if progress is not None:
            progress(record)
        print(json.dumps(dict(stage=name)), flush=True)

    costs = record["preparation_seconds"] = {}
    mark("construction.cpu")
    started = time.perf_counter()
    model = TideModel(config)
    costs["cpu_model"] = time.perf_counter()-started
    mark("construction.parameter_identity")
    started = time.perf_counter()
    fingerprint = hashlib.sha256()
    for name, parameter in model.named_parameters():
        fingerprint.update(name.encode())
        fingerprint.update(str(tuple(parameter.shape)).encode())
        fingerprint.update(str(parameter.dtype).encode())
        fingerprint.update(memoryview(parameter.detach().numpy()).cast("B"))
    record["initial_parameter_sha256"] = fingerprint.hexdigest()
    record["initial_parameter_hash_format"] = "sha256 of ordered name, shape, dtype and contiguous CPU bytes; zero-copy memoryview"
    costs["parameter_identity"] = time.perf_counter()-started
    mark("construction.placement")
    started = time.perf_counter()
    model = _place(model, devices, transport, partition_kind, region_partitions, region_concurrency)
    _sync(devices)
    costs["placement_transfer"] = time.perf_counter()-started
    record["placement"] = placement_report(model)
    mark("construction.optimizer_input")
    started = time.perf_counter()
    optimizer = placement_adamw(model, lr=1e-4, foreach=foreach)
    x_cpu = torch.randn(batch, length, config.input_dim or config.d)
    x = x_cpu.to(devices[0])
    _sync(devices)
    costs["optimizer_input_transfer"] = time.perf_counter()-started
    record["input_sha256"] = hashlib.sha256(memoryview(x_cpu.numpy()).cast("B")).hexdigest()
    record["construction_transfer_seconds"] = sum(value for key, value in costs.items() if key != "parameter_identity")
    record["construction_peak_device_memory"] = _peaks(devices)
    state = None
    sample = _parameter_sample(model)
    for step in range(steps):
        label = f"update.{step}.placed"
        row = dict(index=step, state_before_position=0 if state is None else state.position)
        record["updates"].append(row)
        mark(label+".work_audit")
        started = time.perf_counter()
        with torch.enable_grad(), precision_context(devices[0], precision):
            observed, tail = _full(model, x, state, record_work=True)
        _sync(devices)
        row["work"] = dict(input=observed.work, flush=None if tail is None else tail.work)
        del observed, tail
        _sync(devices)
        row["work_audit_seconds"] = time.perf_counter()-started
        if step == 0:
            mark(label+".operator_audit")
            started = time.perf_counter()
            audit = RuntimeAudit()
            with serial_region_audit(model) as execution, torch.enable_grad(), precision_context(devices[0], precision), audit:
                observed, tail = _full(model, x, state)
            _sync(devices)
            record["operator_witness"] = dict(audit.report(), execution=execution, update_index=step,
                                               state_position=row["state_before_position"])
            del observed, tail
            _sync(devices)
            row["operator_audit_seconds"] = time.perf_counter()-started
        _peaks(devices, reset=True)
        with torch.enable_grad():
            state, result, tail, loss, timing = _capacity_update(
                model, optimizer, x, state, precision, devices, mark=mark, label=label, phase_sync=phase_sync)
        row.update(timing, peak_device_memory=_peaks(devices), state_position=state.position)
        mark(label+".finite_storage_checks")
        started = time.perf_counter()
        tensors = _training_tensors(model, optimizer, state)
        row["actual_storage"] = {name: _storage_record(items) for name, items in tensors.items()}
        row["actual_storage"]["combined"] = _storage_record([item for items in tensors.values() for item in items])
        outputs = [("result.values", result.values), ("result.strengths", result.strengths), ("loss", loss)]
        if tail is not None:
            outputs += [("flush.values", tail.values), ("flush.strengths", tail.strengths)]
        row["finite_checks"] = {name: _finite_record(items) for name, items in dict(tensors, outputs=outputs).items()}
        row["finite"] = all(check["finite"] for check in row["finite_checks"].values())
        row["loss"] = float(loss.detach())
        row["output_positive_strengths"] = int((result.strengths.detach() > 0).sum())
        row["flush_output_positive_strengths"] = 0 if tail is None else int((tail.strengths.detach() > 0).sum())
        row["private_state_requires_grad_tensors"] = sum(value.requires_grad for _, value in tensors["private_state"])
        counters = [float(slots["step"]) for slots in optimizer.state.values() if "step" in slots]
        row["optimizer_step_counters"] = dict(count=len(counters), minimum=min(counters, default=None),
                                                maximum=max(counters, default=None))
        following_sample = _parameter_sample(model)
        row["parameter_update_witness"] = _sample_delta(sample, following_sample)
        sample = following_sample
        row["checks_seconds"] = time.perf_counter()-started
        del tensors, outputs, result, tail, loss
        mark(label+".complete")
        if not row["finite"]:
            record["state"] = "nonfinite"
            record["not_run_updates"] = list(range(step+1, steps))
            mark("stopped.nonfinite")
            return record
    samples = [row["complete_seconds"] for row in record["updates"]]
    record.update(state="completed", completed_updates=len(samples), seconds=samples,
                  median_seconds=statistics.median(samples),
                  input_positions_per_second=batch*length/statistics.median(samples),
                  state_memory=state_memory(state))
    mark("completed")
    return record


def _train(model, optimizer, x, state, precision, devices, *, progress=None, label="update", phase_sync=False):
    following, result, tail, loss, timing = _capacity_update(
        model, optimizer, x, state, precision, devices,
        mark=progress if progress is not None else lambda phase: None, label=label, phase_sync=phase_sync)
    finite = bool(torch.isfinite(loss)) and all(
        bool(torch.isfinite(p).all()) and (p.grad is None or bool(torch.isfinite(p.grad).all()))
        for p in model.parameters())
    return following, dict(
        loss=float(loss.detach()), finite=finite, **timing,
        phase_synchronization=phase_sync,
        timing="complete wall time through final all-device sync and detach includes intervening stage record IO; individual host phases exclude stage IO and measure device latency only with phase_sync",
        output_positive_strengths=(result.strengths.detach() > 0).sum().item(),
        flush_output_positive_strengths=0 if tail is None else (tail.strengths.detach() > 0).sum().item())


def qualify(config, devices, precision, checkpoint, batch, length, steps, *, progress=None,
            transport="auto", phase_sync=False, partition_kind="layer", region_partitions=None,
            region_concurrency="serial"):
    torch.manual_seed(807)
    source = TideModel(config)
    fingerprint = hashlib.sha256()
    for name, parameter in source.named_parameters():
        fingerprint.update(name.encode()); fingerprint.update(parameter.detach().numpy().tobytes())
    record = dict(config=config.to_dict(), precision=precision_policy(devices[0], precision),
                  initial_parameter_sha256=fingerprint.hexdigest(),
                  reference="one-device ordinary model with identical initial CPU parameters and inputs")
    record.update(state="running", progress=[], phase_synchronization=phase_sync)
    def mark(name):
        record["last_stage"] = name
        record["progress"].append(dict(stage=name, monotonic_seconds=time.monotonic()))
        if progress is not None:
            progress(record)
        print(json.dumps(dict(stage=name)), flush=True)
    mark("construction")
    reference = copy.deepcopy(source).to(devices[0])
    candidate = _place(source, devices, transport, partition_kind, region_partitions, region_concurrency)
    record["placement"] = placement_report(candidate)
    x_cpu = torch.randn(batch, length, config.input_dim or config.d)
    record["input_sha256"] = hashlib.sha256(x_cpu.numpy().tobytes()).hexdigest()
    x = x_cpu.to(devices[0]).requires_grad_()
    y = x_cpu.to(devices[0]).detach().clone().requires_grad_()
    comparisons = {}
    record["comparisons"] = comparisons
    mark("initial_forward")
    with precision_context(devices[0], precision):
        expected, expected_tail = _full(reference, x, record_work=True)
        actual, actual_tail = _full(candidate, y, record_work=True)
        reference_loss = _objective(expected)+(0 if expected_tail is None else _objective(expected_tail))
        candidate_loss = _objective(actual)+(0 if actual_tail is None else _objective(actual_tail))
    _sync(devices)
    mark("initial_reference_backward")
    reference_loss.backward(); _sync(devices)
    mark("initial_placed_backward")
    candidate_loss.backward(); _sync(devices)
    mark("initial_comparisons")
    comparisons["whole_result"] = nested_comparison(_canonical_result(candidate, actual), _canonical_result(reference, expected))
    comparisons["flush_result"] = nested_comparison(_canonical_result(candidate, actual_tail), _canonical_result(reference, expected_tail))
    comparisons["input_gradient"] = nested_comparison(y.grad, x.grad, rtol=1e-2)
    comparisons["parameter_gradients"] = nested_comparison(canonical_state_dict(candidate, gradients=True),
        OrderedDict((name, None if p.grad is None else p.grad.detach().cpu())
                    for name, p in reference.named_parameters()))
    record["work"] = actual.work
    mark("continuation_and_operator_audit")
    with torch.no_grad(), precision_context(devices[0], precision):
        split = max(1, length//2)
        first = candidate(y[:, :split])
        reference_first = reference(x[:, :split])
        if split < length:
            second = candidate(y[:, split:], first.state)
            reference_second = reference(x[:, split:], reference_first.state)
            joined = torch.cat((first.readout(), second.readout()), 1)
            reference_joined = torch.cat((reference_first.readout(), reference_second.readout()), 1)
            following, reference_following = second.state, reference_second.state
        else:
            joined, following = first.readout(), first.state
            reference_joined, reference_following = reference_first.readout(), reference_first.state
        comparisons["continued_single_vs_placed"] = nested_comparison(joined, reference_joined)
        comparisons["continued_single_vs_placed_state"] = nested_comparison(canonical_state(candidate, following), reference_following)
        comparisons["reference_whole_vs_continued"] = nested_comparison(reference_joined, expected.readout())
        comparisons["continued_readout"] = nested_comparison(joined, actual.readout())
        comparisons["continued_state"] = nested_comparison(following, actual.state)
        if not candidate.settle:
            tail = candidate.flush(following, record_work=True)
            comparisons["continued_flush"] = nested_comparison(tail, actual_tail)
        audit = RuntimeAudit()
        with serial_region_audit(candidate) as execution, audit:
            witnessed = candidate(y[:, :1])
        record["operator_witness"] = dict(audit.report(), execution=execution)
    del expected, actual, expected_tail, actual_tail, reference_loss, candidate_loss, witnessed
    reference.zero_grad(set_to_none=True); candidate.zero_grad(set_to_none=True)
    optimizers = (torch.optim.AdamW(reference.parameters(), lr=1e-4, foreach=True),
                  placement_adamw(candidate, lr=1e-4, foreach=True))
    states = [None, None]
    record["updates"] = []
    for step in range(steps):
        row = dict(index=step, observations=[])
        record["updates"].append(row)
        for index, (current, optimizer) in enumerate(zip((reference, candidate), optimizers)):
            owner = "reference" if index == 0 else "placed"
            states[index], observed = _train(current, optimizer, x_cpu.to(devices[0]), states[index], precision, devices,
                                             progress=mark, label=f"update.{step}.{owner}", phase_sync=phase_sync)
            row["observations"].append(dict(owner=owner, **observed))
        row["parameters"] = nested_comparison(canonical_state_dict(candidate), reference.state_dict())
        mark(f"update.{step}.complete")
    saved_state = states[1]
    mark("checkpoint.save")
    save_checkpoint(checkpoint, candidate, state=saved_state, optimizer=optimizers[1])
    mark("checkpoint.load")
    restored, restored_state, slots = load_checkpoint(checkpoint, placement_devices=devices)
    resumed = placement_adamw(restored, lr=1e-4, foreach=True)
    resumed.load_state_dict(slots)
    comparisons["checkpoint_data"] = nested_comparison(
        dict(parameters=restored.state_dict(), state=restored_state, slots=resumed.state_dict()),
        dict(parameters=candidate.state_dict(), state=saved_state, slots=optimizers[1].state_dict()), atol=0, rtol=0)
    mark("checkpoint.next_forward")
    with torch.no_grad(), precision_context(devices[0], precision):
        a = restored(x_cpu.to(devices[0]), restored_state)
        b = candidate(x_cpu.to(devices[0]), saved_state)
        comparisons["checkpoint_next_result"] = nested_comparison(a, b, atol=0, rtol=0)
    next_state_a, update_a = _train(candidate, optimizers[1], x_cpu.to(devices[0]), saved_state, precision, devices,
                                   progress=mark, label="checkpoint.original_update", phase_sync=phase_sync)
    next_state_b, update_b = _train(restored, resumed, x_cpu.to(devices[0]), restored_state, precision, devices,
                                   progress=mark, label="checkpoint.restored_update", phase_sync=phase_sync)
    record["restored_updates"] = [update_a, update_b]
    comparisons["restored_updated_parameters"] = nested_comparison(restored.state_dict(), candidate.state_dict())
    comparisons["restored_updated_optimizer"] = nested_comparison(resumed.state_dict(), optimizers[1].state_dict())
    comparisons["restored_updated_state"] = nested_comparison(next_state_b, next_state_a)
    record["state_memory"] = state_memory(next_state_b)
    record["peak_device_memory"] = _peaks(devices)
    structural = any(c["metadata_mismatches"] or c["support_mismatches"] for c in comparisons.values())
    structural |= not comparisons["checkpoint_data"]["exact"]
    finite = all(c["finite"] for c in comparisons.values()) and all(
        observation["finite"] for row in record["updates"] for observation in row["observations"])
    finite &= update_a["finite"] and update_b["finite"]
    close = all(c["within_threshold"] for c in comparisons.values()) and all(
        row["parameters"]["within_threshold"] for row in record["updates"])
    record["state"] = "nonfinite" if not finite else "structural_failed" if structural else "within_threshold" if close else "finite_deviation"
    mark("completed")
    return record


def inference(config, devices, precision, batch, length, steps, warmup, *, transport="auto",
              partition_kind="layer", region_partitions=None, region_concurrency="serial"):
    torch.manual_seed(807)
    started = time.perf_counter()
    model = _place(TideModel(config), devices, transport, partition_kind, region_partitions, region_concurrency)
    x_cpu = torch.randn(batch, length, config.input_dim or config.d)
    x = x_cpu.to(devices[0])
    _sync(devices)
    record = dict(config=config.to_dict(), precision=precision_policy(devices[0], precision),
                  placement=placement_report(model), construction_transfer_seconds=time.perf_counter()-started,
                  comparison="capacity execution only; no duplicate one-device reference",
                  input_sha256=hashlib.sha256(x_cpu.numpy().tobytes()).hexdigest())
    with torch.no_grad(), precision_context(devices[0], precision):
        for _ in range(warmup):
            _full(model, x)
        _sync(devices); _peaks(devices, reset=True)
        samples = []
        for _ in range(steps):
            before = time.perf_counter()
            result, tail = _full(model, x)
            value = result.readout()
            if tail is not None:
                value = torch.cat((value, tail.readout()), 1)
            _sync(devices)
            samples.append(time.perf_counter()-before)
            if not bool(torch.isfinite(value).all()):
                raise ArithmeticError("nonfinite placed inference output")
        timed_peaks = _peaks(devices)
        audit = RuntimeAudit()
        with serial_region_audit(model) as execution, audit:
            result, tail = _full(model, x, record_work=True)
    record.update(seconds=samples, median_seconds=statistics.median(samples),
                  input_positions_per_second=batch*length/statistics.median(samples),
                  work=dict(input=result.work, flush=None if tail is None else tail.work),
                  operator_witness=dict(audit.report(), execution=execution), peak_device_memory=timed_peaks,
                  state_memory=state_memory((result if tail is None else tail).state), state="completed")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cpu"])
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--partition-kind", choices=("layer", "region"), default="layer")
    parser.add_argument("--region-partitions", type=int, nargs="+", help="one partition per region; otherwise exact-parameter greedy balance")
    parser.add_argument("--region-concurrency", choices=("serial", "per_device"), default="serial",
                        help="region only: one same-layer task per device; CPU indices simulate logical devices")
    parser.add_argument("--transport", choices=("auto", "direct", "host_stage", "synchronized"), default="auto")
    parser.add_argument("--phase-sync", action="store_true", help="diagnostic synchronization between forward/backward/update")
    parser.add_argument("--mode", choices=("qualify", "inference", "train"), default="qualify")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--batch", type=int, help="default: train 1, qualify/inference 2")
    parser.add_argument("--length", type=int, help="default: train 8, qualify/inference 5")
    parser.add_argument("--steps", type=int, help="default: train 2, qualify/inference 3")
    parser.add_argument("--warmup", type=int, help="inference only; default 1")
    parser.add_argument("--no-foreach", action="store_true", help="capacity train only: bound AdamW tensor-list temporaries")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name, usual, training in (("batch", 2, 1), ("length", 5, 8), ("steps", 3, 2)):
        if getattr(args, name) is None:
            setattr(args, name, training if args.mode == "train" else usual)
    if args.warmup is None:
        args.warmup = 1 if args.mode == "inference" else 0
    if args.region_partitions is not None and args.partition_kind != "region":
        parser.error("--region-partitions requires --partition-kind region")
    if args.region_concurrency != "serial" and args.partition_kind != "region":
        parser.error("--region-concurrency per_device requires --partition-kind region")
    if args.no_foreach and args.mode != "train":
        parser.error("--no-foreach is only supported in capacity --mode train")
    if args.mode == "train" and args.warmup:
        parser.error("capacity training has no warmup updates; use --steps to declare every real update")
    if min(args.batch, args.length, args.steps, args.threads) < 1 or args.warmup < 0:
        parser.error("counts must be positive and warmup nonnegative")
    if args.output.exists():
        parser.error("choose a new output identity")
    if any(device.startswith("npu") for device in args.devices):
        import torch_npu
        torch.npu.set_device(args.devices[0])
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(args.threads)
    configs = [ModelConfig.from_dict(json.loads(args.config.read_text()))] if args.config else [
        ModelConfig(topology=TopologyConfig(kind="exchange", n=2, depth=1, hubs=1), d=64,
                    memory="X", sharing="independent" if args.partition_kind == "region" else "layer_shared", aggregation="incoming_light_64"),
        ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, profile="compact-5-dd4", hubs=1,
                    timing="timed"), d=64, memory="X", sharing="region_shared"),
    ]
    placement_options = ({} if args.partition_kind == "layer" else
                         dict(partition_kind="region", region_partitions=args.region_partitions,
                              region_concurrency=args.region_concurrency))
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(source=source, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), torch=str(torch.__version__),
                  exit_policy="finite_deviation retained; nonfinite/structural/execution failures exit nonzero",
                  observations=[], state="running")
    atomic(args.output, record)
    for index, config in enumerate(configs):
        record["observations"].append(dict(config=config.to_dict(), state="running", last_stage="start"))
        def progress(partial):
            record["observations"][index] = partial
            atomic(args.output, record)
        try:
            _peaks(args.devices, reset=True)
            if args.mode == "qualify":
                with tempfile.TemporaryDirectory(prefix="placement-", dir=args.output.parent) as directory:
                    row = qualify(config, args.devices, args.precision, Path(directory)/"checkpoint.pt",
                                  args.batch, args.length, args.steps, progress=progress, transport=args.transport, phase_sync=args.phase_sync, **placement_options)
            elif args.mode == "train":
                row = capacity_train(config, args.devices, args.precision, args.batch, args.length, args.steps,
                                     progress=progress, transport=args.transport, foreach=not args.no_foreach,
                                     phase_sync=args.phase_sync, **placement_options)
            else:
                row = inference(config, args.devices, args.precision, args.batch, args.length, args.steps, args.warmup,
                                transport=args.transport, **placement_options)
        except Exception as error:
            import traceback
            traceback.print_exc()
            row = record["observations"][index]
            row.update(state="execution_failed", error=repr(error))
        row["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["observations"][index] = row
        atomic(args.output, record)
        print(json.dumps(dict(index=index, state=row["state"], placement=row.get("placement"))), flush=True)
        if row["state"] == "execution_failed":
            record["not_run_after_execution_failure"] = [config.to_dict() for config in configs[index+1:]]
            break  # A backend failure can poison device state for the entire process.
    failed = any(row["state"] in {"execution_failed", "structural_failed", "nonfinite"} for row in record["observations"])
    record["state"] = "failed" if failed else "completed"
    atomic(args.output, record)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
