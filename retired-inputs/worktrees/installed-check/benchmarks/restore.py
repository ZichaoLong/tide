"""Load one trusted checkpoint in a fresh process and time one real update.

This measures restoration and the first resumed loss/backward/AdamW. It does not
construct an original-owner reference or certify trajectory equivalence. Use
continuous.checkpoint_resume for the separate paired numerical comparison.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import torch

from tide import detach_state, load_checkpoint, placement_adamw
from tide.precision import precision_context, precision_policy
from benchmarks.continuous import _tensors, forward_state, state_memory
from benchmarks.memory_ledger import optimizer_tensors, storage_inventory
from benchmarks.run import DeviceBudgetExceeded, _device_memory, _loss, synchronize
from benchmarks.workloads import paired_input, tensor_hash
from scripts.job import atomic


def checkpoint_stat(path):
    value = Path(path).stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns, device=value.st_dev, inode=value.st_ino)


def checkpoint_identity(path, expected_stat):
    """Stream large files without materializing their payload a second time."""
    if checkpoint_stat(path) != expected_stat:
        raise ValueError("checkpoint changed since loading began")
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8*2**20), b""):
            digest.update(block)
    if checkpoint_stat(path) != expected_stat:
        raise ValueError("checkpoint changed during identity hashing")
    return dict(path=str(Path(path).resolve()), sha256=digest.hexdigest(), **expected_stat)


def all_sync(devices):
    for device in devices:
        synchronize(device)


def device_memory(devices, reset=False):
    return {str(device): _device_memory(device, reset=reset) for device in devices if device.type != "cpu"}


def stage_peaks(phases, names):
    peaks = {}
    for name in names:
        for device, memory in phases[name].get("device_memory", {}).items():
            peaks[device] = max(peaks.get(device, 0), memory["peak_allocated_bytes"])
    return peaks


def phase(label, operation, devices, record, budget_gib):
    all_sync(devices)
    device_memory(devices, reset=True)
    record["last_stage"] = label
    started = time.perf_counter()
    try:
        value = operation()
        all_sync(devices)
    except Exception as error:
        record["phases"][label] = dict(state="failed", seconds=time.perf_counter()-started, error=repr(error))
        raise
    elapsed = time.perf_counter()-started
    memory = device_memory(devices)
    row = dict(state="completed", seconds=elapsed, device_memory=memory,
               process_lifetime_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    record["phases"][label] = row
    if budget_gib is not None and any(v["peak_allocated_bytes"] > budget_gib*2**30 for v in memory.values()):
        row["state"] = "device_budget_exceeded"
        raise DeviceBudgetExceeded(f"{label}: measured per-device peak exceeds {budget_gib} GiB")
    return value


def finite_scan(model, optimizer, state, values=()):
    groups = dict(parameters=list(model.parameters()), gradients=[p.grad for p in model.parameters() if p.grad is not None],
                  optimizer=optimizer_tensors(optimizer), private_state=state, outputs=values)
    result = {}
    for name, group in groups.items():
        tensors = list(_tensors(group))
        result[name] = dict(tensors=len(tensors), finite=all(bool(torch.isfinite(t).all()) for t in tensors))
    if not all(v["finite"] for v in result.values()):
        raise ArithmeticError("nonfinite restored state: "+str(result))
    return result


def optimizer_record(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    steps = {}
    for parameter, slots in optimizer.state.items():
        step = slots.get("step")
        steps[names[id(parameter)]] = None if step is None else float(step.detach().cpu()) if isinstance(step, torch.Tensor) else float(step)
    return dict(state_parameter_tensors=len(optimizer.state), steps=steps,
                groups=[{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups],
                storage=storage_inventory(optimizer_tensors(optimizer))[1])


def parameter_samples(model):
    return {name: parameter.detach().reshape(-1)[:16].cpu().clone() for name, parameter in model.named_parameters()}


def next_forward(model, inputs, state, prefill, chunk):
    schedule = [(start, min(prefill, start+chunk)) for start in range(0, prefill, chunk)]
    schedule += [(start, start+1) for start in range(prefill, inputs.shape[1])]
    values = []
    for start, stop in schedule:
        value, state, raw = forward_state(model, inputs[:, start:stop], state)
        values.append(value)
        del raw
    window_slots = sum(value.shape[0]*value.shape[1] for value in values)
    flush_slots = 0
    if not model.settle:
        tail = model.flush(state)
        value, state = tail.readout(), tail.state
        values.append(value)
        flush_slots = value.shape[0]*value.shape[1]
    return values, state, _loss(values), dict(window_output_slots=window_slots, flush_output_slots=flush_slots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-source", help="optional producing-source provenance; the checkpoint format itself does not encode it")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--placement-devices", nargs="+")
    parser.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--batch", type=int, required=True, help="must match the saved private state's batch")
    parser.add_argument("--length", type=int, default=128, help="next prefill input positions per sample")
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--decode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1108)
    parser.add_argument("--device-budget-gib", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.length, args.threads) < 1 or args.decode < 0 or args.prefill_chunk is not None and args.prefill_chunk < 1:
        parser.error("positive batch/prefill/chunk/threads and nonnegative decode required")
    if args.device_budget_gib is not None and not 0 < args.device_budget_gib < float("inf"):
        parser.error("device budget must be finite and positive")
    if args.output.exists():
        parser.error("choose a new observation identity")
    declared = args.placement_devices or [args.device]
    if any(name != "cpu" for name in declared) and args.device_budget_gib is None:
        parser.error("accelerator restoration requires an explicit per-device budget and outer launcher RSS/time bounds")
    if any(name.startswith("npu") for name in declared):
        import torch_npu
        torch.npu.set_device(next(name for name in declared if name.startswith("npu")))
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    devices = [torch.device(name) for name in dict.fromkeys(declared)]
    torch.set_num_threads(args.threads)
    all_sync(devices)
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(schema="tide-restore-observation-v1", source=source,
                  harness_source=os.environ.get("TIDE_HARNESS_COMMIT", source), state="preparing",
                  arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  checkpoint_source=args.checkpoint_source, checkpoint_source_scope="caller provenance only; producing source is not encoded in the checkpoint format",
                  torch=str(torch.__version__), python=sys.version, executable=sys.executable,
                  loaded_modules=os.environ.get("LOADEDMODULES"), physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), host_load_before=os.getloadavg(),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified"),
                  timing_boundary="fresh process, no model/update warmup; load includes model/state and optimizer restoration plus all-device synchronization; next update includes zero_grad/forward/readout/loss/backward/AdamW/detach/all sync",
                  phase_boundary="phase seconds exclude allocator reads/resets; complete load/update wall time includes those phase probes and Python bookkeeping; no on-disk record I/O inside these two intervals",
                  excluded="runtime/module/device-context setup, input generation/transfer/hash, checkpoint SHA pass, finite/storage/slot/sample checks",
                  filesystem_cache="uncontrolled; fresh process does not imply cold filesystem cache; SHA pass runs after measured loading",
                  comparison_qualification="one restored owner only; no original-owner reference, tolerance comparison or complete-trajectory equivalence claim",
                  numerical_equivalence_qualification=False,
                  operator_dtype_observation="not instrumented in timed restoration; precision policy is explicit, operator qualification is separate",
                  precision=precision_policy(devices[0], args.precision), phases={})
    atomic(args.output, record)
    before_stat = None
    try:
        before_stat = checkpoint_stat(args.checkpoint)
        record["checkpoint_stat_before_load"] = before_stat
        atomic(args.output, record)
        load_start = time.perf_counter()
        model, state, slots = phase("load_model_state", lambda: load_checkpoint(args.checkpoint, device=args.device,
                                   placement_devices=args.placement_devices), devices, record, args.device_budget_gib)
        if slots is None:
            raise ValueError("checkpoint contains no optimizer state; cannot qualify a restored training update")
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("restored FP32/BF16 policy requires saved FP32 master parameters")
        def restore_optimizer():
            owner = placement_adamw(model) if hasattr(model, "_placement_spec") else torch.optim.AdamW(model.parameters())
            owner.load_state_dict(slots)
            return owner
        optimizer = phase("load_optimizer", restore_optimizer, devices, record, args.device_budget_gib)
        if not optimizer.state:
            raise ValueError("checkpoint has no initialized AdamW slots; no prior training update is evidenced")
        del slots
        gc.collect()
        all_sync(devices)
        record["load_complete_seconds"] = time.perf_counter()-load_start
        record["load_peak_device_bytes"] = stage_peaks(record["phases"], ("load_model_state", "load_optimizer"))
        model.train(True)
        record.update(state="running", config=model.config.to_dict(), placement=getattr(model, "_placement_spec", None),
                      parameters=sum(p.numel() for p in model.parameters()),
                      parameter_storage=storage_inventory(list(model.parameters()))[1],
                      loaded_state_position=0 if state is None else state.position,
                      loaded_state_memory=state_memory(state), optimizer_before=optimizer_record(model, optimizer))
        record["loaded_finite"] = phase("validate_loaded", lambda: finite_scan(model, optimizer, state), devices, record, args.device_budget_gib)
        identity_start = time.perf_counter()
        record["checkpoint_file"] = checkpoint_identity(args.checkpoint, before_stat)
        record["checkpoint_hash_seconds"] = time.perf_counter()-identity_start
        if args.expected_checkpoint_sha256 and args.expected_checkpoint_sha256 != record["checkpoint_file"]["sha256"]:
            raise ValueError("checkpoint SHA256 does not match the expected immutable input")
        width = model.config.input_dim or model.config.d
        input_start = time.perf_counter()
        input_cpu = paired_input(args.batch, args.length+args.decode, width, args.seed)
        inputs = input_cpu.to(devices[0])
        all_sync(devices)
        record["input_generation_transfer_seconds"] = time.perf_counter()-input_start
        record.update(input_shape=list(input_cpu.shape), input_sha256=tensor_hash(input_cpu),
                      input_positions=args.batch*(args.length+args.decode))
        samples_before = parameter_samples(model)
        atomic(args.output, record)
        # No profiler, WorkAudit, saved-tensor hooks, second model or second update.
        update_start = time.perf_counter()
        phase("zero_grad", lambda: optimizer.zero_grad(set_to_none=True), devices, record, args.device_budget_gib)
        def forward():
            with precision_context(devices[0], args.precision):
                return next_forward(model, inputs, state, args.length, args.prefill_chunk or args.length)
        values, following, loss, windows = phase("forward_loss_flush", forward, devices, record, args.device_budget_gib)
        phase("backward", loss.backward, devices, record, args.device_budget_gib)
        phase("optimizer", optimizer.step, devices, record, args.device_budget_gib)
        following = phase("detach", lambda: detach_state(following), devices, record, args.device_budget_gib)
        all_sync(devices)
        record["next_update_complete_seconds"] = time.perf_counter()-update_start
        record["next_update_peak_device_bytes"] = stage_peaks(record["phases"], ("zero_grad", "forward_loss_flush", "backward", "optimizer", "detach"))
        record.update(next_update_input_positions_per_second=record["input_positions"]/record["next_update_complete_seconds"],
                      loss=float(loss.detach()), output_windows=windows,
                      restored_state_position=following.position, restored_state_memory=state_memory(following),
                      private_state_detached=all(not t.requires_grad for t in _tensors(following)))
        record["updated_finite"] = phase("validate_updated", lambda: finite_scan(model, optimizer, following, values), devices, record, args.device_budget_gib)
        record["optimizer_after"] = optimizer_record(model, optimizer)
        before_steps, after_steps = record["optimizer_before"]["steps"], record["optimizer_after"]["steps"]
        increment_errors = []
        updated_parameters = 0
        for name, parameter in model.named_parameters():
            expected = before_steps.get(name) or 0
            if parameter.grad is not None:
                expected += 1
                updated_parameters += 1
            if (after_steps.get(name) or 0) != expected:
                increment_errors.append(name)
        samples_after = parameter_samples(model)
        changed = {name: int((sample != samples_before[name]).sum()) for name, sample in samples_after.items()}
        record.update(gradient_storage=storage_inventory([p.grad for p in model.parameters() if p.grad is not None])[1],
                      optimizer_step_increment_errors=increment_errors, parameter_tensors_with_gradients=updated_parameters,
                      parameter_update_sample=dict(scope="first up to 16 scalars of every Parameter; not a full trajectory comparison",
                                                   sampled_scalars=sum(value.numel() for value in samples_after.values()),
                                                   changed_scalars=sum(changed.values()), changed_by_parameter=changed))
        if increment_errors or not record["private_state_detached"] or following.position <= record["loaded_state_position"]:
            raise AssertionError("restored update slot/clock/detach invariant failed")
        if model.settle and following.position-record["loaded_state_position"] != args.length+args.decode:
            raise AssertionError("Settle restored clock differs from actual input count")
        record["state"] = "completed_finite"
    except Exception as error:
        record.update(state="failed", error=repr(error))
        import traceback
        traceback.print_exc()
    if "checkpoint_file" not in record and before_stat is not None:
        try:
            record["checkpoint_file"] = checkpoint_identity(args.checkpoint, before_stat)
        except Exception as error:
            record["checkpoint_identity_error"] = repr(error)
    record["host_load_after"] = os.getloadavg()
    atomic(args.output, record)
    print(json.dumps(dict(output=str(args.output), state=record["state"],
                          load_complete_seconds=record.get("load_complete_seconds"),
                          next_update_complete_seconds=record.get("next_update_complete_seconds"))))
    return int(record["state"] != "completed_finite")


if __name__ == "__main__":
    raise SystemExit(main())
