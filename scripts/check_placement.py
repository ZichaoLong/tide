"""Finite HB layer-shard qualification or a bounded large inference execution.

Run actual accelerators only inside an admitted whole-device-group service.
The reference is an ordinary one-device model initialized from the same CPU data.
Large --mode inference deliberately omits that duplicate capacity requirement.
"""
import argparse
from collections import OrderedDict
import copy
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

from tide import (ModelConfig, TopologyConfig, TideModel, canonical_state_dict, detach_state,
                  load_checkpoint, place_model, placement_adamw, placement_report, save_checkpoint)
from tide.placement import move_tensors
from tide.precision import precision_context, precision_policy
from benchmarks.statistics import RuntimeAudit
from benchmarks.continuous import nested_comparison, state_memory, synchronize
from scripts.job import atomic


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


def _train(model, optimizer, x, state, precision, devices):
    _sync(devices); started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with precision_context(devices[0], precision):
        result, tail = _full(model, x, state)
        loss = _objective(result)+(0 if tail is None else _objective(tail))
    loss.backward()
    optimizer.step(); _sync(devices)
    elapsed = time.perf_counter()-started
    finite = bool(torch.isfinite(loss)) and all(
        bool(torch.isfinite(p).all()) and (p.grad is None or bool(torch.isfinite(p.grad).all()))
        for p in model.parameters())
    return detach_state((result if tail is None else tail).state), dict(
        loss=float(loss.detach()), complete_seconds=elapsed, finite=finite,
        output_positive_strengths=(result.strengths.detach() > 0).sum().item(),
        flush_output_positive_strengths=0 if tail is None else (tail.strengths.detach() > 0).sum().item())


def qualify(config, devices, precision, checkpoint, batch, length, steps):
    torch.manual_seed(807)
    source = TideModel(config)
    fingerprint = hashlib.sha256()
    for name, parameter in source.named_parameters():
        fingerprint.update(name.encode()); fingerprint.update(parameter.detach().numpy().tobytes())
    record = dict(config=config.to_dict(), precision=precision_policy(devices[0], precision),
                  initial_parameter_sha256=fingerprint.hexdigest(),
                  reference="one-device ordinary model with identical initial CPU parameters and inputs")
    reference = copy.deepcopy(source).to(devices[0])
    candidate = place_model(source, devices)
    record["placement"] = placement_report(candidate)
    x_cpu = torch.randn(batch, length, config.input_dim or config.d)
    record["input_sha256"] = hashlib.sha256(x_cpu.numpy().tobytes()).hexdigest()
    x = x_cpu.to(devices[0]).requires_grad_()
    y = x_cpu.to(devices[0]).detach().clone().requires_grad_()
    comparisons = {}
    record["comparisons"] = comparisons
    with precision_context(devices[0], precision):
        expected, expected_tail = _full(reference, x, record_work=True)
        actual, actual_tail = _full(candidate, y, record_work=True)
        reference_loss = _objective(expected)+(0 if expected_tail is None else _objective(expected_tail))
        candidate_loss = _objective(actual)+(0 if actual_tail is None else _objective(actual_tail))
    reference_loss.backward(); candidate_loss.backward(); _sync(devices)
    comparisons["whole_result"] = nested_comparison(actual, expected)
    comparisons["flush_result"] = nested_comparison(actual_tail, expected_tail)
    comparisons["input_gradient"] = nested_comparison(y.grad, x.grad, rtol=1e-2)
    comparisons["parameter_gradients"] = nested_comparison(canonical_state_dict(candidate, gradients=True),
        OrderedDict((name, None if p.grad is None else p.grad.detach().cpu())
                    for name, p in reference.named_parameters()))
    record["work"] = actual.work
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
        comparisons["continued_single_vs_placed_state"] = nested_comparison(following, reference_following)
        comparisons["reference_whole_vs_continued"] = nested_comparison(reference_joined, expected.readout())
        comparisons["continued_readout"] = nested_comparison(joined, actual.readout())
        comparisons["continued_state"] = nested_comparison(following, actual.state)
        if not candidate.settle:
            tail = candidate.flush(following, record_work=True)
            comparisons["continued_flush"] = nested_comparison(tail, actual_tail)
        audit = RuntimeAudit()
        with audit:
            witnessed = candidate(y[:, :1])
        record["operator_witness"] = audit.report()
    del expected, actual, expected_tail, actual_tail, reference_loss, candidate_loss, witnessed
    reference.zero_grad(set_to_none=True); candidate.zero_grad(set_to_none=True)
    optimizers = (torch.optim.AdamW(reference.parameters(), lr=1e-4, foreach=True),
                  placement_adamw(candidate, lr=1e-4, foreach=True))
    states = [None, None]
    record["updates"] = []
    for step in range(steps):
        row = dict(index=step, observations=[])
        for index, (current, optimizer) in enumerate(zip((reference, candidate), optimizers)):
            states[index], observed = _train(current, optimizer, x_cpu.to(devices[0]), states[index], precision, devices)
            row["observations"].append(observed)
        row["parameters"] = nested_comparison(canonical_state_dict(candidate), reference.state_dict())
        record["updates"].append(row)
    saved_state = states[1]
    save_checkpoint(checkpoint, candidate, state=saved_state, optimizer=optimizers[1])
    restored, restored_state, slots = load_checkpoint(checkpoint, placement_devices=devices)
    resumed = placement_adamw(restored, lr=1e-4, foreach=True)
    resumed.load_state_dict(slots)
    comparisons["checkpoint_data"] = nested_comparison(
        dict(parameters=restored.state_dict(), state=restored_state, slots=resumed.state_dict()),
        dict(parameters=candidate.state_dict(), state=saved_state, slots=optimizers[1].state_dict()), atol=0, rtol=0)
    with torch.no_grad(), precision_context(devices[0], precision):
        a = restored(x_cpu.to(devices[0]), restored_state)
        b = candidate(x_cpu.to(devices[0]), saved_state)
        comparisons["checkpoint_next_result"] = nested_comparison(a, b, atol=0, rtol=0)
    next_state_a, update_a = _train(candidate, optimizers[1], x_cpu.to(devices[0]), saved_state, precision, devices)
    next_state_b, update_b = _train(restored, resumed, x_cpu.to(devices[0]), restored_state, precision, devices)
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
    return record


def inference(config, devices, precision, batch, length, steps, warmup):
    torch.manual_seed(807)
    started = time.perf_counter()
    model = place_model(TideModel(config), devices)
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
        with audit:
            result, tail = _full(model, x, record_work=True)
    record.update(seconds=samples, median_seconds=statistics.median(samples),
                  input_positions_per_second=batch*length/statistics.median(samples),
                  work=dict(input=result.work, flush=None if tail is None else tail.work),
                  operator_witness=audit.report(), peak_device_memory=timed_peaks,
                  state_memory=state_memory((result if tail is None else tail).state), state="completed")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cpu"])
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--mode", choices=("qualify", "inference"), default="qualify")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
                    memory="X", sharing="layer_shared", aggregation="incoming_light_64"),
        ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, profile="compact-5-dd4", hubs=1,
                    timing="timed"), d=64, memory="X", sharing="region_shared"),
    ]
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(source=source, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), torch=str(torch.__version__),
                  exit_policy="finite_deviation retained; nonfinite/structural/execution failures exit nonzero",
                  observations=[], state="running")
    atomic(args.output, record)
    for index, config in enumerate(configs):
        try:
            _peaks(args.devices, reset=True)
            if args.mode == "qualify":
                with tempfile.TemporaryDirectory(prefix="placement-", dir=args.output.parent) as directory:
                    row = qualify(config, args.devices, args.precision, Path(directory)/"checkpoint.pt",
                                  args.batch, args.length, args.steps)
            else:
                row = inference(config, args.devices, args.precision, args.batch, args.length, args.steps, args.warmup)
        except Exception as error:
            import traceback
            traceback.print_exc()
            row = dict(config=config.to_dict(), state="execution_failed", error=repr(error))
        row["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["observations"].append(row)
        atomic(args.output, record)
        print(json.dumps(dict(index=index, state=row["state"], placement=row.get("placement"))), flush=True)
    failed = any(row["state"] in {"execution_failed", "structural_failed", "nonfinite"} for row in record["observations"])
    record["state"] = "failed" if failed else "completed"
    atomic(args.output, record)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
