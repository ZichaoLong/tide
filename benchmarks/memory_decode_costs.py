"""Intrusive, bounded Settle decode diagnosis after a real history prefill.

Stages and runtime shapes replay the same prefix independently, at unchanged
parameters/input. Neither pass supplies normal timing or throughput. Metadata,
finite checks and host copies stay outside the captured forward/readout scopes.
Use an external frozen-input job runner for a hard time/process resource bound.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields, is_dataclass
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback

import torch

from benchmarks.baselines import parameter_count
from benchmarks.continuous import _tensors, forward_state, state_memory
from benchmarks.inference import (InferenceSchedule, OutputDiagnostic, _check_budget,
    _clear_model_cache, allocator_stats, bounded_parameter_identity, execution_devices,
    run_request, source_identity, synchronize_all)
from benchmarks.stages import ShapeAudit, StageTimes, stage_hooks
from benchmarks.workloads import paired_input, tensor_hash
from scripts.job import atomic
from tide import ModelConfig, TideModel
from tide.kv import KVState
from tide.precision import precision_context, precision_policy


def state_metadata(state):
    """Shapes, original integer clocks and positive history; no payload copies.

    This is an observed-state summary, not an independent positive-write oracle
    or a full floating-point state identity. Existing cache qualification owns
    the independent write/eviction comparisons.
    """
    tensors, clocks, histories = [], [], []

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            tensors.append(dict(path=path, shape=list(value.shape), stride=list(value.stride()),
                                dtype=str(value.dtype), device=str(value.device)))
            if path.endswith(".times"):
                clock = value.detach().cpu()
                clocks.append(dict(path=path, dtype=str(clock.dtype), shape=list(clock.shape),
                    sha256=tensor_hash(clock), minimum=int(clock.min()) if clock.numel() else None,
                    maximum=int(clock.max()) if clock.numel() else None))
        elif is_dataclass(value):
            if isinstance(value, KVState):
                times, positive = value.times.detach().cpu(), (value.weight.detach() > 0).cpu()
                rows = [times[i, j][positive[i, j]].tolist()
                        for i in range(times.shape[0]) for j in range(times.shape[1])]
                histories.append(dict(path=path, owner_sample_shape=list(times.shape[:2]),
                    slots=times.shape[2], support_sha256=tensor_hash(positive),
                    positive_times_sha256=tensor_hash(times[positive]),
                    positive_counts=[len(row) for row in rows],
                    first_positive_times=[min(row) if row else None for row in rows],
                    last_positive_times=[max(row) if row else None for row in rows]))
            for field in fields(value):
                visit(getattr(value, field.name), path+"."+field.name)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, path+"."+str(key))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, path+"."+str(index))

    visit(state, "state")
    result = dict(position=state.position, storage=state_memory(state), tensors=tensors,
                  private_times=clocks, kv_history=histories,
                  finite=all(not t.is_floating_point() or bool(torch.isfinite(t).all()) for t in _tensors(state)),
                  detached=all(not t.requires_grad for t in _tensors(state)),
                  times_int64=all(row["dtype"] == "torch.int64" for row in clocks))
    if not result["finite"] or not result["detached"] or not result["times_int64"]:
        raise ArithmeticError("private state is nonfinite, attached, or has non-int64 clocks")
    return result


def observe(model, x, schedule, kind, args, row, persist, parameter_identity, input_sha256):
    devices = execution_devices(model, x)
    observer = StageTimes(devices[0]) if kind == "stages" else ShapeAudit()
    row.update(state="running", calls=[], initial_parameters=bounded_parameter_identity(model),
               input_sha256=tensor_hash(x), last_stage="prefill")
    persist()
    if row["initial_parameters"] != parameter_identity or row["input_sha256"] != input_sha256:
        raise AssertionError("diagnostic parameters or inputs differ from initial identity")
    _clear_model_cache(devices)
    allocator_stats(devices, reset=True)
    prefix = InferenceSchedule(schedule.prefill, schedule.prefill_chunk, 0)
    prefix_output = OutputDiagnostic()
    state, _ = run_request(model, x[:, :schedule.prefill], prefix, precision=args.precision,
                           observer=prefix_output, timing=False)
    row["prefix_outputs"] = prefix_output.report(state)
    if not row["prefix_outputs"]["finite_outputs"]:
        raise ArithmeticError("nonfinite prefill output")
    row["prefix_allocator"] = allocator_stats(devices)
    _check_budget(row["prefix_allocator"], args)
    row["before_decode"] = state_metadata(state)
    row["prefix_allocator_including_checks"] = allocator_stats(devices)
    _check_budget(row["prefix_allocator_including_checks"], args)
    row["capture_start_position"] = state.position
    row["last_stage"] = "decode"
    allocator_stats(devices, reset=True)
    output = OutputDiagnostic()
    persist()
    # Global stage wrappers are exclusive to this single-threaded model owner.
    # TorchDispatchMode is entered only around device work, never metadata.
    with torch.no_grad(), precision_context(x.device, args.precision), stage_hooks(model, observer):
        for phase, start, stop in schedule.calls():
            if phase != "decode":
                continue
            before = state.position
            with (observer if kind == "runtime" else nullcontext()), observer.scope("decode"):
                value, state, raw = forward_state(model, x[:, start:stop], state)
                synchronize_all(devices)
            memory = allocator_stats(devices)
            row["calls"].append(dict(start=start, stop=stop, state_before=before,
                state_after=state.position, output_shape=list(value.shape),
                output_dtype=str(value.dtype), allocator_after_forward=memory))
            _check_budget(memory, args)
            # Checks may synchronize/copy; neither observer captures them.
            output(value, raw, phase, state)
            with observer.scope("release"):
                del value, raw
                synchronize_all(devices)
            row["calls"][-1]["allocator_after_release"] = allocator_stats(devices)
            _check_budget(row["calls"][-1]["allocator_after_release"], args)
            persist()
    row["report"] = observer.report()
    if kind == "runtime":
        row["report"]["interpretation"] = (
            "separate intrusive decode-only dispatch pass after the recorded prefill; "
            "shapes include padding; logical materialized bytes are not hardware traffic; "
            "native attention/scan work is opaque here")
    row["decode_allocator_before_final_checks"] = allocator_stats(devices)
    row["outputs"] = output.report(state)
    row["after_decode"] = state_metadata(state)
    row["allocator_including_checks"] = allocator_stats(devices)
    _check_budget(row["allocator_including_checks"], args)
    if not row["outputs"]["finite_outputs"]:
        raise ArithmeticError("nonfinite decode output")
    row.update(state="completed_finite", last_stage="complete")
    del state
    persist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int, default=32)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument("--allocated-budget-gib", type=float, default=48.)
    parser.add_argument("--reserved-budget-gib", type=float, default=56.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.batch, args.threads, args.decode) < 1:
        parser.error("batch, threads and decode must be positive")
    try:
        schedule = InferenceSchedule(args.prefill, args.prefill_chunk, args.decode)
    except ValueError as error:
        parser.error(str(error))
    for limit in (args.allocated_budget_gib, args.reserved_budget_gib):
        if not math.isfinite(limit) or limit <= 0:
            parser.error("sampled budgets must be finite and positive")
    if args.output.exists():
        parser.error("choose a new output identity")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda", "npu"}:
        parser.error("one CPU, CUDA or NPU device is required")
    torch.set_num_threads(args.threads)
    config = ModelConfig.from_dict(json.loads(args.config.read_text()))
    if config.topology.timing != "settle":
        parser.error("this bounded decode diagnosis supports Settle only; no intermediate Timed flush")
    record = dict(schema="tide-memory-decode-costs-v1", state="running", source=source_identity(),
        config=config.to_dict(), arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        schedule=asdict(schedule), precision=precision_policy(device, args.precision),
        physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified"),
        cpu_affinity=sorted(os.sched_getaffinity(0)), torch=str(torch.__version__),
        normal_timing=False, throughput_reported=False, diagnostics={}, last_stage="construction",
        boundary=dict(
            replay="same unchanged model and resident input; each diagnostic independently replays zero-state prefill",
            capture="decode forward, readout and all-device sync; stages also records output release separately",
            excluded="construction, prefill, parameter/input hashing, metadata/finite checks, host copies and JSON IO",
            parameters="full bounded-copy hash before each pass; final hash must equal the initial CPU constructor hash",
            clocks="actual raw int64 private clocks and positive-history hashes/ranges; metadata, not a write oracle",
            allocator="sampled construction/prefill/call peaks; decode peaks reset after prefix checks but include between-call output checks; final inclusive-check peak separate; not normal inference memory or a hard allocator cap",
            interpretation="intrusive diagnostics only; synchronized stages change scheduling; exclusive times are not shares of existing normal timing; logical materialization bytes are not bus traffic",
            hardware="one device, serial region execution; no multi-device coverage",
            scope="finite fixed history window; no task head, ordinary throughput, candidate equivalence or model-quality claim"))
    if device.type == "npu":
        record.update(torch_npu=str(torch_npu.__version__), allow_hf32=dict(matmul=False, conv=False))
    def persist():
        atomic(args.output, record)
    persist()
    current = None
    try:
        torch.manual_seed(args.seed)
        model = TideModel(config).eval()
        record["initial_parameters"] = bounded_parameter_identity(model)
        if not record["initial_parameters"]["finite"]:
            raise ArithmeticError("nonfinite initial parameters")
        record["parameters"] = parameter_count(model)
        model = model.to(device)
        inputs = paired_input(args.batch, schedule.positions, config.input_dim or config.d, args.seed+1)
        record.update(input_sha256=tensor_hash(inputs), input_shape=list(inputs.shape),
                      placement=dict(kind="single", devices=[str(device)]))
        x = inputs.to(device)
        del inputs
        devices = execution_devices(model, x)
        synchronize_all(devices)
        record["construction_allocator"] = allocator_stats(devices)
        _check_budget(record["construction_allocator"], args)
        persist()
        for kind in ("stages", "runtime"):
            record["last_stage"] = kind
            current = record["diagnostics"][kind] = {}
            begin = time.perf_counter()
            observe(model, x, schedule, kind, args, current, persist,
                    record["initial_parameters"], record["input_sha256"])
            current["pass_elapsed_seconds_including_excluded_work"] = time.perf_counter()-begin
            if kind == "runtime" and current["before_decode"] != record["diagnostics"]["stages"]["before_decode"]:
                raise AssertionError("independent prefix replays have different private-state metadata")
            persist()
        record["final_parameters"] = bounded_parameter_identity(model)
        if record["final_parameters"] != record["initial_parameters"]:
            raise AssertionError("diagnostic mutated parameters")
        record.update(state="completed_finite", last_stage="complete")
    except Exception as error:
        if current is not None:
            current.update(state="failed", error=repr(error))
        budget_refused = (isinstance(error, MemoryError) and
                          " observed " in str(error) and " peak exceeds " in str(error) and
                          str(error).endswith("GiB sampled budget"))
        record.update(state="failed", error=repr(error),
                      failure_kind="sampled_budget_refusal" if budget_refused else "execution_or_check_failed")
        traceback.print_exc()
    finally:
        record["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        persist()
    print(json.dumps(dict(state=record["state"], output=str(args.output))))
    return int(record["state"] != "completed_finite")


if __name__ == "__main__":
    raise SystemExit(main())
