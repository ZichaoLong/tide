"""Finite ordinary inference and separate synchronized stage/operator diagnosis.

One subject owns one device. Every warmup, normal sample and diagnostic starts
from zero private state, then continues the complete prefill/decode schedule.
Diagnostic checks and host copies occur between captured calls, never inside
ordinary timing or a stage/operator scope. No task head or token sampler exists.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time
import traceback

import torch

from benchmarks.baselines import TransformerBaseline, parameter_count
from benchmarks.continuous import forward_state
from benchmarks.detailed_stages import DetailedStageTimes, OperatorTimes, detailed_stage_hooks
from benchmarks.inference import (InferenceSchedule, OutputDiagnostic, _check_budget,
    _clear_model_cache, add_availability, allocator_stats, bounded_parameter_identity,
    execution_devices, run_request, source_identity, synchronize_all)
from benchmarks.memory_decode_costs import state_metadata
from benchmarks.workloads import paired_input, tensor_hash
from scripts.job import atomic
from tide import ModelConfig, TideModel
from tide.precision import precision_context, precision_policy


def output_identity(tensor, chunk_bytes=8 << 20):
    """Hash all logical output bytes in C order with bounded host transfers."""
    if chunk_bytes < tensor.element_size():
        raise ValueError("hash chunk must hold at least one tensor element")
    digest = hashlib.sha256()
    elements = chunk_bytes // tensor.element_size()

    def visit(value):
        if not value.numel():
            return
        if value.numel() <= elements:
            piece = value.detach().cpu().contiguous()
            digest.update(piece.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif value.is_contiguous():
            flat = value.detach().view(-1)
            for start in range(0, flat.numel(), elements):
                visit(flat[start:start+elements])
        else:
            row_elements = value.numel() // value.shape[0]
            if row_elements > elements:
                for index in range(value.shape[0]):
                    visit(value[index])
            else:
                rows = max(1, elements // row_elements)
                for start in range(0, value.shape[0], rows):
                    visit(value[start:start+rows])

    visit(tensor)
    return dict(sha256=digest.hexdigest(), shape=list(tensor.shape), dtype=str(tensor.dtype),
                logical_bytes=tensor.numel()*tensor.element_size(), host_copy_chunk_bytes=chunk_bytes)


def private_state_metadata(model, state):
    result = state_metadata(state)
    if isinstance(model, TideModel):
        result["clock_scope"] = "actual private int64 time tensors; observed metadata, not a write oracle"
    else:
        # TransformerState stores position and contiguous KV, not time tensors.
        # Do not turn all([]) from the generic helper into observed dtype proof.
        result["times_int64"] = None
        result["clock_scope"] = "implicit contiguous positions from state.position; no stored private time tensor"
        result["cache_position_bounds"] = [
            dict(layer=i, start=state.position-key.shape[-2], stop=state.position)
            for i, key in enumerate(state.keys)]
    return result


def output_metadata(value, raw, before, state):
    result = dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                  identities=dict(readout=output_identity(value)))
    if raw is None:
        result["time"] = dict(source="implicit contiguous input positions; no raw output-time tensor",
                              start=before, stop=state.position)
        result["support_sha256"] = None
    else:
        if raw.times.dtype != torch.int64:
            raise ArithmeticError("Tide output times are not original int64 values")
        result["time"] = dict(source="actual Tide output times", dtype=str(raw.times.dtype),
                              values=raw.times.detach().cpu().tolist())
        result["support_sha256"] = tensor_hash((raw.strengths > 0).detach())
        result["strength_dtype"] = str(raw.strengths.dtype)
        result["identities"].update(raw_values=output_identity(raw.values),
                                    raw_strengths=output_identity(raw.strengths))
    return result


def observe(model, x, schedule, kind, args, row, persist, parameter_identity, input_sha256):
    """Replay one full request, capturing only forward/readout/sync and release."""
    devices = execution_devices(model, x)
    observer = DetailedStageTimes(devices[0]) if kind == "stages" else OperatorTimes(devices[0])
    row.update(state="running", calls=[], initial_parameters=bounded_parameter_identity(model),
               input_sha256=tensor_hash(x), capture_start_position=0,
               normal_timing=False, throughput_reported=False)
    if row["initial_parameters"] != parameter_identity or row["input_sha256"] != input_sha256:
        raise AssertionError("diagnostic parameters or inputs differ from the normal request identity")
    _clear_model_cache(devices)
    allocator_stats(devices, reset=True)
    output, state = OutputDiagnostic(), None
    persist()
    try:
        with torch.no_grad(), precision_context(x.device, args.precision), detailed_stage_hooks(model, observer):
            for index, (phase, start, stop) in enumerate(schedule.calls()):
                row["last_stage"] = f"{phase}/call_{index:03d}"
                before = 0 if state is None else state.position
                if before != start:
                    raise AssertionError("private state does not continue the actual input schedule")
                with (observer if kind == "operators" else nullcontext()), observer.scope(phase), observer.scope(f"call_{index:03d}"):
                    value, state, raw = forward_state(model, x[:, start:stop], state)
                    synchronize_all(devices)
                call = dict(index=index, phase=phase, start=start, stop=stop,
                            input_positions=x.shape[0]*(stop-start), state_before=before,
                            state_after=state.position, allocator_after_forward=allocator_stats(devices))
                row["calls"].append(call)
                _check_budget(call["allocator_after_forward"], args)
                if state.position != stop:
                    raise AssertionError("Settle/Transformer state did not advance by the input length")
                # No observer context or scope is active during these copies,
                # reductions and state checks. The hook session calls no model.
                call["output"] = output_metadata(value, raw, before, state)
                output(value, raw, phase, state)
                call["output"]["finite_through_call"] = output.finite
                call["output"]["available"] = output.available[-1]
                call["private_state"] = private_state_metadata(model, state)
                if not output.finite:
                    raise ArithmeticError("nonfinite diagnostic output")
                # Release has a separate root. It must not make a captured
                # forward interval include the intervening metadata checks.
                with observer.scope(f"release/{phase}/call_{index:03d}"):
                    del value, raw
                    synchronize_all(devices)
                call["allocator_including_checks"] = allocator_stats(devices)
                _check_budget(call["allocator_including_checks"], args)
                row["report"] = observer.report()
                persist()
        row["outputs"] = output.report(state)
        row["final_state"] = private_state_metadata(model, state)
        row["allocator_including_final_checks"] = allocator_stats(devices)
        _check_budget(row["allocator_including_final_checks"], args)
        row.update(state="completed_finite", last_stage="complete")
    finally:
        row["report"] = observer.report()
        # A sampled refusal still retains every completed captured scope.
        persist()


def _replay_metadata(row):
    return dict(outputs=row["outputs"], final_state=row["final_state"], calls=[
        {**{key: call[key] for key in ("index", "phase", "start", "stop", "input_positions",
                                      "state_before", "state_after", "private_state")},
         "output": {key: value for key, value in call["output"].items() if key != "identities"}}
        for call in row["calls"]])


def output_identity_comparison(reference, current):
    differences = []
    for left, right in zip(reference["calls"], current["calls"]):
        a, b = left["output"]["identities"], right["output"]["identities"]
        changed = {key: dict(reference=a.get(key), current=b.get(key))
                   for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
        if changed:
            differences.append(dict(index=right["index"], phase=right["phase"],
                                    start=right["start"], stop=right["stop"], identities=changed))
    return dict(exact_hashes_identical=not differences, differing_calls=differences,
                scope="complete readout and raw Tide value/strength bytes; finite hash differences are retained without a rounding explanation; this is not tolerance/VJP qualification")


def normal_summary(observations, batch):
    def median(get):
        values = [get(row) for row in observations]
        return statistics.median(values) if all(value is not None for value in values) else None
    devices = {device for row in observations for device in row["allocator"]}
    return dict(samples=len(observations),
        first_backbone_output_seconds=median(lambda r: r["first_any_backbone_output_seconds"]),
        prefill_positions_per_second=median(lambda r: r["prefill"]["positions_per_second"]),
        stable_decode_positions_per_second=median(lambda r: r["stable_decode"]["positions_per_second"]),
        complete_positions_per_second=median(lambda r: r["complete_positions_per_second"]),
        independent_complete_sequences_per_second=median(lambda r: batch/r["complete_seconds"]),
        complete_seconds=median(lambda r: r["complete_seconds"]),
        normal_peak_bytes_by_device={device: {
            kind: max(row["allocator"][device]["peak_"+kind+"_bytes"] for row in observations)
            for kind in ("allocated", "reserved")} for device in sorted(devices)},
        boundary="medians of ordinary samples only; peak is maximum sampled normal allocator peak; diagnostics excluded")


def _reserve_output(path, record):
    """Claim a new identity without racing an existing file or symlink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=("tide", "transformer"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--transformer-d", type=int, default=4352)
    parser.add_argument("--transformer-ffn-width", type=int, default=15208)
    parser.add_argument("--transformer-layers", type=int, default=32)
    parser.add_argument("--transformer-head-dim", type=int, default=64)
    parser.add_argument("--transformer-window", type=int, default=128)
    parser.add_argument("--transformer-chunk", type=int, default=256)
    parser.add_argument("--transformer-backend", choices=("auto", "math", "tiled", "sdpa", "npu_fusion"), default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int, default=32)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument("--diagnostics", nargs="+", default=["stages", "operators"],
                        help="stages operators, or stages,operators; independent full-request passes")
    parser.add_argument("--allocated-budget-gib", type=float, default=48.)
    parser.add_argument("--reserved-budget-gib", type=float, default=56.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.batch, args.threads, args.samples, args.decode) < 1 or args.warmup < 0:
        parser.error("batch, threads, samples and decode must be positive; warmup may be zero")
    args.diagnostics = [part for value in args.diagnostics for part in value.split(",")]
    if (not args.diagnostics or set(args.diagnostics)-{"stages", "operators"}
            or len(set(args.diagnostics)) != len(args.diagnostics)):
        parser.error("choose each of stages/operators at most once")
    try:
        schedule = InferenceSchedule(args.prefill, args.prefill_chunk, args.decode)
    except ValueError as error:
        parser.error(str(error))
    for limit in (args.allocated_budget_gib, args.reserved_budget_gib):
        if not math.isfinite(limit) or limit <= 0:
            parser.error("sampled budgets must be finite and positive")
    if args.output.exists() or args.output.is_symlink():
        parser.error("choose a new output identity")
    config = None
    if args.subject == "tide":
        if args.config is None:
            parser.error("Tide requires --config")
        config = ModelConfig.from_dict(json.loads(args.config.read_text()))
        if config.topology.timing != "settle":
            parser.error("this finite diagnosis supports Settle only; no Timed flush")
    elif args.config is not None:
        parser.error("Transformer shape comes from --transformer-*; --config is Tide-only")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda", "npu"}:
        parser.error("one CPU, CUDA or NPU device is required")
    torch.set_num_threads(args.threads)
    record = dict(schema="tide-inference-breakdown-v1", state="running", source=source_identity(),
        subject=args.subject, config=None if config is None else config.to_dict(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        schedule=asdict(schedule), precision=precision_policy(device, args.precision),
        physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified"),
        cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(),
        torch=str(torch.__version__), host_load_before=os.getloadavg(),
        normal=dict(state="pending", observations=[], warmup=[]), diagnostics={}, last_stage="construction",
        comparison=dict(kind="different-function backbone cost reference; not A-D matching",
                        mathematical_equivalence=False, model_quality_claim=False),
        boundary=dict(
            requests="one resident model and input; zero private state per warmup, normal sample and diagnostic; real continuation within each request",
            normal="ordinary run_request includes forward/readout/synchronization/scheduling/output release; construction, hashes, checks, JSON IO and diagnostics excluded",
            normal_checks="no per-call output checks in normal timing; callwise dtype/finite/support/time checks come from independent diagnostics",
            capture="all prefill and decode forward/readout/sync calls; release recorded as a separate root after excluded checks",
            checks="per-call output/state checks, hashes, host copies, allocator reads and JSON IO outside stage scopes and operator context",
            stages="synchronized intrusive wall time; nested inclusive rows overlap; exclusive rows are not normal-time shares",
            operators="separate synchronized dispatch diagnosis; logical bytes are not bus traffic or peak storage; opaque native kernels remain opaque",
            replay_checks="same-candidate support/time/private-state metadata checked separately from complete output SHA; finite output-byte differences retained, no tolerance/VJP equivalence qualification",
            output_identity="every diagnostic readout and raw Tide value/strength fully hashed with bounded 8 MiB host pieces outside capture; normal samples retain no output payload or hash",
            allocator="sampled 48/56 GiB defaults, not hard allocator caps; normal peaks precede diagnostics; diagnostic peaks include between-call checks",
            hardware="single device only; no shared-device admission or multi-device path",
            first_output="first available backbone output from a separate support witness; no language-model TTFT"))
    if device.type == "npu":
        record.update(torch_npu=str(torch_npu.__version__), allow_hf32=dict(matmul=False, conv=False))
    try:
        _reserve_output(args.output, record)
    except FileExistsError:
        parser.error("choose a new output identity")

    def persist():
        atomic(args.output, record)

    current = None
    try:
        torch.manual_seed(args.seed)
        model = (TideModel(config) if args.subject == "tide" else TransformerBaseline(
            d=args.transformer_d, layers=args.transformer_layers, ffn_width=args.transformer_ffn_width,
            head_dim=args.transformer_head_dim, window=args.transformer_window,
            chunk=args.transformer_chunk, backend=args.transformer_backend)).eval()
        record["initial_parameters"] = bounded_parameter_identity(model)
        if not record["initial_parameters"]["finite"]:
            raise ArithmeticError("nonfinite initial parameters")
        record["parameters"] = parameter_count(model)
        record["spec"] = config.to_dict() if config is not None else model.configuration()
        model = model.to(device)
        width = (config.input_dim or config.d) if config is not None else model.d
        inputs = paired_input(args.batch, schedule.positions, width, args.seed+1)
        shared_width = min(width, 768)
        record.update(input_sha256=tensor_hash(inputs), input_shape=list(inputs.shape),
                      shared_input_prefix=dict(width=shared_width, sha256=tensor_hash(inputs[..., :shared_width]),
                          rule="feature-wise CPU seed; same sample/time/feature prefix at different widths"),
                      placement=dict(kind="single", devices=[str(device)]))
        x = inputs.to(device)
        del inputs
        devices = execution_devices(model, x)
        if len(devices) != 1:
            raise ValueError("inference breakdown requires exactly one execution device")
        synchronize_all(devices)
        record["construction_allocator"] = allocator_stats(devices)
        _check_budget(record["construction_allocator"], args)
        current = record["normal"]
        current["state"] = "running"
        _clear_model_cache(devices)
        for index in range(args.warmup):
            record["last_stage"] = f"normal/warmup_{index}"
            persist()
            allocator_stats(devices, reset=True)
            state, _ = run_request(model, x, schedule, precision=args.precision)
            memory = allocator_stats(devices)
            current["warmup"].append(dict(final_state_position=state.position, allocator=memory))
            del state
            _check_budget(memory, args)
        for index in range(args.samples):
            record["last_stage"] = f"normal/sample_{index}"
            persist()
            allocator_stats(devices, reset=True)
            state, measured = run_request(model, x, schedule, precision=args.precision)
            measured["allocator"] = allocator_stats(devices)
            current["observations"].append(measured)
            if state.position != schedule.positions:
                raise AssertionError("normal request did not advance the full input schedule")
            del state
            _check_budget(measured["allocator"], args)
        current["state"] = "completed"
        persist()
        reference, reference_row, reference_kind = None, None, None
        record["replay_checks"] = {}
        for kind in args.diagnostics:
            record["last_stage"] = kind
            current = record["diagnostics"][kind] = {}
            begin = time.perf_counter()
            observe(model, x, schedule, kind, args, current, persist,
                    record["initial_parameters"], record["input_sha256"])
            current["pass_elapsed_seconds_including_excluded_work"] = time.perf_counter()-begin
            metadata = _replay_metadata(current)
            if reference is None:
                reference, reference_row, reference_kind = metadata, current, kind
                for measured in record["normal"]["observations"]:
                    add_availability(measured, current["outputs"])
                record["normal"]["summary"] = normal_summary(record["normal"]["observations"], args.batch)
            else:
                equal = metadata == reference
                record["replay_checks"]["independent_diagnostic_metadata_identical"] = equal
                record["replay_checks"]["output_identity"] = dict(
                    reference_diagnostic=reference_kind, current_diagnostic=kind,
                    **output_identity_comparison(reference_row, current))
                if not equal:
                    raise AssertionError("independent diagnostic replays differ in output support/time or private-state metadata")
            persist()
        record["final_parameters"] = bounded_parameter_identity(model)
        if record["final_parameters"] != record["initial_parameters"]:
            raise AssertionError("inference or diagnostics mutated parameters")
        record.update(state="completed_finite", last_stage="complete")
    except Exception as error:
        if current is not None:
            current.update(state="failed", error=repr(error))
        budget_refused = (isinstance(error, MemoryError) and " observed " in str(error)
                          and " peak exceeds " in str(error) and str(error).endswith("GiB sampled budget"))
        record.update(state="failed", error=repr(error),
                      failure_kind="sampled_budget_refusal" if budget_refused else "execution_or_check_failed")
        traceback.print_exc()
    finally:
        record.update(process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      host_load_after=os.getloadavg())
        persist()
    print(json.dumps(dict(state=record["state"], output=str(args.output))))
    return int(record["state"] != "completed_finite")


if __name__ == "__main__":
    raise SystemExit(main())
