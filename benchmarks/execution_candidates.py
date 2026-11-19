"""Finite ordinary FP32 costs for same-source FFN execution candidates.

One invocation measures one candidate. A caller schedules serial paired repeats;
this entry never selects devices, starts services or calls a best point optimal.
Work and storage observers run separately, after every ordinary timed sample.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import resource
import statistics
import traceback

import torch

from benchmarks.baselines import TransformerBaseline, parameter_count
from benchmarks.continuous import run_stream, training_state_observation
from benchmarks.costs import TrainingAnchor, compact_work, state_hash
from benchmarks.inference import (InferenceSchedule, _check_budget, add_availability,
    allocator_stats, bounded_parameter_identity, diagnose_request, run_request,
    source_identity, synchronize_all)
from benchmarks.inference_breakdown import _reserve_output, normal_summary
from benchmarks.matching import match_transformer
from benchmarks.memory_ledger import diagnose as memory_diagnose, forward_window
from benchmarks.streaming_audit import StreamingWorkAudit
from benchmarks.training_breakdown import copy_tree, identity_tree
from benchmarks.workloads import paired_input, tensor_hash
from scripts.job import atomic
from tide import ModelConfig, TideModel
from tide.precision import precision_context, precision_policy


VARIANTS = ("baseline", "weight_access", "packing", "combined")


def variant_options(name):
    """Only W/P options vary; region batching/reception policy are inherited."""
    if name not in VARIANTS:
        raise ValueError("unknown execution candidate: " + str(name))
    options = dict(ffn_packing="power2" if name in {"packing", "combined"} else "rectangular")
    options["ffn_weight_access"] = "grouped" if name in {"weight_access", "combined"} else "gather"
    return options


def resolved_config(base, name):
    return replace(base, **variant_options(name))


def configure_device(device, threads):
    if str(device).split(":", 1)[0] == "npu":
        # Explicit CPU runs must not import the vendor runtime. Conversely,
        # disabled backend autoload needs registration before torch.device.
        import torch_npu  # noqa: F401
    device = torch.device(device)
    if device.type not in {"cpu", "npu"}:
        raise ValueError("this finite plan supports CPU or NPU only")
    torch.set_num_threads(threads)
    if device.type == "npu":
        torch.npu.set_device(device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    return device


def rss_record():
    return dict(process_lifetime_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope="Linux process-lifetime RSS high-water mark; includes construction/host copies; not device allocator bytes or per-sample incremental memory")


def training_summary(rows, batch, length):
    times = [row["complete_seconds"] for row in rows]
    return dict(samples=len(rows), complete_seconds_median=statistics.median(times),
        positions_per_second=batch*length*len(times)/sum(times),
        median_sample_positions_per_second=statistics.median(batch*length/t for t in times),
        independent_batch_sequences_per_second=batch*len(times)/sum(times),
        phase_seconds_median={key: statistics.median(row[key] for row in rows) for key in
                              ("prefill_seconds", "backward_seconds", "optimizer_seconds")},
        boundary="ordinary continuing full updates only; repeated resident input, graph across chunks, detach after each update; diagnostic timings excluded")


def training_work(model, x, state, chunk):
    """Actual graph-enabled forward from a restored training anchor, no update."""
    context = StreamingWorkAudit(model) if isinstance(model, TideModel) else nullcontext()
    with context as audit, torch.enable_grad(), precision_context(x.device, "fp32"):
        outputs, following = forward_window(model, x, state, chunk=chunk)
    synchronize_all([x.device])
    finite = all(bool(torch.isfinite(value).all()) for value in outputs)
    result = dict(finite=finite, state_position=following.position,
        actual_work=None if audit is None else compact_work(audit.report()),
        scope="separate graph-enabled full forward at the first measured training anchor; no backward/recompute or optimizer counted; no ordinary timing")
    if not finite:
        raise ArithmeticError("nonfinite training work observation")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subject", choices=("tide", "transformer"), default="tide")
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--mode", choices=("train", "inference"), default="train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--pair-id", default="unspecified")
    parser.add_argument("--optimizer-foreach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--observe-work", action="store_true")
    parser.add_argument("--memory-ledger", action="store_true")
    parser.add_argument("--transformer-layers", type=int, default=32)
    parser.add_argument("--transformer-d", type=int)
    parser.add_argument("--transformer-ffn-width", type=int)
    parser.add_argument("--transformer-head-dim", type=int, default=64)
    parser.add_argument("--transformer-window", type=int, default=128)
    parser.add_argument("--allocated-budget-gib", type=float, default=48.)
    parser.add_argument("--reserved-budget-gib", type=float, default=56.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.batch, args.length, args.chunk, args.warmup, args.samples, args.threads, args.transformer_layers) < 1:
        parser.error("counts must be positive; a real warmup establishes the training anchor")
    if args.decode < 2 or args.repeat < 0 or args.threads > 64:
        parser.error("decode must be >=2, repeat nonnegative and threads <=64")
    if any(not math.isfinite(x) or x <= 0 for x in (args.allocated_budget_gib, args.reserved_budget_gib)):
        parser.error("sampled budgets must be finite and positive")
    if args.subject == "transformer" and args.variant != "baseline":
        parser.error("Tide execution options have no Transformer counterpart")
    if (args.transformer_d is None) != (args.transformer_ffn_width is None):
        parser.error("an explicit Transformer requires both d and FFN width")
    base = ModelConfig.from_dict(json.loads(args.config.read_text()))
    try:
        config = resolved_config(base, args.variant)
        device = configure_device(args.device, args.threads)
    except ValueError as error:
        parser.error(str(error))
    if config.topology.timing != "settle":
        parser.error("this finite continuing performance plan supports Settle only")
    with torch.device("meta"):
        target = parameter_count(TideModel(config))
    if args.mode == "train" and target > 200_000_000:
        parser.error("training is bounded to 200M parameters; no 9B training or checkpoints")
    selected = match_transformer(args.transformer_layers, target, window=args.transformer_window)
    if args.transformer_d is not None:
        selected = dict(kind="explicit fixed shape", constructor=dict(d=args.transformer_d,
            ffn_width=args.transformer_ffn_width, layers=args.transformer_layers,
            head_dim=args.transformer_head_dim, window=args.transformer_window))
    record = dict(schema="tide-execution-costs-v1", state="running", source=source_identity(),
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        config=config.to_dict(), base_config=base.to_dict(), variant_options=variant_options(args.variant),
        subject=args.subject, precision=precision_policy(device, "fp32"), transformer_match=selected,
        ffn_weight_backend=("not applicable to Transformer" if args.subject != "tide" else
            "ordinary gathered BMM" if config.ffn_weight_access == "gather" else
            "CPU grouped weight-view reference" if device.type == "cpu" else "NPU grouped_matmul with explicit adjoint"),
        physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified"),
        cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(),
        torch=str(torch.__version__), host_load_before=os.getloadavg(),
        warmup=[], normal=[], diagnostics={}, last_stage="construction",
        boundary=dict(normal="ordinary resident-input run_stream full updates or run_request zero-state requests; constructor, hashes, observers and IO excluded",
            training="synthetic mean-square joined backbone readout, no task head or model-quality claim; chunks retain graphs until update",
            inference="first available backbone output, prefill and continuing stable decode; no task head, sampling or LM TTFT",
            comparisons="same-source Tide execution variants at identical semantic config; fixed-depth Transformer is auxiliary, not A-D or output equivalence",
            diagnostics="untimed separate work/storage passes after all ordinary samples; training restores the warmed first-sample anchor",
            memory="normal allocator peaks sampled before checks; RSS is process lifetime; ledger de-duplicates visible storage and leaves allocator residual unclassified",
            repeats="one invocation is one finite repeat; serial paired order and at least three repeats belong to the external driver"))
    try:
        _reserve_output(args.output, record)
    except FileExistsError:
        parser.error("choose a new output identity")
    def persist():
        atomic(args.output, record)
    try:
        torch.manual_seed(args.seed)
        model = (TideModel(config) if args.subject == "tide" else
                 TransformerBaseline(**selected["constructor"], chunk=256, backend="auto"))
        count = parameter_count(model)
        if args.mode == "train" and count > 200_000_000:
            raise ValueError("training model exceeds 200M parameter bound")
        record.update(parameters=count, initial_parameters=bounded_parameter_identity(model),
            spec=config.to_dict() if args.subject == "tide" else model.configuration())
        if not record["initial_parameters"]["finite"]:
            raise ArithmeticError("nonfinite initial parameters")
        model = model.to(device).train(args.mode == "train")
        schedule = InferenceSchedule(args.length, args.chunk, args.decode)
        width = (config.input_dim or config.d) if args.subject == "tide" else model.d
        cpu_input = paired_input(args.batch, args.length if args.mode == "train" else schedule.positions, width, args.seed+1)
        prefix = min(width, config.input_dim or config.d)
        record.update(input_sha256=tensor_hash(cpu_input), input_shape=list(cpu_input.shape),
            shared_input_prefix=dict(width=prefix, sha256=tensor_hash(cpu_input[..., :prefix])),
            schedule=None if args.mode == "train" else asdict(schedule))
        x = cpu_input.to(device)
        del cpu_input
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=args.optimizer_foreach) if args.mode == "train" else None
        state, anchor, private_anchor = None, None, None
        if optimizer is not None:
            record["optimizer"] = dict(kind="AdamW", lr=1e-4, betas=[.9, .999], eps=1e-8,
                weight_decay=.01, foreach=args.optimizer_foreach)
        def ordinary(destination):
            nonlocal state
            allocator_stats([device], reset=True)
            if optimizer is None:
                temporary, row = run_request(model, x, schedule, precision="fp32")
                del temporary
            else:
                before = 0 if state is None else state.position
                output, state, row = run_stream(model, x, args.length, args.chunk,
                    state=state, mode="train", precision="fp32", optimizer=optimizer)
                del output
                row.update(state_before=before, state_after=state.position)
            row.update(allocator=allocator_stats([device]), rss=rss_record())
            record[destination].append(row)
            persist()
            _check_budget(row["allocator"], args)
            if optimizer is not None:
                row["continuation"] = training_state_observation(model, optimizer, state)
                row["phase_times"] = {key: row[key] for key in
                    ("prefill_seconds", "backward_seconds", "optimizer_seconds", "complete_seconds")}
                if state.position != row["state_before"]+args.length:
                    raise AssertionError("training stream did not advance the whole input")
                if (not math.isfinite(row["loss"]) or not all(row[key] for key in
                        ("finite_output", "finite_parameters", "finite_gradients")) or
                        not all(row["continuation"][key] for key in
                        ("optimizer_finite", "private_state_finite", "private_state_detached"))):
                    raise ArithmeticError("nonfinite normal training state")
        for index in range(args.warmup):
            record["last_stage"] = f"warmup:{index}"
            ordinary("warmup")
        if optimizer is not None and (args.observe_work or args.memory_ledger):
            anchor, private_anchor = TrainingAnchor(model, optimizer, device), copy_tree(state, "cpu")
            record["anchor"] = dict(**anchor.identity, private_state_sha256=state_hash(identity_tree(private_anchor)),
                private_state_position=state.position, cpu_training_tensor_bytes=anchor.cpu_tensor_bytes)
        for index in range(args.samples):
            record["last_stage"] = f"normal:{index}"
            ordinary("normal")
        record["normal_end_rss"] = rss_record()
        record["normal_final_parameters"] = bounded_parameter_identity(model)
        if optimizer is not None:
            record["normal_summary"] = training_summary(record["normal"], args.batch, args.length)
            del state
        else:
            record["last_stage"] = "untimed_availability_and_optional_work"
            diagnostic, work = diagnose_request(model, x, schedule, precision="fp32", audit=args.observe_work)
            record["diagnostics"]["availability"] = diagnostic
            if not diagnostic["finite_outputs"] or not diagnostic["finite_private_state"]:
                raise ArithmeticError("nonfinite inference diagnostic")
            if work is not None:
                record["diagnostics"]["work"] = compact_work(work)
            for row in record["normal"]:
                add_availability(row, diagnostic)
            record["normal_summary"] = normal_summary(record["normal"], args.batch)
        if optimizer is not None and args.observe_work:
            record["last_stage"] = "untimed_training_work"
            anchor.restore(model, optimizer)
            record["diagnostics"]["work"] = training_work(model, x, copy_tree(private_anchor, device), args.chunk)
        if args.memory_ledger:
            record["last_stage"] = "untimed_memory_ledger"
            if anchor is not None:
                anchor.restore(model, optimizer)
            _, ledger = memory_diagnose(model, x, mode=args.mode, precision="fp32", optimizer=optimizer,
                state=copy_tree(private_anchor, device), chunk=args.chunk,
                decode=0 if args.mode == "train" else args.decode,
                device_budget_gib=args.allocated_budget_gib)
            ledger["objective_scope"] = "summed per-chunk squared readout divided by total elements; same mean-square formula with a distinct reduction order; storage witness, no timed-update equivalence claim"
            record["diagnostics"]["memory_ledger"] = ledger
            _check_budget(allocator_stats([device]), args)
        if optimizer is None and bounded_parameter_identity(model) != record["initial_parameters"]:
            raise AssertionError("inference mutated parameters")
        record.update(state="completed_finite", last_stage="complete")
    except Exception as error:
        record.update(state="failed", error=repr(error))
        traceback.print_exc()
    finally:
        record.update(final_rss=rss_record(), host_load_after=os.getloadavg())
        persist()
    print(json.dumps(dict(state=record["state"], output=str(args.output))))
    return int(record["state"] != "completed_finite")


if __name__ == "__main__":
    raise SystemExit(main())
