"""Finite continuing training with separately restored stage/operator diagnostics.

The Transformer is an explicitly fixed-depth near-total-parameter reference,
not an A-D replacement. All diagnostics replay the first measured update from
the same private state, parameters, gradients, AdamW slots and RNG anchor.
"""
import argparse
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
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
from benchmarks.continuous import (nested_comparison,
    run_stream, training_state_observation)
from benchmarks.costs import TrainingAnchor, _cpu_copy, state_hash
from benchmarks.detailed_stages import DetailedStageTimes, OperatorTimes, detailed_stage_hooks
from benchmarks.inference import (_check_budget, allocator_stats, bounded_parameter_identity,
    source_identity, synchronize_all)
from benchmarks.matching import match_transformer
from benchmarks.workloads import paired_input, tensor_hash
from scripts.job import atomic
from tide import ModelConfig, TideModel, detach_state
from tide.precision import precision_context, precision_policy


def copy_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device, copy=True)
    if is_dataclass(value):
        return type(value)(**{f.name: copy_tree(getattr(value, f.name), device) for f in fields(value)})
    if isinstance(value, dict):
        return {k: copy_tree(v, device) for k, v in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(copy_tree(v, device) for v in value))
    if isinstance(value, (list, tuple)):
        return type(value)(copy_tree(v, device) for v in value)
    return value


def identity_tree(value):
    """Represent dataclasses structurally; never hash their abbreviated repr."""
    if is_dataclass(value):
        return dict(type=type(value).__name__, fields={f.name: identity_tree(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, dict):
        return {k: identity_tree(v) for k, v in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return dict(type=type(value).__name__, fields={key: identity_tree(getattr(value, key)) for key in value._fields})
    if isinstance(value, (list, tuple)):
        return type(value)(identity_tree(v) for v in value)
    return value


def update_snapshot(model, optimizer, state, output):
    return dict(parameters=_cpu_copy(model.state_dict()), optimizer=_cpu_copy(optimizer.state_dict()),
                gradients=_cpu_copy({name: p.grad for name, p in model.named_parameters()}),
                private_state=copy_tree(state, "cpu"), output=copy_tree(output, "cpu"))


def snapshot_identity(snapshot):
    return {key: state_hash(identity_tree(value)) for key, value in snapshot.items()}


def observed_update(model, x, state, optimizer, chunk, observer):
    """Same Settle stream/loss/update as run_stream; hooks only in this pass."""
    values = []
    with observer.scope("training"):
        with observer.scope("zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with observer.scope("forward"):
            for start in range(0, x.shape[1], chunk):
                with observer.scope("chunk"), precision_context(x.device, "fp32"):
                    with observer.scope("backbone"):
                        # Separate readout from the backbone for Tide.
                        if isinstance(model, TideModel):
                            raw = model(x[:, start:start+chunk], state)
                            state = raw.state
                        else:
                            value, state = model(x[:, start:start+chunk], state, return_state=True)
                            raw = None
                    with observer.scope("readout"):
                        if raw is not None:
                            value = raw.readout()
                    values.append(value)
                    del raw, value
        with observer.scope("loss"):
            joined = torch.cat(values, 1)
            loss = joined.float().square().mean()
        with observer.scope("backward"):
            loss.backward()
        with observer.scope("optimizer"):
            optimizer.step()
        with observer.scope("detach"):
            state = detach_state(state)
        with observer.scope("synchronize"):
            synchronize_all([x.device])
    scalar = float(loss.detach())
    if not math.isfinite(scalar):
        raise ArithmeticError("nonfinite observed training loss")
    return joined, state, scalar


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subject", choices=("tide", "transformer"), default="tide")
    parser.add_argument("--transformer-layers", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument("--optimizer-foreach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostics", nargs="+", choices=("stages", "operators"), default=["stages", "operators"])
    parser.add_argument("--allocated-budget-gib", type=float, default=48.)
    parser.add_argument("--reserved-budget-gib", type=float, default=56.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.batch, args.length, args.chunk, args.samples, args.threads, args.transformer_layers) < 1 or args.warmup < 1:
        parser.error("counts must be positive; at least one warmup establishes a nonempty anchor")
    if len(set(args.diagnostics)) != len(args.diagnostics):
        parser.error("diagnostic names must be unique")
    if any(not math.isfinite(x) or x <= 0 for x in (args.allocated_budget_gib, args.reserved_budget_gib)):
        parser.error("sampled budgets must be finite and positive")
    if args.output.exists():
        parser.error("choose a new output identity")
    config = ModelConfig.from_dict(json.loads(args.config.read_text()))
    if config.topology.timing != "settle":
        parser.error("this continuing cost witness supports Settle only")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    with torch.device("meta"):
        target = parameter_count(TideModel(config))
    selected = match_transformer(args.transformer_layers, target, window=128)
    record = dict(schema="tide-continuing-training-breakdown-v1", state="running", source=source_identity(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        config=config.to_dict(), subject=args.subject, precision=precision_policy(device, "fp32"),
        physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY"),
        cpu_affinity=sorted(os.sched_getaffinity(0)), torch=str(torch.__version__), host_load_before=os.getloadavg(),
        transformer_match=selected, comparison="fixed-depth near-total auxiliary; not A-D or mathematical equivalence",
        warmup=[], normal=[], diagnostics={}, last_stage="construction",
        boundary=dict(normal="run_stream complete FP32 AdamW update; chunks retain graphs, state detaches between updates; repeated resident input; constructor/hash/checks/transfers excluded",
            diagnostics="independent replay of first measured update from warmed CPU anchor; stages and operators run separately; synchronization changes scheduling",
            training="synthetic mean-square backbone objective; no task head or model-quality claim",
            scope="single-device finite continuing mechanism/performance check; not 9B training qualification",
            output_comparison="readout values and private-state support/metadata; raw output strengths/times are not retained by ordinary run_stream",
            normal_backward="run_stream backward interval can include pending concatenation/loss work; pure loss/backward stages come from the separate synchronized diagnostic"))
    def persist():
        atomic(args.output, record)
    persist()
    try:
        torch.manual_seed(args.seed)
        model = (TideModel(config) if args.subject == "tide" else
                 TransformerBaseline(**selected["constructor"], chunk=256, backend="auto"))
        record.update(parameters=parameter_count(model), initial_parameters=bounded_parameter_identity(model),
                      spec=config.to_dict() if args.subject == "tide" else model.configuration())
        expected_count = target if args.subject == "tide" else selected["actual"]["parameters"]
        if record["parameters"] != expected_count or not record["initial_parameters"]["finite"]:
            raise AssertionError("constructor count/finite check failed")
        model = model.to(device).train()
        width = (config.input_dim or config.d) if args.subject == "tide" else model.d
        cpu_input = paired_input(args.batch, args.length, width, args.seed+1)
        prefix_width = config.input_dim or config.d
        record.update(input_sha256=tensor_hash(cpu_input), input_shape=list(cpu_input.shape),
                      shared_input_prefix=dict(width=prefix_width, sha256=tensor_hash(cpu_input[..., :prefix_width])))
        x = cpu_input.to(device)
        del cpu_input
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=args.optimizer_foreach)
        record["optimizer"] = dict(kind="AdamW", lr=1e-4, betas=[0.9, 0.999], eps=1e-8,
                                   weight_decay=0.01, foreach=args.optimizer_foreach)
        state = None
        def ordinary(destination):
            nonlocal state
            before = 0 if state is None else state.position
            allocator_stats([device], reset=True)
            output, state, measured = run_stream(model, x, args.length, args.chunk,
                state=state, mode="train", precision="fp32", optimizer=optimizer)
            memory = allocator_stats([device])
            measured["allocator"] = memory
            measured.update(state_before=before, state_after=state.position)
            record[destination].append(measured)
            persist()  # Retain actual peaks even when the following check refuses.
            _check_budget(memory, args)
            measured["continuation"] = training_state_observation(model, optimizer, state)
            if state.position != before+args.length:
                raise AssertionError("continuing training did not advance by the input length")
            if (not math.isfinite(measured["loss"]) or
                    not all(measured[k] for k in ("finite_output", "finite_parameters", "finite_gradients")) or
                    not all(measured["continuation"][k] for k in
                            ("optimizer_finite", "private_state_finite", "private_state_detached"))):
                raise ArithmeticError("nonfinite normal training state")
            return output, measured
        for index in range(args.warmup):
            record["last_stage"] = f"warmup:{index}"
            output, row = ordinary("warmup")
            del output
            persist()
        anchor = TrainingAnchor(model, optimizer, device)
        private_anchor = copy_tree(state, "cpu")
        private_identity = state_hash(identity_tree(private_anchor))
        record["anchor"] = dict(**anchor.identity, private_state_sha256=private_identity,
                                private_state_position=state.position, cpu_training_tensor_bytes=anchor.cpu_tensor_bytes)
        reference = None
        for index in range(args.samples):
            record["last_stage"] = f"normal:{index}"
            output, row = ordinary("normal")
            if index == 0:
                reference = update_snapshot(model, optimizer, state, output)
                record["first_measured_update"] = snapshot_identity(reference)
            del output
            persist()
        times = [row["complete_seconds"] for row in record["normal"]]
        record["normal_summary"] = dict(samples=len(times), complete_seconds_median=statistics.median(times),
            positions_per_second=args.batch*args.length*len(times)/sum(times),
            median_sample_positions_per_second=statistics.median(args.batch*args.length/t for t in times),
            phase_seconds_median={k: statistics.median(row[k] for row in record["normal"]) for k in
                                  ("prefill_seconds", "backward_seconds", "optimizer_seconds")})
        del state
        for kind in args.diagnostics:
            record["last_stage"] = kind
            row = record["diagnostics"][kind] = dict(state="running")
            persist()
            allocator_stats([device], reset=True)
            row["anchor_restored"] = anchor.restore(model, optimizer)
            state = copy_tree(private_anchor, device)
            row["private_state_start_sha256"] = state_hash(identity_tree(state))
            if row["private_state_start_sha256"] != private_identity:
                raise AssertionError("private state anchor restore failed")
            row["restore_allocator"] = allocator_stats([device])
            persist()
            _check_budget(row["restore_allocator"], args)
            observer = DetailedStageTimes(device) if kind == "stages" else OperatorTimes(device)
            allocator_stats([device], reset=True)
            begin = time.perf_counter()
            with detailed_stage_hooks(model, observer), (observer if kind == "operators" else nullcontext()):
                output, state, loss = observed_update(model, x, state, optimizer, args.chunk, observer)
            row["intrusive_elapsed_seconds"] = time.perf_counter()-begin
            row.update(loss=loss, report=observer.report(), allocator=allocator_stats([device]))
            row.update(state_before=private_anchor.position, state_after=state.position)
            persist()
            _check_budget(row["allocator"], args)
            row["continuation"] = training_state_observation(model, optimizer, state)
            if not all(row["continuation"][k] for k in
                       ("optimizer_finite", "private_state_finite", "private_state_detached")):
                raise ArithmeticError("diagnostic continuation is nonfinite or attached")
            actual = update_snapshot(model, optimizer, state, output)
            row["update_identity"] = snapshot_identity(actual)
            row["normal_update_comparison"] = {k: nested_comparison(actual[k], reference[k]) for k in actual}
            checks = list(row["normal_update_comparison"].values())
            structural = any(not c["finite"] or c["metadata_mismatches"] or c["support_mismatches"] for c in checks)
            row["numerical_state"] = ("structural_failed" if structural else "within_threshold" if
                all(c["within_threshold"] for c in checks) else "finite_deviation")
            if structural:
                raise ArithmeticError("diagnostic changed state metadata/support or produced nonfinite values")
            row["state"] = "completed_finite"
            del actual, state, output
            persist()
        record.update(state="completed_finite", last_stage="complete", host_load_after=os.getloadavg())
    except Exception as error:
        record.update(state="failed", error=repr(error))
        if record["last_stage"] in record["diagnostics"]:
            record["diagnostics"][record["last_stage"]].update(state="failed", error=repr(error))
        traceback.print_exc()
    finally:
        record["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        persist()
    print(json.dumps(dict(state=record["state"], output=str(args.output))))
    return int(record["state"] != "completed_finite")


if __name__ == "__main__":
    raise SystemExit(main())
