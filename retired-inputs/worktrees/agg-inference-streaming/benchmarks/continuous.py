"""Stateful prefill/decode, full-stream training and actual checkpoint updates.

No task head is included: first output means first available backbone output.
Every chunk advances real private state; Transformer uses its incremental cache.
"""
import argparse
from dataclasses import fields, is_dataclass
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time

import torch

from tide import ModelConfig, TideModel, detach_state, load_checkpoint, save_checkpoint
from tide.precision import precision_context, precision_policy
from benchmarks.baselines import TransformerBaseline, parameter_count
from benchmarks.matching import match_transformers
from benchmarks.statistics import WorkAudit
from benchmarks.workloads import paired_input, parameter_hash, tensor_hash


def synchronize(device):
    device = torch.device(device)
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _tensors(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def state_memory(state):
    """Both visible payload and unique retained storage, including view tails."""
    tensors = list(_tensors(state))
    stores = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
              for t in tensors}
    return dict(payload_bytes=sum(t.numel()*t.element_size() for t in tensors),
                retained_storage_bytes=sum(stores.values()),
                dtypes=sorted({str(t.dtype) for t in tensors}), tensor_count=len(tensors))


def training_state_observation(model, optimizer, state):
    """Untimed continuation invariants, including actual retained KV history.

    The largest slot count is capacity, not a count of valid writes. Distinct
    original timestamps and positive weights remain in the saved state.
    """
    from tide.kv import KVState
    slots = []
    def visit(value):
        if isinstance(value, KVState):
            slots.append(value.times.shape[-1])
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
    visit(state)
    steps = [] if optimizer is None else [float(slot["step"]) for slot in optimizer.state.values()
                                         if "step" in slot]
    tensors = [] if optimizer is None else list(_tensors(optimizer.state))
    return dict(private_state_finite=all(not t.is_floating_point() or bool(torch.isfinite(t).all())
                                        for t in _tensors(state)),
                private_state_detached=all(not t.requires_grad for t in _tensors(state)),
                optimizer_finite=all(not t.is_floating_point() or bool(torch.isfinite(t).all()) for t in tensors),
                optimizer_parameter_owners=0 if optimizer is None else len(optimizer.state),
                optimizer_step_min=min(steps) if steps else None, optimizer_step_max=max(steps) if steps else None,
                kv_retained_slot_min=min(slots) if slots else None, kv_retained_slot_max=max(slots) if slots else None,
                kv_slot_limits={kind: memory.slots for kind, memory in getattr(model, "memories", {}).items()
                                if hasattr(memory, "slots")})


def nested_comparison(actual, expected, *, atol=1e-6, rtol=2e-3):
    """Keep metadata/support, finite values and original thresholds separate."""
    result = dict(exact=True, finite=True, metadata_mismatches=[], support_mismatches=[],
                  numerical_failures=[], max_abs=0.0, max_scaled_error=0.0,
                  atol=atol, rtol=rtol, floating_tensors=0)

    def visit(a, b, path, semantic_support=False):
        if type(a) is not type(b):
            result["metadata_mismatches"].append(path+":type")
            result["exact"] = False
        elif isinstance(a, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                result["metadata_mismatches"].append(path+":shape/dtype")
                result["exact"] = False
                return
            a, b = a.detach().cpu(), b.detach().cpu()
            result["exact"] &= torch.equal(a, b)
            if not a.is_floating_point():
                if not torch.equal(a, b):
                    result["metadata_mismatches"].append(path+":value")
                return
            result["floating_tensors"] += 1
            if semantic_support and not torch.equal(a > 0, b > 0):
                result["support_mismatches"].append(path)
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            result["finite"] &= finite
            if not finite:
                result["numerical_failures"].append(dict(path=path, nonfinite=True))
                return
            if not a.numel():
                return
            delta = (a.double()-b.double()).abs()
            absolute = float(delta.max())
            scaled = float((delta/(atol+rtol*b.double().abs())).max()) if atol or rtol else (0.0 if not absolute else None)
            result["max_abs"] = max(result["max_abs"], absolute)
            if scaled is not None:
                result["max_scaled_error"] = max(result["max_scaled_error"], scaled)
            if scaled is None or scaled > 1:
                result["numerical_failures"].append(dict(path=path, max_abs=absolute, scaled_error=scaled))
        elif is_dataclass(a):
            from tide.kv import KVState
            from tide.model import LayerOutput, Result
            for field in fields(a):
                support = (isinstance(a, KVState) and field.name == "weight" or
                           isinstance(a, (LayerOutput, Result)) and field.name == "strengths")
                visit(getattr(a, field.name), getattr(b, field.name), path+"."+field.name, support)
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                result["metadata_mismatches"].append(path+":keys")
                result["exact"] = False
            for key in a.keys() & b.keys():
                visit(a[key], b[key], path+"."+str(key))
        elif isinstance(a, (tuple, list)):
            if len(a) != len(b):
                result["metadata_mismatches"].append(path+":length")
                result["exact"] = False
            for index, (left, right) in enumerate(zip(a, b)):
                visit(left, right, path+f".{index}")
        elif a != b:
            result["metadata_mismatches"].append(path+":value")
            result["exact"] = False
    visit(actual, expected, "root")
    result["numerical_failures"].sort(key=lambda row: row["path"])
    result["within_threshold"] = not (result["metadata_mismatches"] or result["numerical_failures"])
    return result


def forward_state(model, x, state=None):
    if isinstance(model, TideModel):
        result = model(x, state)
        return result.readout(), result.state, result
    value, state = model(x, state, return_state=True)
    return value, state, None


def checkpoint_resume(model, optimizer, state, next_input, precision, path):
    """Save/restore and execute one real update on both copies, retaining errors.

    Mutates the supplied model/optimizer by one update. State is detached at this
    declared truncated-backpropagation boundary. No threshold failure is hidden.
    """
    if not isinstance(model, TideModel):
        raise TypeError("this checkpoint API belongs to TideModel")
    if Path(path).exists():
        raise FileExistsError(path)
    state = detach_state(state)
    synchronize(next_input.device)
    start = time.perf_counter()
    save_checkpoint(path, model, state=state, optimizer=optimizer)
    save_seconds = time.perf_counter()-start
    start = time.perf_counter()
    restored, saved, slots = load_checkpoint(path, device=next_input.device)
    resumed_optimizer = torch.optim.AdamW(restored.parameters())
    resumed_optimizer.load_state_dict(slots)
    synchronize(next_input.device)
    load_seconds = time.perf_counter()-start
    record = dict(save_seconds=save_seconds, load_seconds=load_seconds,
                  file_bytes=Path(path).stat().st_size,
                  boundary="state detached before checkpoint; complete next window plus required Timed flush contributes to one loss/backward/AdamW on each owner",
                  saved_data=nested_comparison(
                      dict(parameters=restored.state_dict(), state=saved, optimizer=resumed_optimizer.state_dict()),
                      dict(parameters=model.state_dict(), state=state, optimizer=optimizer.state_dict()), atol=0, rtol=0))
    with torch.no_grad(), precision_context(next_input.device, precision):
        a, sa, ra = forward_state(restored, next_input, saved)
        b, sb, rb = forward_state(model, next_input, state)
        record["next_forward"] = nested_comparison(a, b, atol=0, rtol=0)
        record["next_state"] = nested_comparison(sa, sb, atol=0, rtol=0)
        if not model.settle:
            flush_a, flush_b = restored.flush(sa), model.flush(sb)
            record["next_flush_result"] = nested_comparison(flush_a, flush_b, atol=0, rtol=0)
    gradients, inputs, losses, results, windows = [], [], [], [], []
    for current, step_optimizer, current_state in ((model, optimizer, state),
                                                  (restored, resumed_optimizer, saved)):
        step_optimizer.zero_grad(set_to_none=True)
        x = next_input.detach().clone().requires_grad_(True)
        with precision_context(x.device, precision):
            value, following, _ = forward_state(current, x, current_state)
            window_slots, flush_slots = value.shape[1], 0
            if not current.settle:
                flushed = current.flush(following)
                flush_value, following = flushed.readout(), flushed.state
                flush_slots = flush_value.shape[1]
                value = torch.cat((value, flush_value), 1)
            loss = value.float().square().mean()
        windows.append(dict(input_positions=x.shape[0]*x.shape[1],
                            window_output_slots=x.shape[0]*window_slots,
                            flush_output_slots=x.shape[0]*flush_slots,
                            final_state_position=following.position))
        loss.backward()
        gradients.append({name: None if p.grad is None else p.grad.detach().clone()
                          for name, p in current.named_parameters()})
        inputs.append(x.grad)
        results.append(detach_state(following))
        losses.append(float(loss.detach()))
        step_optimizer.step()
    synchronize(next_input.device)
    record.update(losses=losses, update_windows=windows, input_gradients=nested_comparison(inputs[1], inputs[0], rtol=1e-2),
                  parameter_gradients=nested_comparison(gradients[1], gradients[0]),
                  updated_parameters=nested_comparison(restored.state_dict(), model.state_dict()),
                  updated_optimizer=nested_comparison(resumed_optimizer.state_dict(), optimizer.state_dict()),
                  updated_state=nested_comparison(results[1], results[0]))
    checks = [value for value in record.values() if isinstance(value, dict) and "finite" in value]
    structural = any(c["metadata_mismatches"] for c in checks) or not record["saved_data"]["exact"]
    finite = all(c["finite"] for c in checks)
    record["state"] = ("nonfinite" if not finite else "structural_failed" if structural else
                       "within_threshold" if all(c["within_threshold"] for c in checks) else "finite_deviation")
    return record


def run_stream(model, x, prefill, prefill_chunk, *, state=None, mode="inference", precision="fp32",
               optimizer=None):
    """Prefill chunks, single-position decode, then complete Timed flush.

    Full training retains the graph across every chunk and across the KV window,
    builds one scalar loss, performs backward/AdamW, and detaches only afterwards.
    The next call can therefore continue a stream after a truncated update.
    """
    if not 0 < prefill <= x.shape[1] or prefill_chunk < 1:
        raise ValueError("prefill/chunk must be positive and prefill fit the input")
    if (mode == "train") != (optimizer is not None):
        raise ValueError("train mode requires an optimizer; other modes must omit it")
    if mode not in {"inference", "grad_forward", "train"}:
        raise ValueError("unknown execution mode")
    schedule = [("prefill", start, min(prefill, start+prefill_chunk))
                for start in range(0, prefill, prefill_chunk)]
    schedule += [("decode", start, start+1) for start in range(prefill, x.shape[1])]
    outputs, observations, availability = [], [], []
    synchronize(x.device)
    stream_start = time.perf_counter()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(mode != "inference"):
        for phase, start, stop in schedule:
            before = time.perf_counter()
            with precision_context(x.device, precision):
                value, state, raw = forward_state(model, x[:, start:stop], state)
            synchronize(x.device)
            end = time.perf_counter()
            outputs.append(value)
            # Tensor support is inspected after timing; no implicit host scalar
            # extraction is charged as a backbone operator.
            availability.append(None if raw is None else raw.strengths.detach())
            observations.append(dict(phase=phase, inputs=stop-start, seconds=end-before,
                                     output_ready_seconds=end-stream_start))
        input_end = time.perf_counter()
        if isinstance(model, TideModel) and not model.settle:
            before = time.perf_counter()
            with precision_context(x.device, precision):
                result = model.flush(state)
                value, state = result.readout(), result.state
            synchronize(x.device)
            end = time.perf_counter()
            outputs.append(value); availability.append(result.strengths.detach())
            observations.append(dict(phase="flush", inputs=0, logical_ticks=value.shape[1],
                                     seconds=end-before, output_ready_seconds=end-stream_start))
        forward_end = time.perf_counter()
        joined = torch.cat(outputs, 1)
        loss = joined.float().square().mean() if optimizer is not None else None
    backward_seconds = update_seconds = 0.0
    if optimizer is not None:
        before = time.perf_counter()
        loss.backward(); synchronize(x.device)
        after_backward = time.perf_counter()
        optimizer.step(); synchronize(x.device)
        update_end = time.perf_counter()
        backward_seconds, update_seconds = after_backward-before, update_end-after_backward
        state = detach_state(state)  # Required update boundary belongs to complete time.
    else:
        # The final concatenation is asynchronous too. Phase synchronizations
        # precede it, so forward-only complete time needs its own ready boundary.
        synchronize(x.device)
    total_end = time.perf_counter()
    # Checks and state accounting are excluded from the measured stream.
    first = next((row["output_ready_seconds"] for row, support in zip(observations, availability)
                  if support is None or bool((support > 0).any())), None)
    prefills = [row["seconds"] for row in observations if row["phase"] == "prefill"]
    decodes = [row["seconds"] for row in observations if row["phase"] == "decode"]
    flush = sum(row["seconds"] for row in observations if row["phase"] == "flush")
    stable = decodes[1:] if len(decodes) > 1 else decodes
    phases = dict(prefill_seconds=sum(prefills), prefill_positions_per_second=x.shape[0]*prefill/sum(prefills),
                  first_backbone_output_seconds=first, decode_seconds=decodes,
                  stable_decode_median_seconds=statistics.median(stable) if stable else None,
                  stable_decode_positions_per_second=x.shape[0]*len(stable)/sum(stable) if stable else None,
                  stable_decode_rule="omit first decode when at least two are present",
                  input_window_seconds=input_end-stream_start, flush_seconds=flush,
                  input_plus_flush_seconds=forward_end-stream_start,
                  backward_seconds=backward_seconds, optimizer_seconds=update_seconds,
                  complete_seconds=total_end-stream_start, input_positions=x.shape[0]*x.shape[1],
                  output_slots=joined.shape[0]*joined.shape[1], calls=observations,
                  state_memory=state_memory(state), loss=None if loss is None else float(loss.detach()))
    phases["finite_output"] = bool(torch.isfinite(joined).all())
    phases["finite_parameters"] = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    phases["finite_gradients"] = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    return joined, state, phases


def continuous_joint_match(config, joint_class, *, device, precision, mode, batch, prefill,
                           prefill_chunk, decode, seed):
    """Match from the exact initial stream schedule, outside all timed samples.

    Training uses its graph-enabled forward before any update. WorkAudit sees
    each actual chunk, decode, and necessary Timed flush; no fresh-prefill audit
    from another batch/length/trajectory supplies the active-parameter target.
    """
    if joint_class not in {"A", "B", "C", "D"}:
        raise ValueError("joint match must be A, B, C or D")
    torch.manual_seed(seed)
    model = TideModel(config)
    initial_hash = parameter_hash(model)
    input_cpu = paired_input(batch, prefill+decode, config.input_dim or config.d, seed+1)
    nodes, depth = len(model.graph.nodes), len(model.graph.widths)
    model = model.to(device)
    audit_mode = "grad_forward" if mode == "train" else mode
    model.train(audit_mode != "inference")
    with WorkAudit(model) as audit:
        value, state, phase = run_stream(model, input_cpu.to(device), prefill, prefill_chunk,
                                         mode=audit_mode, precision=precision)
    if not (phase["finite_output"] and phase["finite_parameters"] and phase["finite_gradients"]):
        raise ArithmeticError("nonfinite Tide initial stream cannot define a joint comparison")
    measured = audit.report()
    selected = match_transformers(nodes, depth, measured["parameters"]["unique_parameters"],
                                   measured["effective_touched"]["mean"])[joint_class]
    metadata = dict(kind="joint", joint_class=joint_class, match=selected, initial_audit=measured,
                    tide_initial_parameter_sha256=initial_hash, tide_input_sha256=tensor_hash(input_cpu),
                    tide_input_shape=list(input_cpu.shape), audit_device=str(device),
                    audit_requested_mode=mode, audit_effective_mode=audit_mode,
                    audit_precision=precision_policy(device, precision),
                    audit_boundary=dict(initial_state="zero", batch=batch, prefill=prefill,
                                        prefill_chunk=prefill_chunk, decode=decode,
                                        timed_flush=not model.settle, optimizer_updates=0),
                    matching_boundary="initial actual stream only; fixed for every later timed update/continuation",
                    comparisons_equivalent=False)
    # Do not retain the graph, audit hooks or Tide weights while timing the
    # matched Transformer. The hashes/statistics above contain no live tensors.
    del value, state, model, audit
    return selected["constructor"], metadata


def _persist(path, record):
    from scripts.job import atomic
    atomic(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-overrides", type=json.loads, default={},
                        help="JSON top-level config overrides, included in the resolved Tide spec")
    parser.add_argument("--subject", choices=("tide", "transformer"), default="tide")
    parser.add_argument("--joint-match", choices=("A", "B", "C", "D"))
    parser.add_argument("--transformer-width", type=int)
    parser.add_argument("--transformer-layers", type=int)
    parser.add_argument("--ffn-width", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--transformer-chunk", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--mode", choices=("inference", "grad_forward", "train"), default="inference")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--decode", type=int, default=8)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument("--optimizer-foreach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-budget-gib", type=float,
                        help="sampled allocated peak limit; not a hard allocator cap")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not isinstance(args.config_overrides, dict):
        parser.error("config overrides must be a JSON object")
    if min(args.batch, args.prefill, args.prefill_chunk, args.updates, args.threads, args.transformer_chunk) < 1 or min(args.decode, args.warmup) < 0:
        parser.error("counts must be positive; decode/warmup can be zero")
    if args.device_budget_gib is not None and (not math.isfinite(args.device_budget_gib) or args.device_budget_gib <= 0):
        parser.error("device budget must be finite and positive")
    if args.output.exists() or args.checkpoint is not None and args.checkpoint.exists():
        parser.error("choose new output/checkpoint identities")
    if args.checkpoint is not None and (args.subject != "tide" or args.mode != "train"):
        parser.error("checkpoint requires Tide train mode")
    if args.joint_match is not None and args.subject != "transformer":
        parser.error("--joint-match requires --subject transformer")
    if args.joint_match is not None and any(value is not None for value in
            (args.transformer_width, args.transformer_layers, args.ffn_width, args.head_dim)):
        parser.error("joint matching chooses width/layers/FFN; omit explicit constructor overrides")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = ModelConfig.from_dict(json.loads(Path(args.config).read_text()) | args.config_overrides)
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(source=source, harness_source=os.environ.get("TIDE_HARNESS_COMMIT", source),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  precision=precision_policy(args.device, args.precision),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), torch=str(torch.__version__),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified"),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), host_load_before=os.getloadavg(),
                  timing="input transfer/construction/initial matching audit/warmup/checkpoint excluded; full stream train includes concatenation/loss/backward/AdamW/sync",
                  training_boundary="full BPTT within prefill+decode+flush; detach after each optimizer update is included in complete time",
                  grad_forward_boundary="state detached between measured observations; graph retained across every chunk within each segment",
                  segment_boundary="Timed flush ends each measured segment and advances its clock; the next segment resumes that flushed state",
                  objective="mechanism-only mean squared readout over every returned window and flush slot",
                  initial_state="zero; subsequent measured streams continue previous private state",
                  first_output="first available backbone output, no task head or language-model TTFT",
                  input_pairing="feature-wise seeded CPU inputs; the shared feature prefix is identical at different model widths",
                  process_peak_rss_scope="cumulative process high-water including construction and initial Tide matching audit; not isolated baseline peak",
                  observations=[], state="preparing", last_stage="construct_or_match")
    _persist(args.output, record)
    try:
        if args.subject == "tide":
            model = TideModel(config)
            width, spec = config.input_dim or config.d, config.to_dict()
            comparison = dict(kind="tide", comparisons_equivalent=False)
        else:
            if args.joint_match is not None:
                constructor, comparison = continuous_joint_match(
                    config, args.joint_match, device=args.device, precision=args.precision, mode=args.mode,
                    batch=args.batch, prefill=args.prefill, prefill_chunk=args.prefill_chunk,
                    decode=args.decode, seed=args.seed)
                torch.manual_seed(args.seed)
                model = TransformerBaseline(**dict(constructor, chunk=args.transformer_chunk))
                resolved = torch.device(args.device)
                if resolved.type in {"npu", "cuda"}:
                    getattr(torch, resolved.type).empty_cache()
            else:
                if args.transformer_width is None or args.transformer_layers is None:
                    raise ValueError("Transformer requires --joint-match or explicit width/layers")
                model = TransformerBaseline(args.transformer_width, args.transformer_layers,
                                            ffn_width=args.ffn_width, head_dim=args.head_dim or 32,
                                            chunk=args.transformer_chunk)
                comparison = dict(kind="auxiliary_explicit_constructor", comparisons_equivalent=False,
                                  qualification="explicit width/layers do not establish any of the four joint matches")
            width, spec = model.d, model.configuration()
        x_cpu = paired_input(args.batch, args.prefill+args.decode, width, args.seed+1)
        prefix = min(width, config.input_dim or config.d)
        comparison["audit_source"] = source if args.joint_match is not None else None
        comparison["shared_input_prefix_width"] = prefix
        comparison["shared_input_prefix_sha256"] = tensor_hash(x_cpu[..., :prefix])
        record.update(spec=spec, parameters=parameter_count(model), comparison=comparison,
                      initial_parameters_sha256=parameter_hash(model), input_sha256=tensor_hash(x_cpu),
                      input_shape=list(x_cpu.shape), state="running", last_stage="warmup")
        _persist(args.output, record)
    except Exception as error:
        import traceback
        traceback.print_exc()
        record.update(state="execution_failed", error=repr(error))
        _persist(args.output, record)
        return 1
    from benchmarks.run import check_memory, DeviceBudgetExceeded, _device_memory
    device = torch.device(args.device)
    failed = False
    try:
        model = model.to(args.device)
        model.train(args.mode != "inference")
        x = x_cpu.to(args.device)
        # Warmup is a separate inference owner state and does not update parameters.
        for _ in range(args.warmup):
            run_stream(model, x, args.prefill, args.prefill_chunk, precision=args.precision)
            check_memory(device, args.device_budget_gib)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=args.optimizer_foreach) if args.mode == "train" else None
        state = None
        for step in range(args.updates):
            record["last_stage"] = f"update_{step}"
            device = torch.device(args.device)
            if device.type == "npu":
                torch.npu.reset_peak_memory_stats()
            elif device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            value, state, row = run_stream(model, x, args.prefill, args.prefill_chunk, state=state,
                                          mode=args.mode, precision=args.precision, optimizer=optimizer)
            state = detach_state(state)  # grad_forward has no cross-observation backward.
            row.update(index=step, state_position=state.position,
                       peak_device_bytes=torch.npu.max_memory_allocated() if device.type == "npu" else
                       torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                       process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            row["device_memory"] = _device_memory(device)
            record["observations"].append(row)
            check_memory(device, args.device_budget_gib)
            row["continuation"] = training_state_observation(model, optimizer, state)
            if not (row["finite_output"] and row["finite_parameters"] and row["finite_gradients"]):
                raise ArithmeticError("nonfinite continuous stream")
            if not all(row["continuation"][key] for key in ("private_state_finite", "private_state_detached", "optimizer_finite")):
                raise ArithmeticError("invalid detached state or nonfinite optimizer in continued stream")
            _persist(args.output, record)
            del value
        if args.checkpoint is not None:
            record["last_stage"] = "checkpoint_and_paired_update"
            record["checkpoint"] = checkpoint_resume(model, optimizer, state, x[:, :min(args.prefill_chunk, x.shape[1])],
                                                      args.precision, args.checkpoint)
            failed |= record["checkpoint"]["state"] in {"nonfinite", "structural_failed"}
            record["checkpoint_device_memory"] = check_memory(device, args.device_budget_gib)
        record["state"] = "failed" if failed else "completed"
    except Exception as error:
        import traceback
        traceback.print_exc()
        record.update(state="execution_failed", error=repr(error),
                      failure_kind="device_budget_exceeded" if isinstance(error, DeviceBudgetExceeded) else "execution_failed")
        record["failure_device_memory"] = _device_memory(device)
        failed = True
    _persist(args.output, record)
    print(json.dumps(record), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
