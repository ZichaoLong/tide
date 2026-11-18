"""Stateful prefill/decode, full-stream training and actual checkpoint updates.

No task head is included: first output means first available backbone output.
Every chunk advances real private state; Transformer uses its incremental cache.
"""
import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
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
                  boundary="state detached before checkpoint; one full next window + AdamW on each owner",
                  saved_data=nested_comparison(
                      dict(parameters=restored.state_dict(), state=saved, optimizer=resumed_optimizer.state_dict()),
                      dict(parameters=model.state_dict(), state=state, optimizer=optimizer.state_dict()), atol=0, rtol=0))
    with torch.no_grad(), precision_context(next_input.device, precision):
        a, sa, ra = forward_state(restored, next_input, saved)
        b, sb, rb = forward_state(model, next_input, state)
        record["next_forward"] = nested_comparison(a, b, atol=0, rtol=0)
        record["next_state"] = nested_comparison(sa, sb, atol=0, rtol=0)
    gradients, inputs, losses, results = [], [], [], []
    for current, step_optimizer, current_state in ((model, optimizer, state),
                                                  (restored, resumed_optimizer, saved)):
        step_optimizer.zero_grad(set_to_none=True)
        x = next_input.detach().clone().requires_grad_(True)
        with precision_context(x.device, precision):
            value, following, _ = forward_state(current, x, current_state)
            loss = value.float().square().mean()
        loss.backward()
        gradients.append({name: None if p.grad is None else p.grad.detach().clone()
                          for name, p in current.named_parameters()})
        inputs.append(x.grad)
        results.append(detach_state(following))
        losses.append(float(loss.detach()))
        step_optimizer.step()
    synchronize(next_input.device)
    record.update(losses=losses, input_gradients=nested_comparison(inputs[1], inputs[0], rtol=1e-2),
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
    if optimizer is not None:
        state = detach_state(state)
    return joined, state, phases


def _persist(path, record):
    from scripts.job import atomic
    atomic(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", choices=("tide", "transformer"), default="tide")
    parser.add_argument("--transformer-width", type=int)
    parser.add_argument("--transformer-layers", type=int)
    parser.add_argument("--ffn-width", type=int)
    parser.add_argument("--head-dim", type=int, default=32)
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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.prefill, args.prefill_chunk, args.updates, args.threads) < 1 or min(args.decode, args.warmup) < 0:
        parser.error("counts must be positive; decode/warmup can be zero")
    if args.output.exists() or args.checkpoint is not None and args.checkpoint.exists():
        parser.error("choose new output/checkpoint identities")
    if args.checkpoint is not None and (args.subject != "tide" or args.mode != "train"):
        parser.error("checkpoint requires Tide train mode")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = ModelConfig.from_dict(json.loads(Path(args.config).read_text()))
    if args.subject == "tide":
        model = TideModel(config)
        width = config.input_dim or config.d
        spec = config.to_dict()
    else:
        if args.transformer_width is None or args.transformer_layers is None:
            parser.error("Transformer requires explicit matched --transformer-width/--transformer-layers")
        model = TransformerBaseline(args.transformer_width, args.transformer_layers,
                                    ffn_width=args.ffn_width, head_dim=args.head_dim)
        width, spec = args.transformer_width, model.configuration()
    generator = torch.Generator(device="cpu").manual_seed(args.seed+1)
    x_cpu = torch.randn(args.batch, args.prefill+args.decode, width, generator=generator)
    identity = hashlib.sha256()
    for name, tensor in model.named_parameters():
        identity.update(name.encode()); identity.update(tensor.detach().numpy().tobytes())
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(source=source, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  spec=spec, parameters=parameter_count(model), precision=precision_policy(args.device, args.precision),
                  initial_parameters_sha256=identity.hexdigest(), input_sha256=hashlib.sha256(x_cpu.numpy().tobytes()).hexdigest(),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), torch=str(torch.__version__),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), host_load_before=os.getloadavg(),
                  timing="input transfer/construction/warmup/checkpoint excluded; full stream train includes concatenation/loss/backward/AdamW/sync",
                  training_boundary="full BPTT within prefill+decode+flush; detach after each optimizer update",
                  initial_state="zero; subsequent measured streams continue previous private state",
                  first_output="first available backbone output, no task head or language-model TTFT",
                  observations=[], state="running")
    _persist(args.output, record)
    model = model.to(args.device)
    model.train(args.mode != "inference")
    x = x_cpu.to(args.device)
    # Warmup is a separate inference owner state and does not update parameters.
    for _ in range(args.warmup):
        run_stream(model, x, args.prefill, args.prefill_chunk, precision=args.precision)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True) if args.mode == "train" else None
    state = None
    failed = False
    try:
        for step in range(args.updates):
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
            if not (row["finite_output"] and row["finite_parameters"] and row["finite_gradients"]):
                raise ArithmeticError("nonfinite continuous stream")
            record["observations"].append(row)
            _persist(args.output, record)
            del value
        if args.checkpoint is not None:
            record["checkpoint"] = checkpoint_resume(model, optimizer, state, x[:, :min(args.prefill_chunk, x.shape[1])],
                                                      args.precision, args.checkpoint)
            failed |= record["checkpoint"]["state"] in {"nonfinite", "structural_failed"}
        record["state"] = "failed" if failed else "completed"
    except Exception as error:
        import traceback
        traceback.print_exc()
        record.update(state="execution_failed", error=repr(error))
        failed = True
    _persist(args.output, record)
    print(json.dumps(record), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
