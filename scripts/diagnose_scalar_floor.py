#!/usr/bin/env python3
"""REQ-006 clamp-floor witness; no model monkeypatch or algorithm substitution.

Compare a Python float Scalar bound with a reusable same-dtype/device 0D Tensor.
The independent CPU formula is where(x >= bound, x, bound), preserving NaNs,
with input VJP where(x >= bound, bar, 0), including equality. The bound is constant, not a root.
Measure/profile are separate processes; this is an operator cost, not a model
update. See docs/evidence/req006-diagnostics-20261112.md for the observed trigger.
"""
import argparse
from contextlib import nullcontext
import hashlib
import math
import os
from pathlib import Path
import shutil
import sys
import time

from durable_records import write_json
from req006_material import tensor_digest
from source_identity import source_state


def bound_value(name, dtype):
    import torch
    return {"zero": 0., "tiny": torch.finfo(dtype).tiny,
            "log-tiny": math.log(torch.finfo(dtype).tiny)}[name]


def bound_argument(like, value, variant):
    if variant == "scalar":
        return float(value)
    if variant == "tensor":
        return like.new_tensor(value)  # Constant setup is outside operator timing.
    raise ValueError("unknown bound variant")


def represented_bound(value, dtype, representation):
    """Choose an explicit Scalar identity; Tensor bounds always use input dtype."""
    if representation == "requested":
        return value
    if representation == "dtype":
        import torch
        return torch.tensor(value, dtype=dtype).item()
    raise ValueError("unknown bound representation")


def boundary_material(dtype, value):
    """Representable neighbors, exact threshold, signed zero, tiny/subnormal."""
    import torch
    b = torch.tensor(value, dtype=dtype)
    tiny = torch.tensor(torch.finfo(dtype).tiny, dtype=dtype)
    subnormal = torch.nextafter(torch.zeros((), dtype=dtype), torch.ones((), dtype=dtype))
    x = torch.stack((torch.nextafter(b, b.new_tensor(-math.inf)), b,
                     torch.nextafter(b, b.new_tensor(math.inf)), b-1, b+1,
                     b.new_tensor(-0.), b.new_tensor(0.), -tiny, tiny,
                     -subnormal, subnormal, tiny/2, tiny*2))
    bar = torch.tensor([2., -3., 5., 7., -11., 13., 0., -17., 19., 23., -29., 31., -37.], dtype=dtype)
    return x, bar


def independent_formula(x, bar, value):
    import torch
    bound = x.new_tensor(value)
    mask = x >= bound
    y = torch.where(torch.isnan(x), x, torch.where(mask, x, bound))
    return y, torch.where(mask, bar, torch.zeros_like(bar))


def evaluate(x, bar, bound, mode, *, labelled=False):
    import torch
    def scope(name):
        return torch.profiler.record_function("req006::floor_"+name) if labelled else nullcontext()
    with torch.set_grad_enabled(mode == "backward"):
        with scope("forward"):
            y = torch.clamp_min(x, bound)
        if mode == "backward":
            with scope("backward"):
                (gradient,) = torch.autograd.grad(y, x, grad_outputs=bar)
        else:
            gradient = None
    return y, gradient


def compare(result, expected):
    import torch
    y, gradient = result
    ey, eg = expected
    y = y.detach().cpu()
    gradient = None if gradient is None else gradient.detach().cpu()
    finite = bool(torch.isfinite(y).all()) and (gradient is None or bool(torch.isfinite(gradient).all()))
    return dict(finite=finite, value_equal=torch.equal(y, ey),
                gradient_equal=None if gradient is None else torch.equal(gradient, eg),
                value_mismatches=int((y != ey).sum()),
                gradient_mismatches=None if gradient is None else int((gradient != eg).sum()),
                signed_zero_mismatches=int(((y == 0) & (ey == 0) & (torch.signbit(y) != torch.signbit(ey))).sum()))


def require_formula(checks, observe_finite=False):
    """A diagnostic may observe finite differences, never erase their failure."""
    if not all(c["finite"] for c in checks):
        raise FloatingPointError("nonfinite independent formula result")
    passed = all(c["value_equal"] and c["gradient_equal"] is not False for c in checks)
    if not passed and not observe_finite:
        raise AssertionError("independent formula mismatch; see recorded values/VJPs")
    return passed


def boundary_checks(device, dtype, value):
    """Small independent checks for both overloads, outside all measured regions."""
    import torch
    host_x, host_bar = boundary_material(dtype, value)
    expected = independent_formula(host_x, host_bar, value)
    result = {}
    for variant in ("scalar", "tensor"):
        x = host_x.to(device).requires_grad_()
        bar = host_bar.to(device)
        arg = bound_argument(x, value, variant)
        actual = evaluate(x, bar, arg, "backward")
        result[variant] = dict(compare(actual, expected), values=actual[0].detach().cpu().tolist(),
                               gradients=actual[1].detach().cpu().tolist())
    return dict(inputs=host_x.tolist(), cotangents=host_bar.tolist(), expected_values=expected[0].tolist(),
                expected_gradients=expected[1].tolist(), variants=result)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--bound", choices=("zero", "tiny", "log-tiny"), default="log-tiny")
    p.add_argument("--variant", choices=("scalar", "tensor"), default="scalar")
    p.add_argument("--bound-representation", choices=("requested", "dtype"), default="requested",
                   help="keep the requested Python float, or explicitly round it to input dtype before either overload")
    p.add_argument("--observe-finite-mismatch", action="store_true",
                   help="retain strict failure and exit 1, but collect finite operator timing/profile for diagnosis")
    p.add_argument("--shape", default="4,16,256,256", help="actual tensor shape, comma separated positive dimensions")
    p.add_argument("--mode", choices=("forward", "backward"), default="backward")
    p.add_argument("--action", choices=("check", "measure", "profile"), default="check")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--max-input-bytes", type=int, default=256*2**20)
    p.add_argument("--max-trace-bytes", type=int, default=512*2**20)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    try:
        a.shape = tuple(int(n) for n in a.shape.split(","))
    except ValueError:
        p.error("shape must contain positive integers")
    if not a.shape or min(*a.shape, a.threads, a.iterations, a.max_input_bytes, a.max_trace_bytes) < 1 or a.warmup < 0:
        p.error("positive shape/threads/iterations/budgets and nonnegative warmup required")
    if a.action == "profile" and a.iterations != 1:
        p.error("one profiled iteration per process")
    return a


def main():
    a = arguments()
    root = Path(__file__).resolve().parents[1]
    output = a.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    record = dict(schema="tide-scalar-floor-diagnostic-v1", source=source, dirty=dirty, state="running",
                  request={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  affinity=sorted(os.sched_getaffinity(0)), observations=[],
                  scope="Constant-bound forward or forward+input VJP only; no optimizer/model/throughput claim. "
                        "Input and cotangent creation/upload, reusable 0D bound creation, checks, output reads/hashes, "
                        "warmup and profiler export excluded. Per-step device synchronization included. "
                        "No gradient with respect to the bound, no double-backward qualification.")
    write_json(output/"result.json", record)
    try:
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        sys.path.insert(0, str(root/"python"))
        import torch
        from tidegraph.runtime import resolve_device, manifest
        torch.set_num_threads(a.threads)
        torch.set_num_interop_threads(1)
        device, reason = resolve_device(a.device)
        dtype = getattr(torch, a.dtype)
        requested = bound_value(a.bound, dtype)
        value = represented_bound(requested, dtype, a.bound_representation)
        record.update(runtime=manifest(device, reason, dtype), bound_requested=requested,
                      bound_argument_value=value,
                      bound_effective=torch.tensor(value, dtype=dtype).item(),
                      device_visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
        def sync():
            if device.type != "cpu":
                getattr(torch, device.type).synchronize(device)
        checks = boundary_checks(device, dtype, value)
        record["boundary_checks"] = checks
        relevant = checks["variants"].values() if a.action == "check" else (checks["variants"][a.variant],)
        passed = require_formula(tuple(relevant), a.observe_finite_mismatch)
        record["strict_formula_passed"] = passed
        if a.action != "check":
            count = math.prod(a.shape)
            if 2*count*torch.empty((), dtype=dtype).element_size() > a.max_input_bytes:
                raise ValueError("input+cotangent material exceeds declared byte bound; autograd workspace is additional")
            pattern, bars = boundary_material(dtype, value)
            host_x = pattern.repeat(math.ceil(count/len(pattern)))[:count].reshape(a.shape).contiguous()
            host_bar = bars.repeat(math.ceil(count/len(bars)))[:count].reshape(a.shape).contiguous()
            record["material_sha256"] = tensor_digest(dict(x=host_x, bar=host_bar))
            x = host_x.to(device).requires_grad_(a.mode == "backward")
            bar = host_bar.to(device)
            bound = bound_argument(x, value, a.variant)
            expected = independent_formula(host_x, host_bar, value)
            def step():
                return evaluate(x, bar, bound, a.mode, labelled=a.action == "profile")
            for _ in range(a.warmup):
                warm = step()
                del warm
            sync()
            if device.type != "cpu":
                getattr(torch, device.type).reset_peak_memory_stats(device)
            for _ in range(a.iterations):
                if a.action == "profile":
                    from req006_profile import profile_call, summarize_trace
                    if shutil.disk_usage(output).free < a.max_trace_bytes:
                        raise OSError("insufficient trace filesystem space")
                    os.chdir(output)  # Isolate CANN export scratch files.
                    trace = output/"trace.json"
                    result = profile_call(step, device, trace)
                    if trace.stat().st_size > a.max_trace_bytes:
                        raise RuntimeError("trace exceeds bound; raw trace retained")
                    record["trace"] = summarize_trace(trace)
                    record["trace"]["sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
                    if device.type != "cpu" and not record["trace"]["kernel_events"]:
                        raise RuntimeError("requested accelerator profile contains no kernels")
                    observation = dict(instrumented=True)
                else:
                    sync()
                    began = time.perf_counter()
                    result = step()
                    sync()
                    observation = dict(instrumented=False, seconds=time.perf_counter()-began)
                observation["formula_check"] = compare(result, expected)
                observation["output_sha256"] = tensor_digest(result)
                record["observations"].append(observation)
                passed = require_formula((observation["formula_check"],), a.observe_finite_mismatch) and passed
                record["strict_formula_passed"] = passed
                del result
            if device.type != "cpu":
                record["allocator_peak_bytes"] = getattr(torch, device.type).max_memory_allocated(device)
        record.update(execution="completed", state="passed" if passed else "failed",
                      failure_kind=None if passed else "finite_numeric")
    except BaseException as error:
        record.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(output/"result.json", record)
    return 0 if record["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
