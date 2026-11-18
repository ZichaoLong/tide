"""Finite, same-function Attention comparisons at the three instance sites.

These are post-projection primitives. ``forward_backward`` includes a scalar
cotangent contraction and leaf-gradient accumulation, with no optimizer or
model preparation; it is not complete training. Inputs/masks are resident.
Normal samples run without diagnostic wrappers. One separate audited call
records effective dispatch, tensor dtypes, operators, outputs and gradients.
"""
import argparse
from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time

import torch

from tide.attention import weighted_attention
from tide.precision import ArithmeticAudit
from scripts.job import atomic
from scripts.qualify_precision import fingerprint


# Deliberate shape/prior selections, rather than a topology/shape Cartesian grid.
CASES = {
    "kv-unit": ("kv", "unit", ("auto", "tiled")),
    "kv-zero": ("kv", "zero", ("auto", "math", "tiled")),
    "reception-positive": ("reception", "positive", ("auto", "math")),
    "region-zero": ("region", "zero", ("auto", "math", "tiled")),
    "reception-zero": ("reception", "zero", ("auto", "math", "tiled")),
    "region-positive": ("region", "positive", ("auto", "math")),
}
DEFAULT_CASES = ("kv-unit", "kv-zero", "reception-positive", "region-zero")


def make_fixture(case, *, batch=8, length=128, nodes=8, edges=8, d=128,
                 window=128, reception_width=64, seed=181118):
    """CPU FP32 source fixture, shared by every backend/dtype/mode."""
    site, prior, _ = CASES[case]
    rng = torch.Generator().manual_seed(seed + tuple(CASES).index(case))
    if site == "kv":
        old, keys = window - 1, window - 1 + length
        q_shape, k_shape = (nodes, batch, d//32, length, 32), (nodes, batch, d//32, keys, 32)
        weight_shape = (nodes, batch, keys)
        query_times = torch.arange(length, dtype=torch.int64) + old
        key_times = torch.arange(keys, dtype=torch.int64)
        delta = query_times[:, None] - key_times[None, :]
        mask = ((delta >= 0) & (delta < window))[None, None, None]
        meaning = "selected KV Full tile; prior window-1 old keys plus current-window writes; original-time causal window"
    elif site == "reception":
        old = 0
        q_shape = k_shape = (nodes, batch, length, reception_width//32, edges, 32)
        weight_shape, mask = (nodes, batch, length, edges), None
        meaning = "incoming message self-Attention per node/sample/logical position; all incoming edges are queries and keys"
    else:
        old = 0
        q_shape = k_shape = (batch, length, d//64, nodes, 32)
        weight_shape, mask = (batch, length, nodes), None
        meaning = "regional candidate self-Attention per sample/logical position; encoded width d/2"
    q = torch.randn(q_shape, generator=rng)
    k, v = [torch.randn(k_shape, generator=rng) for _ in range(2)]
    weights = None if prior == "unit" else .25 + .75*torch.rand(weight_shape, generator=rng)
    if prior == "zero":
        weights[..., ::4] = 0
        # Empty same-tick rows occur in Timed reception/regions. A selected KV
        # Full bank keeps positive visible history, so it has no invented empty bank.
        if site != "kv":
            weights.reshape(-1, weight_shape[-1])[::7] = 0
    cotangent = torch.randn(q_shape, generator=rng)
    return {"q": q, "k": k, "v": v, "weights": weights, "mask": mask, "cotangent": cotangent}, {
        "case": case, "site": site, "prior": prior, "meaning": meaning,
        "q_shape": q_shape, "k_shape": k_shape, "v_shape": k_shape,
        "prior_shape": None if weights is None else weight_shape,
        "mask_shape": None if mask is None else tuple(mask.shape), "old_history_keys": old,
        "head_width": 32, "scale": 32**-.5,
        "zero_prior_entries": 0 if weights is None else int((weights == 0).sum()),
        "prior_entries": 0 if weights is None else weights.numel(),
        "prior_gradient": "absent unit prior" if weights is None else
            "FP32 explicit key-prior derivative, including the primitive's right derivative at zero; caller no-write masking is outside this primitive",
    }


@contextmanager
def dispatch_observation(device):
    """Observe the actual adapter/custom-function call only outside timing."""
    module = importlib.import_module("tide.attention")
    original_native = module._native_attention
    original_tiled = module._TiledAttention.__dict__["forward"]
    observed = {"effective": "math", "native_api": None}
    def native(*args, **kwargs):
        observed.update(effective=args[5], native_api="torch SDPA" if args[5] == "sdpa" else "torch_npu.npu_fusion_attention_v2")
        return original_native(*args, **kwargs)
    def tiled(*args, **kwargs):
        observed["effective"] = "tiled"
        return original_tiled.__func__(*args, **kwargs)
    module._native_attention = native
    module._TiledAttention.forward = staticmethod(tiled)
    vendor, original_fusion = None, None
    if device.type == "npu":
        vendor = importlib.import_module("torch_npu")
        original_fusion = getattr(vendor, "npu_fusion_attention_v2", None)
        if original_fusion is not None:
            def fusion(q, k, v, *args, **kwargs):
                observed["native_qkv_shapes"] = [list(x.shape) for x in (q, k, v)]
                observed["native_qkv_dtypes"] = [str(x.dtype) for x in (q, k, v)]
                observed["native_pse_present"] = kwargs.get("pse") is not None
                observed["native_scale"] = kwargs.get("scale")
                return original_fusion(q, k, v, *args, **kwargs)
            vendor.npu_fusion_attention_v2 = fusion
    try:
        yield observed
    finally:
        module._native_attention = original_native
        module._TiledAttention.forward = original_tiled
        if original_fusion is not None:
            vendor.npu_fusion_attention_v2 = original_fusion


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(fixture, spec, *, device, precision, mode, backend, warmup=2, steps=5):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    backward = mode == "forward_backward"
    payload = {name: fixture[name].detach().to(device=device, dtype=dtype).requires_grad_(backward)
               for name in ("q", "k", "v")}
    weights = fixture["weights"]
    payload["weights"] = None if weights is None else weights.detach().to(device).requires_grad_(backward)
    mask = None if fixture["mask"] is None else fixture["mask"].to(device)
    seed = fixture["cotangent"].to(device) if backward else None
    roots = {name: value for name, value in payload.items() if value is not None}
    def clear():
        for value in roots.values():
            value.grad = None
    def compute():
        clear()
        with torch.set_grad_enabled(backward):
            out = weighted_attention(payload["q"], payload["k"], payload["v"], payload["weights"], mask, backend)
            if backward:
                loss = (out.float() * seed).mean()
                loss.backward()
        synchronize(device)
        return out
    for _ in range(warmup):
        compute()
    clear()
    synchronize(device)
    baseline_memory = peak_memory = peak_reserved = None
    if device.type in {"cuda", "npu"}:
        api = getattr(torch, device.type)
        baseline_memory = api.memory_allocated(device)
        api.reset_peak_memory_stats(device)
    samples = []
    for _ in range(steps):
        start = time.perf_counter()
        compute()  # Returned tensor is released before the next timed call.
        samples.append(time.perf_counter() - start)
    if device.type in {"cuda", "npu"}:
        api = getattr(torch, device.type)
        peak_memory, peak_reserved = api.max_memory_allocated(device), api.max_memory_reserved(device)
    # This observation is intrusive and has its own explicit boundary.
    with dispatch_observation(device) as dispatch, ArithmeticAudit() as audit:
        out = compute()
    gradients = {}
    for name, value in roots.items():
        gradient = value.grad
        gradients[name] = {"required": backward, "present": gradient is not None,
                           "dtype": None if gradient is None else str(gradient.dtype),
                           "finite": None if gradient is None else bool(torch.isfinite(gradient).all())}
    old = spec["old_history_keys"]
    if backward and old:
        gradients["retained_history_k"] = {"finite": bool(torch.isfinite(payload["k"].grad[..., :old, :]).all()),
                                           "nonzero_entries": int(torch.count_nonzero(payload["k"].grad[..., :old, :]))}
        gradients["retained_history_v"] = {"finite": bool(torch.isfinite(payload["v"].grad[..., :old, :]).all()),
                                           "nonzero_entries": int(torch.count_nonzero(payload["v"].grad[..., :old, :]))}
    finite = bool(torch.isfinite(out).all()) and all(row.get("finite") is not False for row in gradients.values())
    connected = all(not row.get("required") or row.get("present") for row in gradients.values())
    resident = sum(x.numel()*x.element_size() for x in (*roots.values(), mask, seed) if x is not None)
    seconds = statistics.median(samples)
    return {"state": "passed" if finite and connected else "nonfinite_or_disconnected",
            "requested_backend": backend, "precision": precision, "mode": mode,
            "payload_dtype": str(dtype), "prior_dtype": None if weights is None else "torch.float32",
            "seconds": samples, "median_seconds": seconds,
            "attention_query_head_rows_per_second": math.prod(out.shape[:-1])/seconds,
            "resident_input_bytes": resident, "key_value_bank_bytes": sum(payload[k].numel()*payload[k].element_size() for k in ("k", "v")),
            "device_allocated_before_samples": baseline_memory, "peak_device_allocated_bytes": peak_memory,
            "peak_device_reserved_bytes": peak_reserved,
            "process_peak_rss_kib_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "actual_payload_sha256": fingerprint(roots), "output_sha256": fingerprint(out),
            "audit": {"scope": "separate call after normal samples; forward and backward when requested",
                      "dispatch": dispatch, "operators": audit.report(), "output_dtype": str(out.dtype),
                      "output_finite": bool(torch.isfinite(out).all()), "gradients": gradients,
                      "hardware_accumulation_inferred": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(DEFAULT_CASES))
    parser.add_argument("--precisions", nargs="+", choices=("fp32", "bf16"), default=["fp32", "bf16"])
    parser.add_argument("--modes", nargs="+", choices=("inference", "forward_backward"), default=["inference", "forward_backward"])
    parser.add_argument("--backends", nargs="+", choices=("auto", "math", "tiled"), help="override each case's small default candidate list")
    for name, default in (("batch", 8), ("length", 128), ("nodes", 8), ("edges", 8), ("d", 128),
                          ("window", 128), ("reception-width", 64), ("threads", 4), ("steps", 5)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output identity")
    if min(args.batch, args.length, args.nodes, args.edges, args.d, args.window, args.reception_width, args.threads, args.steps) < 1 or args.warmup < 0:
        parser.error("counts must be positive and warmup nonnegative")
    if args.d % 64 or args.reception_width % 32:
        parser.error("d must be divisible by 64 and reception width by 32")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    torch.set_num_threads(args.threads)
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = {"source": source, "harness_source": os.environ.get("TIDE_HARNESS_COMMIT", source),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "torch": str(torch.__version__), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "threads": torch.get_num_threads(), "physical_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "resource_identity": os.environ.get("TIDE_RESOURCE_IDENTITY", "unmanaged observation; shared host"),
              "boundary": "resident post-projection primitive, prior validation/dispatch, output; backward adds clear-grad, FP32 cotangent mean, backward and synchronization; no optimizer",
              "precision_scope": "q/k/v and output use requested payload dtype; explicit priors FP32; math/tiled compute FP32; native actual inputs/operators audited",
              "memory_scope": "device peak reset per row after warmup; CPU RSS is a cumulative process high-water mark",
              "rows": []}
    if device.type == "npu":
        record["torch_npu"] = str(torch_npu.__version__)
        record["allow_hf32"] = {"matmul": bool(torch.npu.matmul.allow_hf32), "conv": bool(torch.npu.conv.allow_hf32)}
    for case in args.cases:
        fixture, spec = make_fixture(case, batch=args.batch, length=args.length, nodes=args.nodes, edges=args.edges,
                                     d=args.d, window=args.window, reception_width=args.reception_width)
        for precision in args.precisions:
            for mode in args.modes:
                for backend in args.backends or CASES[case][2]:
                    row = {"case": case, "shape": spec, "canonical_input_sha256": fingerprint(fixture),
                           "precision": precision, "mode": mode, "requested_backend": backend, "state": "running",
                           "host_load_before": os.getloadavg()}
                    record["rows"].append(row)
                    atomic(args.output, record)
                    try:
                        row.update(evaluate(fixture, spec, device=device, precision=precision, mode=mode, backend=backend,
                                            warmup=args.warmup, steps=args.steps))
                    except Exception as error:
                        import traceback
                        traceback.print_exc()
                        row.update(state="execution_failed", error=repr(error))
                    row["host_load_after"] = os.getloadavg()
                    atomic(args.output, record)
                    print(json.dumps({k: row[k] for k in ("case", "precision", "mode", "requested_backend", "state")}), flush=True)
    return int(any(row["state"] != "passed" for row in record["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
