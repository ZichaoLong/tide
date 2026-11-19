"""Manually invoked Attention qualification; never collected by pytest.

Example (device admission and environment activation are the caller's job):
  PYTHONPATH=src python tests/device_attention.py --device npu:0 \
      --output artifacts/attention-device.json
--device cpu validates the harness without importing TorchNPU or using hardware.
Reports distinguish measured numerical failures from runtime errors. They do
not certify a whole model, other shapes, or other software/device versions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import traceback


CASES = ("unit", "positive_prior", "frozen_prior", "masked_zero", "visible_zero")
SHAPES = {"short": (2, 2, 9, 13, 32), "long": (1, 2, 67, 259, 32)}


def _atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _reference(q, k, v, weights, mask):
    """Independent FP32 exponential quotient, with no implementation helpers."""
    import torch

    logits = (q @ k.transpose(-2, -1)) / q.shape[-1] ** .5
    peak = logits.masked_fill(~mask, -torch.inf).amax(-1, keepdim=True)
    peak = torch.where(torch.isfinite(peak), peak, 0)
    terms = torch.where(mask, torch.exp(logits - peak), 0)
    if weights is not None:
        terms = terms * weights[:, None, None, :]
    mass = terms.sum(-1, keepdim=True)
    normalized = terms / torch.where(mass > 0, mass, 1)
    output = normalized @ v
    return torch.where(mass > 0, output, 0)


def _graph_nodes(output):
    nodes, pending, seen = Counter(), [output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        nodes[type(node).__name__] += 1
        pending.extend(next_node for next_node, _ in node.next_functions)
    return dict(sorted(nodes.items()))


def _metrics(actual, reference, atol, rtol):
    import torch

    if actual is None:
        return {"state": "disconnected", "within_tolerance": False}
    actual, reference = actual.detach().cpu().double(), reference.detach().cpu().double()
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    difference = (actual - reference).abs()
    bound = atol + rtol * reference.abs()
    invalid = int((~finite).sum())
    failed = int((~finite | (difference > bound)).sum())
    return {
        "shape": list(reference.shape), "elements": reference.numel(),
        "nonfinite": invalid, "outside_tolerance": failed,
        "max_abs": float(difference[finite].max()) if bool(finite.any()) else None,
        "rms_abs": float(difference[finite].square().mean().sqrt()) if bool(finite.any()) else None,
        "max_tolerance_ratio": float((difference[finite] / bound[finite]).max()) if bool(finite.any()) else None,
        "within_tolerance": failed == 0,
    }


def _one_case(device, case, shape_name, atol, rtol):
    import torch
    from tide.attention import weighted_attention

    dtype = torch.float32
    batch, heads, queries, keys, width = SHAPES[shape_name]
    generator = torch.Generator().manual_seed(7083)
    shapes = ((batch, heads, queries, width), (batch, heads, keys, width), (batch, heads, keys, width))
    source = [torch.randn(shape, generator=generator) * .4 for shape in shapes]
    mask = torch.arange(keys)[None, :] <= (torch.arange(queries)[:, None] * keys // queries)
    mask = mask[None, None].expand(batch, 1, queries, keys).clone()
    mask[:, :, 0, :] = False  # A true empty row, including on native fusion.
    weights = None
    if case != "unit":
        weights = .25 + torch.rand(batch, keys, generator=generator)
        if case in {"masked_zero", "visible_zero"}:
            weights[:, 2::5] = 0
            if case == "masked_zero":
                mask[..., 2::5] = False
    weight_dtype = dtype
    trainable_prior = weights is not None and case != "frozen_prior"
    quantized = [value.to(dtype).float() for value in source]
    quantized_w = None if weights is None else weights.to(weight_dtype).float()
    cpu_roots = [value.detach().requires_grad_() for value in quantized]
    cpu_w = None if weights is None else quantized_w.detach().requires_grad_(trainable_prior)
    cpu_output = _reference(*cpu_roots, cpu_w, mask)
    seed = torch.randn(cpu_output.shape, generator=generator).to(dtype).float()
    reference_roots = cpu_roots + ([cpu_w] if trainable_prior else [])
    reference_gradients = torch.autograd.grad(cpu_output, reference_roots, seed)

    roots = [value.to(device=device, dtype=dtype).detach().requires_grad_() for value in quantized]
    prior = None if weights is None else quantized_w.to(device=device, dtype=weight_dtype).detach().requires_grad_(trainable_prior)
    backend = "auto" if case == "visible_zero" else "npu_fusion" if device.type == "npu" else "sdpa"
    output = weighted_attention(*roots, prior, mask.to(device), backend)
    nodes = _graph_nodes(output)
    actual_roots = roots + ([prior] if trainable_prior else [])
    actual_gradients = torch.autograd.grad(output, actual_roots, seed.to(device=device, dtype=dtype), allow_unused=True)
    if device.type == "npu":
        torch.npu.synchronize(device)
    fields = {"forward": _metrics(output, cpu_output, atol, rtol)}
    root_names = ("q", "k", "v", "weights") if trainable_prior else ("q", "k", "v")
    fields.update({f"vjp_{name}": _metrics(actual, expected, atol, rtol)
                   for name, actual, expected in zip(root_names, actual_gradients, reference_gradients)})
    state = "within_tolerance" if all(item["within_tolerance"] for item in fields.values()) else "numerical_failure"
    return {"state": state, "requested_backend": backend, "q_dtype": "float32",
            "weights_dtype": None if prior is None else str(prior.dtype),
            "input_policy": "CPU FP32 reference receives the exact quantized device inputs and VJP seed",
            "autograd_node_types": nodes, "fields": fields}


def run_qualification(device_name, cases=CASES, lengths=("short", "long"),
                      output_path=None, atol=3e-5, rtol=3e-4):
    """Callable entry; initializes the selected backend before any CPU VJP."""
    import torch

    family = str(device_name).split(":")[0]
    if family not in {"cpu", "npu"}:
        raise ValueError("this qualification compares CPU with CPU or NPU only")
    npu_version, device_label, hf32 = None, platform.machine(), None
    if family == "npu":
        import torch_npu  # Must precede CPU autograd to avoid late registration.
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
        hf32 = {"matmul": torch.npu.matmul.allow_hf32, "conv": torch.npu.conv.allow_hf32}
        npu_version = torch_npu.__version__
    device = torch.device(device_name)
    if device.type == "npu":
        torch.npu.set_device(device)
        device_label = torch.npu.get_device_name(device)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    source = root / "src/tide/attention.py"
    report = {"scope": "weighted_attention forward and first-order VJP; no model or performance claim",
              "run_state": "running", "started": datetime.now(timezone.utc).isoformat(),
              "device": str(device), "device_name": device_label, "torch": torch.__version__,
              "torch_npu": npu_version, "allow_hf32": hf32, "python": platform.python_version(),
              "source_head": os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "attention_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "results": []}
    for length in lengths:
        for case in cases:
            row = {"case": case, "shape_name": length, "shape_BHQKD": list(SHAPES[length]),
                   "dtype": "float32", "atol": atol, "rtol": rtol, "state": "running"}
            report["results"].append(row)
            if output_path is not None:
                _atomic_json(output_path, report)
            try:
                row.update(_one_case(device, case, length, atol, rtol))
            except Exception as error:
                row.update(state="runtime_error", error_type=type(error).__name__,
                           message=str(error), traceback=traceback.format_exc(limit=4))
            print(json.dumps({key: row[key] for key in ("case", "dtype", "shape_name", "state")}), flush=True)
            if output_path is not None:
                _atomic_json(output_path, report)
    report["run_state"] = "completed"
    report["finished"] = datetime.now(timezone.utc).isoformat()
    report["counts"] = dict(Counter(item["state"] for item in report["results"]))
    if output_path is not None:
        _atomic_json(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="cpu for harness check, or an explicitly admitted npu:N")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--lengths", nargs="+", choices=tuple(SHAPES), default=list(SHAPES))
    for name, default in (("atol", 3e-5), ("rtol", 3e-4)):
        parser.add_argument(f"--{name}", type=float, default=default)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; select a new evidence path")
    report = run_qualification(args.device, args.cases, args.lengths, args.output, args.atol, args.rtol)
    return int(any(row["state"] != "within_tolerance" for row in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
