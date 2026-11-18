"""Finite FP32/BF16 qualification with frozen CPU fixtures and explicit failures.

Run CPU with --save-tensors first. A separate NPU process takes --reference;
it never uses a CPU autograd trajectory as its execution schedule. Raw tensor
fixtures are diagnostic artifacts, while --output is the small durable record.
"""
import argparse
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import torch

from tide import ModelConfig, TideModel, detach_state
from tide.attention import weighted_attention
from tide.precision import ArithmeticAudit, precision_context, precision_policy
from scripts.job import atomic


CONFIGS = ["fc-k", "fc-s", "hb-lattice-k", "hb-exchange-x",
           "hb-multiscale-x-settle", "hb-multiscale-x-timed", "fc-k-149m"]
THRESHOLDS = {
    "fp32": {"atol": 1e-6, "rtol": 2e-3},
    "bf16": {"atol": 2e-3, "rtol": 5e-2},
    "fp64": {"atol": 1e-10, "rtol": 1e-8},
}


def tensors(value, path="root"):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(getattr(value, field.name), path + "." + field.name)
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from tensors(value[key], path + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from tensors(item, path + f".{i}")


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value):
        return type(value)(**{f.name: cpu_tree(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(cpu_tree(v) for v in value))
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def comparison(actual, reference, tolerance):
    a, b = dict(tensors(actual)), dict(tensors(reference))
    metadata = sorted(a.keys() ^ b.keys())
    work_differences = []
    def structure(x, y, path="root"):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return
        if type(x) is not type(y):
            metadata.append(path + ".type")
        elif is_dataclass(x):
            for field in fields(x):
                structure(getattr(x, field.name), getattr(y, field.name), path + "." + field.name)
        elif isinstance(x, dict):
            if x.keys() != y.keys():
                metadata.append(path + ".keys")
            for key in x.keys() & y.keys():
                structure(x[key], y[key], path + "." + str(key))
        elif isinstance(x, (tuple, list)):
            if len(x) != len(y):
                metadata.append(path + ".length")
            for i, (xx, yy) in enumerate(zip(x, y)):
                structure(xx, yy, path + f".{i}")
        elif x != y:
            if ".work." in path:
                work_differences.append({"field": path, "actual": x, "reference": y})
            else:
                metadata.append(path)
    structure(actual, reference)
    failures, nonfinite, support, dtype, exact_failures, differences = [], [], [], [], [], []
    for name in sorted(a.keys() & b.keys()):
        x, y = a[name], b[name]
        if x.shape != y.shape or x.is_floating_point() != y.is_floating_point():
            metadata.append(name)
            continue
        x, y = x.detach().cpu(), y.detach().cpu()
        if x.dtype != y.dtype:
            dtype.append({"tensor": name, "actual": str(x.dtype), "reference": str(y.dtype)})
        if not x.is_floating_point():
            if not torch.equal(x, y):
                metadata.append(name)
            continue
        x, y = x.double(), y.double()
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            nonfinite.append(name)
            continue
        if not torch.equal(x, y):
            exact_failures.append(name)
        # Parameter matrices called "weight" are not message-support tensors.
        support_tensor = (name.rsplit(".", 1)[-1] == "strengths" or
                          (".state." in name or ".final_state." in name) and
                          name.rsplit(".", 1)[-1] in {"weight", "short_weight", "long_weight"})
        if support_tensor:
            count = int(((x > 0) != (y > 0)).sum())
            if count:
                support.append({"tensor": name, "different_entries": count, "entries": x.numel()})
        if x.numel():
            delta = (x - y).abs()
            scaled = float((delta / (tolerance["atol"] + tolerance["rtol"] * y.abs())).max())
            row = {"tensor": name, "max_abs": float(delta.max()), "scaled_error": scaled,
                   "relative_l2": float((x-y).norm() / y.norm().clamp_min(1e-30))}
            if scaled > 1:
                failures.append(name)
            differences.append(row)
    return {"tolerance": tolerance, "metadata_failures": sorted(set(metadata)), "dtype_differences": dtype,
            "nonfinite": nonfinite, "support_differences": support, "threshold_failures": failures,
            "work_differences": work_differences,
            "exact": not (metadata or dtype or nonfinite or exact_failures),
            "largest_errors": sorted(differences, key=lambda r: r["scaled_error"], reverse=True)[:8]}


def fingerprint(value):
    result = hashlib.sha256()
    for name, item in tensors(value):
        result.update(name.encode())
        result.update(str(tuple(item.shape)).encode())
        result.update(str(item.dtype).encode())
        result.update(item.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def fields_of(result):
    return {"values": result.values, "strengths": result.strengths, "times": result.times,
            "readout": result.readout(), "state": result.state}


def _primitive_fixture(prior):
    """Independent scalar quotient/VJP; no CPU autograd in device processes."""
    q = (torch.arange(24).reshape(1, 2, 3, 4) % 7 - 3).double() / 4
    k = (torch.arange(32).reshape(1, 2, 4, 4) % 9 - 4).double() / 4
    v = (torch.arange(32).reshape(1, 2, 4, 4) % 5 - 2).double() / 2
    w = torch.tensor([prior], dtype=torch.float64)
    mask = torch.tensor([[True, True, False, True], [True, True, True, False], [True]*4])
    seed = (torch.arange(24).reshape_as(q) % 3 - 1).double()
    value, dq, dk, dv, dw = [torch.zeros_like(x) for x in (q, q, k, v, w)]
    # Deliberately independent small scalar enumeration: no tensor matmul,
    # softmax or autograd is used to construct the expected result/VJP.
    for h in range(2):
        for query in range(3):
            terms = [math.exp(sum(float(q[0, h, query, d]) * float(k[0, h, key, d])
                                 for d in range(4)) / 2) if mask[query, key] else 0. for key in range(4)]
            mass = sum(terms[key] * prior[key] for key in range(4))
            if mass == 0:
                continue  # Declared all-zero-row VJP.
            probabilities = [terms[key] * prior[key] / mass for key in range(4)]
            for d in range(4):
                value[0, h, query, d] = sum(probabilities[key] * float(v[0, h, key, d]) for key in range(4))
            for key in range(4):
                centered = sum(float(seed[0, h, query, d]) *
                               (float(v[0, h, key, d]) - float(value[0, h, query, d])) for d in range(4))
                dl = probabilities[key] * centered
                dw[0, key] += terms[key] / mass * centered
                for d in range(4):
                    dq[0, h, query, d] += dl * float(k[0, h, key, d]) / 2
                    dk[0, h, key, d] += dl * float(q[0, h, query, d]) / 2
                    dv[0, h, key, d] += probabilities[key] * float(seed[0, h, query, d])
    grads = (dq, dk, dv, dw)
    return cpu_tree({"inputs": [q, k, v, w], "mask": mask, "seed": seed,
                     "expected": {"value": value, "gradients": grads}})


def primitive_fixture():
    return {"format": "tide-precision-primitives-2", "oracle": "independent_scalar_quotient_and_analytic_vjp",
            "cases": {"positive_prior": _primitive_fixture([.25, 2**-12, .5, 1.]),
                      "zero_prior": _primitive_fixture([0., 2**-12, .5, 1.]),
                      "empty": _primitive_fixture([0., 0., 0., 0.])}}


def primitive_checks(fixture, device, precision):
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}[precision]
    rows = []
    cases = fixture["cases"] if "format" in fixture else {"legacy_zero_prior": fixture}
    for case, data in cases.items():
        for backend in ("math", "tiled", "auto"):
            q, k, v = [x.to(device=device, dtype=dtype).requires_grad_() for x in data["inputs"][:3]]
            wdtype = torch.float64 if precision == "fp64" else torch.float32
            w = data["inputs"][3].to(device=device, dtype=wdtype).requires_grad_()
            with ArithmeticAudit() as audit:
                out = weighted_attention(q, k, v, w, data["mask"].to(device), backend)
            seed = data["seed"].to(device=device, dtype=dtype)
            grad = torch.autograd.grad((out * seed).sum(), (q, k, v, w))
            rows.append({"case": case, "backend": backend, "arithmetic": audit.report(),
                         **comparison({"value": out, "gradients": grad}, data["expected"], THRESHOLDS[precision])})
    return rows


def atomic_tensor(path, payload):
    if path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp64"), default="fp32")
    parser.add_argument("--configs", nargs="+", default=CONFIGS)
    parser.add_argument("--width", type=int, default=64, help="finite diagnostic width; 0 preserves config width")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint", action="store_true", help="check exact saved data and an actual resumed AdamW update")
    parser.add_argument("--primitives-only", action="store_true", help="fixed independent scalar formula/VJP witnesses only")
    parser.add_argument("--reference", type=Path, help="directory of trusted CPU fixture .pt files")
    parser.add_argument("--save-tensors", type=Path, help="new directory for optional diagnostic fixtures")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; select a new record identity")
    if args.precision == "fp64" and not args.device.startswith("cpu"):
        parser.error("FP64 is a CPU diagnostic only")
    if args.precision == "fp64" and args.checkpoint:
        parser.error("checkpoint qualification uses FP32/BF16; FP64 is a finite arithmetic diagnostic")
    if min(args.batch, args.length, args.steps, args.threads) < 1 or args.width < 0:
        parser.error("batch, length, steps and threads must be positive; width must be nonnegative")
    if args.device.startswith("npu"):
        if args.reference is None and not args.primitives_only:
            parser.error("NPU qualification requires a separately generated CPU --reference")
        import torch_npu
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    torch.set_num_threads(args.threads)
    if args.save_tensors:
        args.save_tensors.mkdir(parents=True, exist_ok=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = {"source": source, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "torch": str(torch.__version__), "physical_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "policy": precision_policy(device, args.precision) if args.precision != "fp64" else {"name": "cpu_fp64_diagnostic"},
              "thresholds": THRESHOLDS, "old_deviations": "Original FP32 report remains unchanged; no relabeling.",
              "exit_policy": "nonfinite, execution and metadata errors fail; finite tolerance/support differences remain explicit",
              "rows": []}
    if device.type == "npu":
        record["torch_npu"] = str(torch_npu.__version__)
        record["allow_hf32"] = {"matmul": bool(torch.npu.matmul.allow_hf32), "conv": bool(torch.npu.conv.allow_hf32)}
    fixture = (torch.load(args.reference / "attention.pt", weights_only=False) if args.reference else primitive_fixture())
    if args.save_tensors:
        atomic_tensor(args.save_tensors / "attention.pt", fixture)
    atomic(args.output, record)
    try:
        record["independent_attention"] = primitive_checks(fixture, device, args.precision)
    except Exception as exc:
        record["independent_attention_error"] = repr(exc)
        atomic(args.output, record)
        raise
    record["primitive_fixture"] = fixture.get("format", "legacy_zero_prior_v1")
    primitive_failed = any(r["nonfinite"] or r["metadata_failures"] for r in record["independent_attention"])
    record["independent_attention_state"] = ("failed" if primitive_failed else
        "finite_deviation" if any(r["threshold_failures"] for r in record["independent_attention"]) else "within_threshold")
    atomic(args.output, record)
    if args.primitives_only:
        print(json.dumps({"independent_attention_state": record["independent_attention_state"]}), flush=True)
        return int(primitive_failed)
    context = (lambda: nullcontext()) if args.precision == "fp64" else (lambda: precision_context(device, args.precision))
    dtype = torch.float64 if args.precision == "fp64" else torch.float32
    for index, name in enumerate(args.configs):
        row = {"case": name, "state": "running"}
        record["rows"].append(row)
        atomic(args.output, record)
        try:
            path = Path(name) if name.endswith(".json") else Path("configs") / (name + ".json")
            cfg = json.loads(path.read_text())
            if args.width:
                cfg["d"] = args.width
            cfg = ModelConfig.from_dict(cfg)
            row["config"] = cfg.to_dict()
            # Subset/reordered diagnostic requests preserve each case's fixture.
            seed = 31415 + int(hashlib.sha256(path.stem.encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            row["seed"] = seed
            model = TideModel(cfg)
            x_cpu = torch.randn(args.batch, args.length, cfg.input_dim or cfg.d)
            row["initial_parameter_sha256"] = fingerprint(model.state_dict())
            row["input_sha256"] = fingerprint(x_cpu)
            model = model.to(device=device, dtype=dtype)
            x = x_cpu.to(device=device, dtype=dtype).requires_grad_()
            with context(), ArithmeticAudit() as audit:
                result = model(x, record_work=True)
                loss = result.readout().square().mean()
            row["arithmetic"] = audit.report()
            loss.backward()
            first = cpu_tree({"result": fields_of(result), "input_gradient": x.grad,
                              "gradients": {n: p.grad for n, p in model.named_parameters()}})
            row["none_gradients"] = [n for n, p in model.named_parameters() if p.grad is None]
            row["first_work"] = result.work
            with torch.no_grad(), context():
                infer = model(x.detach())
            row["inference_vs_grad_forward"] = comparison(fields_of(infer), first["result"], THRESHOLDS[args.precision])
            row["persistent_dtypes"] = sorted({str(v.dtype) for _, v in tensors(result.state)})
            del infer, result
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)
            steps, state = [], None
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                # A declared truncated-gradient boundary between actual updates.
                with context():
                    result = model(x.detach(), state, record_work=True)
                    loss = result.readout().square().mean()
                loss.backward()
                nonfinite_gradients = [n for n, p in model.named_parameters()
                                       if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
                optimizer.step()
                nonfinite_parameters = [n for n, p in model.named_parameters() if not bool(torch.isfinite(p).all())]
                steps.append(cpu_tree({"readout": result.readout(), "strengths": result.strengths,
                                       "loss": loss, "position": result.state.position, "work": result.work}))
                state = detach_state(result.state)
                if nonfinite_gradients or nonfinite_parameters:
                    raise ArithmeticError(f"nonfinite at step {step}: {nonfinite_gradients}, {nonfinite_parameters}")
            row["updates"] = [{"step": i + 1, "loss": float(s["loss"]), "position": s["position"], "work": s["work"]}
                              for i, s in enumerate(steps)]
            row["optimizer_dtypes"] = sorted({str(v.dtype) for _, v in tensors(optimizer.state_dict())})
            snapshot = {"identity": {"config": cfg.to_dict(), "initial_parameter_sha256": row["initial_parameter_sha256"],
                                     "input_sha256": row["input_sha256"]}, "first": first,
                        "updates": steps, "final_parameters": cpu_tree(model.state_dict()), "final_state": cpu_tree(state),
                        "none_gradients": row["none_gradients"]}
            if args.reference:
                reference = torch.load(args.reference / (path.stem + ".pt"), weights_only=False)
                if snapshot["identity"] != reference["identity"]:
                    raise ValueError("reference fixture config/input/initial parameters differ")
                row["reference_comparison"] = comparison(snapshot, reference, THRESHOLDS[args.precision])
                row["gradient_connectivity_difference"] = sorted(set(snapshot["none_gradients"]) ^ set(reference["none_gradients"]))
            if args.save_tensors:
                atomic_tensor(args.save_tensors / (path.stem + ".pt"), snapshot)
            if args.checkpoint:
                from benchmarks.continuous import checkpoint_resume
                with tempfile.TemporaryDirectory(prefix="tide-precision-checkpoint-") as temporary:
                    row["checkpoint_resume"] = checkpoint_resume(model, optimizer, state, x.detach(), args.precision,
                                                                  Path(temporary) / "resume.pt")
            if args.precision == "bf16" and not any("bfloat16" in value for op in row["arithmetic"] for value in op["inputs"]):
                raise RuntimeError("BF16 requested but no BF16 matrix operation observed")
            comps = [row["inference_vs_grad_forward"]] + ([row["reference_comparison"]] if args.reference else [])
            restore = row.get("checkpoint_resume", {}).get("state")
            row["state"] = ("nonfinite" if any(c["nonfinite"] for c in comps) or restore == "nonfinite" else
                            "metadata_failed" if any(c["metadata_failures"] for c in comps) or
                            row.get("gradient_connectivity_difference") or restore == "structural_failed" else
                            "finite_deviation" if any(c["threshold_failures"] or c["support_differences"] for c in comps) or
                            restore == "finite_deviation" else
                            "within_threshold")
            del snapshot, first, model, optimizer, state, result, loss, x
            if device.type == "npu":
                torch.npu.empty_cache()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            row.update(state="execution_failed", error=repr(exc))
        atomic(args.output, record)
        print(json.dumps({"case": name, "state": row["state"]}), flush=True)
    primitive_failed = any(r["nonfinite"] or r["metadata_failures"] for r in record["independent_attention"])
    return int(primitive_failed or any(r["state"] in {"nonfinite", "metadata_failed", "execution_failed"}
                                      for r in record["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
