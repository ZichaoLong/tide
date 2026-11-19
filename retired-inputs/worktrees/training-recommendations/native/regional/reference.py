"""Portable fixtures and the Python operator-layout reference for the native slice.

The C++ executable does not import or embed this module. A fixture is all FP32
little-endian tensors with an explicit shape and FNV-1a-64 corruption checksum;
the outer manifest additionally records SHA-256 for durable input identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

MAGIC = "tide-regional-tensors-v1 little-endian float32"
PARAMETERS = ("q_weight", "k_weight", "v_weight", "o_weight", "first_weight",
              "first_bias", "second_weight", "second_bias", "readout")


def atomic_json(path, value):
    path = Path(path)
    descriptor, stage = tempfile.mkstemp(dir=path.parent, prefix=".regional-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(stage, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(stage):
            os.unlink(stage)


def fnv1a(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def write_bundle(path, tensors):
    import numpy as np
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    rows, hashes = [MAGIC], {}
    for name, tensor in sorted(tensors.items()):
        if not name or any(not (c.isascii() and (c.isalnum() or c in "_.")) for c in name):
            raise ValueError("invalid tensor name")
        array = tensor.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite tensor: {name}")
        data = array.tobytes()
        file = path / f"{name}.f32"
        with file.open("wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        hashes[file.name] = hashlib.sha256(data).hexdigest()
        rows.append(" ".join([name, str(array.ndim), *map(str, array.shape), fnv1a(data)]))
    index = "\n".join(rows) + "\n"
    with (path / "index.txt").open("w") as stream:
        stream.write(index); stream.flush(); os.fsync(stream.fileno())
    hashes["index.txt"] = hashlib.sha256(index.encode()).hexdigest()
    atomic_json(path / "sha256.json", hashes)


def read_bundle(path):
    import numpy as np
    import torch
    path = Path(path)
    lines = (path / "index.txt").read_text().splitlines()
    if not lines or lines[0] != MAGIC:
        raise ValueError("invalid tensor bundle header")
    result = {}
    for line in lines[1:]:
        name, rank, *fields = line.split()
        rank = int(rank)
        if not 0 <= rank <= 8 or len(fields) != rank + 1 or name in result:
            raise ValueError("invalid tensor bundle row")
        if any(not (c.isascii() and (c.isalnum() or c in "_.")) for c in name):
            raise ValueError("invalid tensor name")
        shape = tuple(map(int, fields[:-1]))
        if any(d <= 0 for d in shape):
            raise ValueError("invalid tensor dimension")
        data = (path / f"{name}.f32").read_bytes()
        count = 1
        for dim in shape:
            count *= dim
        if len(data) != count * 4 or fnv1a(data) != fields[-1]:
            raise ValueError(f"tensor integrity failure: {name}")
        array = np.frombuffer(data, dtype="<f4").reshape(shape).copy()
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite tensor: {name}")
        result[name] = torch.from_numpy(array)
    return result


def fixture(path, *, width=32, nodes=5, batch=2, positions=3, tasks=4, seed=1729):
    import torch
    if width < 32 or width % 32 or min(nodes, batch, positions, tasks) < 1 or tasks > 64:
        raise ValueError("invalid fixture dimensions")
    generator = torch.Generator().manual_seed(seed)
    def sample(*shape, scale=1):
        return torch.randn(shape, generator=generator) * scale
    tensors = {name: sample(width, width, scale=width ** -.5) for name in PARAMETERS[:4]}
    tensors.update(first_weight=sample(2 * width, width, scale=width ** -.5),
                   first_bias=sample(2 * width, scale=.03),
                   second_weight=sample(width, 2 * width, scale=(2 * width) ** -.5),
                   second_bias=sample(width, scale=.03), readout=sample(width, scale=width ** -.5))
    shape = (tasks, nodes, batch, positions)
    tensors.update(z=sample(*shape, width, scale=.2), score=sample(*shape, scale=.1),
                   w=torch.rand(shape, generator=generator) * .8 + .1, target=sample(*shape))
    # Positive, visible-zero and empty rows all occur in the same shared owner.
    for task in range(tasks):
        if task % 4 == 1:
            tensors["w"][task, 0] = 0
        elif task % 4 == 2:
            tensors["w"][task] = 0
        elif task % 4 == 3:
            tensors["w"][task, ::2, :, ::2] = 0
    write_bundle(path, tensors)
    atomic_json(Path(path) / "fixture.json", dict(schema=1, seed=seed, width=width,
        nodes=nodes, batch=batch, positions=positions, tasks=tasks,
        prior_cases=["positive", "visible_zero", "empty", "mixed"]))
    return tensors


def region_type():
    import torch
    import torch.nn.functional as F
    from torch.autograd.function import once_differentiable

    class WeightedSoftmax(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, weights):
            positive = weights > 0
            weighted = torch.where(positive,
                logits + torch.where(positive, weights, torch.ones_like(weights)).log(), -torch.inf)
            nonempty = positive.any(-1, keepdim=True)
            safe = torch.where(nonempty, weighted, torch.zeros_like(weighted))
            log_z = safe.logsumexp(-1, keepdim=True)
            p = torch.where(nonempty, (safe - log_z).exp(), torch.zeros_like(safe))
            ctx.save_for_backward(logits, p, log_z, nonempty)
            return p

        @staticmethod
        @once_differentiable
        def backward(ctx, grad):
            logits, p, log_z, nonempty = ctx.saved_tensors
            centered = grad - (p * grad).sum(-1, keepdim=True)
            factor = torch.where(nonempty, logits - log_z, -torch.inf).exp()
            return p * centered, factor * centered

    class Region(torch.nn.Module):
        def __init__(self, tensors, device):
            super().__init__()
            self.heads = tensors["z"].shape[-1] // 32
            for name in PARAMETERS:
                self.register_parameter(name, torch.nn.Parameter(tensors[name].to(device).clone()))

        def forward(self, z, score, w, *, bf16=False, literal=False):
            device = z.device.type
            work_dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
            with torch.autocast(device, dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
                x = z.permute(1, 2, 0, 3)
                def project(weight):
                    value = F.linear(x, weight)
                    return value.reshape(*x.shape[:-1], self.heads, 32).transpose(-3, -2)
                q, k, v = [project(getattr(self, name)) for name in PARAMETERS[:3]]
                with torch.autocast(device, enabled=False):
                    logits = q.to(work_dtype) @ k.to(work_dtype).transpose(-2, -1) / 32 ** .5
                    weights = w.permute(1, 2, 0).unsqueeze(-2).unsqueeze(-2).expand_as(logits)
                    if literal:
                        # Independent quotient expression, for small FP64 inputs.
                        numerator = logits.exp() * weights
                        denominator = numerator.sum(-1, keepdim=True)
                        p = numerator / denominator.clamp_min(torch.finfo(work_dtype).tiny)
                        p = torch.where(denominator > 0, p, numerator * 0)
                    else:
                        p = WeightedSoftmax.apply(logits, weights)
                    read = (p @ v.to(work_dtype)).to(q.dtype)
                delta = F.linear(read.transpose(-3, -2).flatten(-2), self.o_weight)
                value = x + delta
                norm_work = value.to(work_dtype)
                norm = (norm_work * torch.rsqrt(norm_work.square().mean(-1, keepdim=True) + 1e-5)).to(value.dtype)
                hidden = F.silu(F.linear(norm, self.first_weight, self.first_bias))
                with torch.autocast(device, enabled=False):
                    correction = delta.to(work_dtype) @ self.readout
                    correction = correction + hidden.to(work_dtype) @ (self.second_weight.T @ self.readout)
                    correction = correction + self.second_bias @ self.readout
                return score + torch.where(w > 0, correction.permute(2, 0, 1), torch.zeros_like(score))

    return Region


def run_reference(tensors, *, device="cpu", dtype="float32", mode="train", repeats=1,
                  warmup=0, output=None):
    import torch
    if repeats < 1 or warmup < 0 or mode not in {"inference", "grad_forward", "train"}:
        raise ValueError("invalid mode or iteration bound")
    if dtype not in {"float32", "bfloat16"}:
        raise ValueError("unsupported dtype")
    if device == "auto":
        device = "cpu"
    if device.startswith("npu"):
        import torch_npu  # Only an explicitly requested Python NPU path registers it.
    device = torch.device(device)
    def sync():
        if device.type == "npu":
            torch.npu.synchronize(device)
        elif device.type == "cuda":
            torch.cuda.synchronize(device)
    def owner():
        model = region_type()(tensors, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, eps=.001, weight_decay=.01, foreach=False)
        inputs = []
        for task in range(tensors["z"].shape[0]):
            inputs.append({key: tensors[key][task].to(device).clone().requires_grad_(
                mode != "inference" and key != "target") for key in ("z", "w", "score", "target")})
        return model, optimizer, inputs
    def step(model, optimizer, inputs):
        if mode == "train":
            optimizer.zero_grad(set_to_none=True)
            for task in inputs:
                for key in ("z", "w", "score"):
                    task[key].grad = None
        outputs = []
        with torch.set_grad_enabled(mode != "inference"):
            for task in inputs:
                y = model(task["z"], task["score"], task["w"], bf16=dtype == "bfloat16")
                outputs.append(y)
                if mode == "train":
                    ((y * task["target"]).sum() / (y.numel() * len(inputs))).backward()
        if mode == "train":
            optimizer.step()
        return outputs
    if warmup:
        warm = owner()
        for _ in range(warmup):
            step(*warm)
        sync()
        del warm
    model, optimizer, inputs = owner()
    times = []
    for _ in range(repeats):
        sync(); start = time.perf_counter()
        outputs = step(model, optimizer, inputs)
        sync(); times.append((time.perf_counter() - start) * 1000)
    values = {f"output.{i}": item for i, item in enumerate(outputs)}
    for name, parameter in model.named_parameters():
        values[f"parameter.{name}"] = parameter
        if mode == "train":
            values[f"grad.parameter.{name}"] = parameter.grad
            values[f"moment1.{name}"] = optimizer.state[parameter]["exp_avg"]
            values[f"moment2.{name}"] = optimizer.state[parameter]["exp_avg_sq"]
    if mode == "train":
        for i, task in enumerate(inputs):
            for name in ("z", "w", "score"):
                values[f"grad.{i}.{name}"] = task[name].grad
    metadata = dict(schema=1, state="completed", client="Python matching operator layout",
        device=str(device), dtype=dtype, mode=mode, repeats=repeats, warmup=warmup,
        samples_ms=times, workers=1, intra_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(), torch_version=torch.__version__,
        torch_file=torch.__file__, cpu_affinity=sorted(os.sched_getaffinity(0)),
        sample_policy="fixed inputs; train parameters and AdamW continue; discarded warmup owner",
        timing_boundary="region tasks; train includes zero_grad, backward, AdamW; excludes fixture/model load and serialization")
    if output is not None:
        output = Path(output); output.mkdir(parents=True, exist_ok=False)
        write_bundle(output / "tensors", values)
        atomic_json(output / "result.json", metadata)
    return values, metadata


def compare(actual, expected, *, rtol, atol):
    import torch
    if actual.keys() != expected.keys():
        raise ValueError("tensor result keys differ")
    checks = []
    for key in sorted(actual):
        a, b = actual[key].detach().cpu(), expected[key].detach().cpu()
        same_shape = a.shape == b.shape
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        checks.append(dict(name=key, shape=list(a.shape), finite=finite,
            passed=same_shape and finite and torch.allclose(a, b, rtol=rtol, atol=atol),
            exact=same_shape and torch.equal(a, b),
            max_abs=float((a-b).abs().max()) if same_shape else None))
    return dict(state="passed" if all(row["passed"] for row in checks) else "failed",
                rtol=rtol, atol=atol, checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("fixture")
    create.add_argument("--output-dir", type=Path, required=True)
    for key, default in (("width",32),("nodes",5),("batch",2),("positions",3),("tasks",4),("seed",1729)):
        create.add_argument(f"--{key}", type=int, default=default)
    run = sub.add_parser("run")
    run.add_argument("--input-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--dtype", choices=("float32","bfloat16"), default="float32")
    run.add_argument("--mode", choices=("inference","grad_forward","train"), default="train")
    run.add_argument("--inner-threads", type=int, default=1)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--warmup", type=int, default=0)
    cmp = sub.add_parser("compare")
    cmp.add_argument("actual", type=Path); cmp.add_argument("expected", type=Path)
    cmp.add_argument("--rtol", type=float, default=2e-5); cmp.add_argument("--atol", type=float, default=2e-6)
    cmp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fixture":
        options = vars(args).copy(); options.pop("command"); path = options.pop("output_dir")
        fixture(path, **options)
    elif args.command == "run":
        import torch
        torch.set_num_threads(args.inner_threads); torch.set_num_interop_threads(1)
        run_reference(read_bundle(args.input_dir), device=args.device, dtype=args.dtype,
            mode=args.mode, repeats=args.repeats, warmup=args.warmup, output=args.output_dir)
    else:
        result = compare(read_bundle(args.actual), read_bundle(args.expected), rtol=args.rtol, atol=args.atol)
        atomic_json(args.output, result)
        raise SystemExit(0 if result["state"] == "passed" else 1)


if __name__ == "__main__":
    main()
