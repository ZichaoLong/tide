"""Finite SSD matrix-scan candidates on one frozen resident FP32 fixture.

This is a primitive measurement, not model training or token throughput. It
includes all five root VJPs in backward mode; no optimizer is present. B/C use
the instance's one group expanded over heads. A separate intrusive call records
operators/dtypes, and a bounded CPU FP64 literal recurrence checks every root.
The launcher owns hardware admission, affinity, time and memory limits.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from benchmarks.run import _device_memory, synchronize
from benchmarks.workloads import tensor_hash
from scripts.job import atomic
from tide.precision import ArithmeticAudit
import tide.ssd as ssd_implementation
from tide.ssd import ssd_scan


ROOT_NAMES = ("decay", "write_value", "B_group", "C_group", "initial_matrix")


class OperatorCounts(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


def fixture(args):
    generator = torch.Generator().manual_seed(args.seed)
    shape = (args.nodes, args.batch, args.length, args.heads)
    decay = .8 + .2 * torch.rand(shape, generator=generator)
    values = .1 * torch.randn(*shape, args.value_width, generator=generator)
    # Include exact forgetting and exact no-write rows without dividing by a.
    decay[..., 2::17, :] = 0
    decay[..., 11::19, :] = 1
    values[..., 11::19, :, :] = 0
    group_shape = (*shape[:-1], 1, args.state_width)
    B, C = [.1 * torch.randn(group_shape, generator=generator) for _ in range(2)]
    initial = .1 * torch.randn(args.nodes, args.batch, args.heads, args.value_width,
                               args.state_width, generator=generator)
    roots = (decay, values, B, C, initial)
    probes = tuple(torch.randn(x.shape, generator=generator) for x in (values, initial))
    return roots, probes


def expanded(roots):
    decay, values, B, C, initial = roots
    shape = (*decay.shape, B.shape[-1])
    return decay, values, B.expand(shape), C.expand(shape), initial


def execute(roots, probes, backend, mode, chunk):
    with torch.set_grad_enabled(mode != "inference"), torch.autocast(roots[0].device.type, enabled=False):
        outputs = ssd_scan(*expanded(roots), chunk_size=chunk, backend=backend)
        gradients = torch.autograd.grad(outputs, roots, probes) if mode == "backward" else ()
    return outputs, gradients


def literal(roots):
    """Chronological FP64 formula, independent of scan/prefix/block helpers."""
    decay, values, B, C, state = roots
    rows = []
    for t in range(decay.shape[-2]):
        state = (decay[..., t, :, None, None] * state
                 + torch.einsum("...hv,...hk->...hvk", values[..., t, :, :], B[..., t, :, :]))
        rows.append(torch.einsum("...hvk,...hk->...hv", state, C[..., t, :, :]))
    return torch.stack(rows, -3), state


def bounded(roots, probes):
    a, v, b, c, state = roots
    limited = (a[:1, :1, :65, :2], v[:1, :1, :65, :2, :4],
               b[:1, :1, :65, :, :4], c[:1, :1, :65, :, :4], state[:1, :1, :2, :4, :4])
    cotangents = (probes[0][:1, :1, :65, :2, :4], probes[1][:1, :1, :2, :4, :4])
    return limited, cotangents


def compare(actual, expected, atol, rtol):
    rows = []
    names = ("reads", "final_matrix", *ROOT_NAMES)
    for name, got, want in zip(names, (*actual[0], *actual[1]), (*expected[0], *expected[1])):
        x, y = got.detach().cpu().double(), want.detach().cpu().double()
        delta = (x - y).abs()
        finite = bool(torch.isfinite(x).all() & torch.isfinite(y).all())
        rows.append(dict(name=name, shape=list(x.shape), finite=finite,
                         exact=torch.equal(x, y), max_absolute=float(delta.max()),
                         relative_l2=float(torch.linalg.vector_norm(x-y) /
                                           torch.linalg.vector_norm(y).clamp_min(1e-300)),
                         within_threshold=finite and bool((delta <= atol + rtol*y.abs()).all())))
    return dict(atol=atol, rtol=rtol, all_finite=all(r["finite"] for r in rows),
                within_threshold=all(r["within_threshold"] for r in rows), tensors=rows)


def source_identity():
    path = Path(ssd_implementation.__file__).resolve()
    return dict(commit=os.environ.get("TIDE_SOURCE_COMMIT") or
                subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                working_tree=subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines(),
                implementation_path=str(path), implementation_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backends", nargs="+", choices=("block", "incremental", "doubling"),
                        default=["block", "incremental", "doubling"])
    parser.add_argument("--modes", nargs="+", choices=("inference", "grad_forward", "backward"),
                        default=["inference", "grad_forward", "backward"])
    for name, default in (("nodes", 8), ("batch", 8), ("length", 128), ("heads", 4),
                          ("value-width", 32), ("state-width", 32), ("chunk", 32),
                          ("threads", 8), ("warmup", 2), ("steps", 5), ("seed", 1911)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--atol", type=float, default=3e-5)
    parser.add_argument("--rtol", type=float, default=3e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new observation identity")
    if (min(args.nodes, args.batch, args.length, args.heads, args.value_width, args.state_width,
            args.chunk, args.threads, args.steps) < 1 or args.chunk > 32 or args.warmup < 0):
        parser.error("positive dimensions/counts, chunk <=32 and nonnegative warmup are required")
    if not all(0 <= x < float("inf") for x in (args.atol, args.rtol)):
        parser.error("comparison thresholds must be finite and nonnegative")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(args.threads)
    cpu_roots, cpu_probes = fixture(args)
    roots = tuple(x.to(device).detach().requires_grad_() for x in cpu_roots)
    probes = tuple(x.to(device) for x in cpu_probes)
    small, small_probes = bounded(cpu_roots, cpu_probes)
    fp64_roots = tuple(x.double().detach().requires_grad_() for x in small)
    expected = literal(fp64_roots)
    reference = (expected, torch.autograd.grad(expected, fp64_roots, tuple(x.double() for x in small_probes)))
    reference = tuple(tuple(x.detach() for x in group) for group in reference)
    small_device = tuple(x.contiguous().to(device).detach().requires_grad_() for x in small)
    small_cotangents = tuple(x.contiguous().to(device) for x in small_probes)
    record = dict(schema=1, completed=False, source=source_identity(),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  torch=str(torch.__version__), dtype="float32; autocast disabled",
                  python_executable=sys.executable,
                  device_name=(getattr(torch, device.type).get_device_name(device)
                               if device.type in {"npu", "cuda"} else "cpu"),
                  torch_npu=str(torch_npu.__version__) if device.type == "npu" else None,
                  reduced_precision_matmul="NPU HF32 and CUDA TF32 disabled when applicable",
                  cpu_affinity=sorted(os.sched_getaffinity(0)), host_load_before=os.getloadavg(),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified; admission belongs to launcher"),
                  fixture=[dict(name=name, shape=list(x.shape), sha256=tensor_hash(x))
                           for name, x in zip(ROOT_NAMES, cpu_roots)],
                  cotangent_sha256=[tensor_hash(x) for x in cpu_probes],
                  scan_argument_strides=[list(x.stride()) for x in expanded(roots)],
                  resident_input_and_cotangent_bytes=sum(x.numel()*x.element_size() for x in (*roots, *probes)),
                  analytical_storage_components=dict(
                      one_pairwise_tile_bytes=4*args.nodes*args.batch*args.heads*min(args.chunk, args.length)**2,
                      one_boundary_state_bytes=roots[-1].numel()*roots[-1].element_size(),
                      max_doubling_stages=(min(args.chunk, args.length)-1).bit_length(),
                      blocks=(args.length+args.chunk-1)//args.chunk,
                      interpretation="components, not total peak; doubling backward can retain a tile per stage per block; L=1 incremental/doubling omits pairwise tiles"),
                  bounded_reference_shapes=[list(x.shape) for x in small],
                  boundary="resident frozen inputs and cotangents; scan plus synchronization; backward includes all five root VJPs, no optimizer; destruction of returned graph/results included",
                  operator_observation="separate intrusive forward and VJP; dispatch operators/dtypes do not certify AICore/AICPU kernel placement",
                  candidates=[], rows=[])
    atomic(args.output, record)
    # Keep the baseline diagnostic off device while timing each candidate.
    baseline = execute(roots, probes, "block", "backward", args.chunk)
    baseline = tuple(tuple(x.detach().cpu() for x in group) for group in baseline)
    for backend in args.backends:
        observed = execute(roots, probes, backend, "backward", args.chunk)
        large_report = compare(observed, baseline, args.atol, args.rtol)
        del observed
        observed_small = execute(small_device, small_cotangents, backend, "backward", args.chunk)
        small_report = compare(observed_small, reference, args.atol, args.rtol)
        del observed_small
        with OperatorCounts() as counts, ArithmeticAudit() as arithmetic:
            execute(roots, probes, backend, "backward", args.chunk)
        synchronize(device)
        record["candidates"].append(dict(backend=backend, same_fixture_vs_block=large_report,
                                         bounded_vs_literal_fp64=small_report,
                                         operators=dict(sorted(counts.counts.items())), arithmetic=arithmetic.report()))
        for mode in args.modes:
            gc.collect()
            if device.type in {"npu", "cuda"}:
                getattr(torch, device.type).empty_cache()
            for _ in range(args.warmup):
                execute(roots, probes, backend, mode, args.chunk)
                synchronize(device)
            synchronize(device)
            memory_before = _device_memory(device, reset=True)
            samples = []
            for _ in range(args.steps):
                started = time.perf_counter()
                execute(roots, probes, backend, mode, args.chunk)
                synchronize(device)
                samples.append(time.perf_counter() - started)
            row = dict(backend=backend, mode=mode, seconds=samples, median_seconds=statistics.median(samples),
                       memory_before=memory_before, memory_after=_device_memory(device),
                       process_peak_rss_kib_cumulative=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            record["rows"].append(row)
            atomic(args.output, record)
            print(json.dumps(row), flush=True)
    record["completed"] = True
    record["within_threshold"] = all(c[key]["within_threshold"] for c in record["candidates"]
                                       for key in ("same_fixture_vs_block", "bounded_vs_literal_fp64"))
    record["host_load_after"] = os.getloadavg()
    atomic(args.output, record)
    return int(not record["within_threshold"])


if __name__ == "__main__":
    raise SystemExit(main())
