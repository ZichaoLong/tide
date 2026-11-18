"""Bounded projection-fusion candidates; no replacement of production defaults.

Concatenating compatible base matrices removes projection launches. Private
per-head/per-branch adapters remain separate mathematical matrices and every
original Parameter stays owned by its original module. Concatenation and its
backward are real costs included in timings.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time
from types import MethodType

import torch

from tide.parameters import FeedForward, HeadQuery, NodeOwners, norm, owner_ids
from tide.precision import ArithmeticAudit, precision_context, precision_policy
from scripts.job import atomic


def fused_node_linears(layers, x, owners=None):
    """Concatenate compatible projections, preserving all private adapters.

    The head/branch axis is batched only for the private adapter's second
    contraction. No adapter is shared or collapsed into a dense base matrix.
    """
    first = layers[0]
    if any(layer.in_features != first.in_features or layer.out_features != first.out_features or
           layer._host_groups != first._host_groups or layer.shared != first.shared or
           (layer.bias is None) != (first.bias is None) for layer in layers):
        raise ValueError("fused node projections require compatible dimensions and ownership")
    if owners is None:
        owners = torch.arange(first.nodes, device=first.groups.device)
    ids, uniform = owner_ids(owners), first._uniform_group(owners)
    n, branches, width = len(ids), len(layers), first.out_features
    flat = x.reshape(n, -1, first.in_features)
    if uniform is None:
        groups = first.groups[ids]
        weights = torch.cat([layer.weight[groups] for layer in layers], 1)
        output = torch.bmm(flat, weights.transpose(1, 2))
    else:
        weights = torch.cat([layer.weight[uniform] for layer in layers], 0)
        output = torch.mm(flat.reshape(-1, first.in_features), weights.T)
        output = output.reshape(n, -1, branches * width)
    if first.shared:
        rank = first.right.shape[1]
        if any(layer.right.shape[1] != rank for layer in layers):
            raise ValueError("adapter ranks must agree")
        right = torch.cat([layer.right[ids] for layer in layers], 1)
        low = torch.bmm(flat, right.transpose(1, 2))
        low = low.reshape(n, -1, branches, rank).transpose(1, 2).reshape(n * branches, -1, rank)
        left = torch.stack([layer.left[ids] for layer in layers], 1).reshape(n * branches, width, rank)
        private = torch.bmm(low, left.transpose(1, 2))
        private = private.reshape(n, branches, -1, width).transpose(1, 2).reshape(n, -1, branches * width)
        output = output + private
    if first.bias is not None:
        bias = (torch.cat([layer.bias[groups] for layer in layers], 1)[:, None] if uniform is None else
                torch.cat([layer.bias[uniform] for layer in layers], 0))
        output = output + bias
    return output.reshape(*x.shape[:-1], branches * width)


def _head_forward(self, x, owners=None):
    return fused_node_linears(self.heads, x, owners)


def _ffn_forward(self, x, owners=None):
    gate, up = fused_node_linears((self.gate, self.up), norm(x), owners).chunk(2, -1)
    return x + self.down(torch.nn.functional.silu(gate) * up, owners)


@contextmanager
def projection_candidate(model, candidate="both"):
    """Temporarily select a candidate without changing state_dict/Parameters."""
    if candidate not in {"native", "head_query", "gate_up", "both"}:
        raise ValueError("unknown projection candidate")
    patched = []
    for module in model.modules():
        implementation = (_head_forward if isinstance(module, HeadQuery) and candidate in {"head_query", "both"} else
                          _ffn_forward if isinstance(module, FeedForward) and candidate in {"gate_up", "both"} else None)
        if implementation is not None:
            patched.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
            module.forward = MethodType(implementation, module)
    try:
        yield model
    finally:
        for module, had_instance_forward, previous in reversed(patched):
            if had_instance_forward:
                module.forward = previous
            else:
                del module.forward


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--kind", choices=("head_query", "ffn"), default="head_query")
    parser.add_argument("--sharing", choices=("independent", "shared"), default="shared")
    parser.add_argument("--mode", choices=("inference", "grad_forward", "train"), default="train")
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; select a new observation identity")
    if min(args.d, args.nodes, args.batch, args.length, args.threads, args.steps) < 1 or args.d % 32 or args.warmup < 0:
        parser.error("positive dimensions/counts, width divisible by 32, and nonnegative warmup are required")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(1811)
    groups = [0] * args.nodes if args.sharing == "shared" else None
    constructor = HeadQuery if args.kind == "head_query" else FeedForward
    original = constructor(args.nodes, args.d, groups=groups)
    x_cpu = torch.randn(args.nodes, args.batch, args.length, args.d)
    x = x_cpu.to(device)
    owners = NodeOwners(torch.arange(args.nodes, device=device), tuple(range(args.nodes)))
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = {"source": source, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "policy": precision_policy(device, args.precision), "parameters": sum(p.numel() for p in original.parameters()),
              "torch": str(torch.__version__), "cpu_affinity": sorted(os.sched_getaffinity(0)), "host_load_before": os.getloadavg(),
              "input_sha256": hashlib.sha256(x_cpu.numpy().tobytes()).hexdigest(),
              "physical_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "boundary": "resident input; projection concat/copy included; train includes loss/backward/AdamW/sync",
              "rows": []}
    for candidate in ("native", "both"):
        module = copy.deepcopy(original).to(device)
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4, foreach=True) if args.mode == "train" else None
        def step():
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(args.mode != "inference"), precision_context(device, args.precision):
                output = module(x, owners)
                loss = output.float().square().mean() if optimizer else None
            if optimizer:
                loss.backward()
                optimizer.step()
            synchronize(device)
        with projection_candidate(module, candidate):
            for _ in range(args.warmup):
                step()
            if device.type == "npu":
                torch.npu.reset_peak_memory_stats()
            elif device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for _ in range(args.steps):
                start = time.perf_counter()
                step()
                samples.append(time.perf_counter() - start)
            peak = (torch.npu.max_memory_allocated() if device.type == "npu" else
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
            with torch.no_grad(), precision_context(device, args.precision), ArithmeticAudit() as audit:
                output = module(x, owners)
            row = {"candidate": candidate, "seconds": samples, "median_seconds": statistics.median(samples),
                   "peak_device_bytes": peak, "process_peak_rss_kib_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "finite": bool(torch.isfinite(output).all()) and all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                                                                 for p in module.parameters()),
                   "arithmetic": audit.report()}
        record["rows"].append(row)
        atomic(args.output, record)
        print(json.dumps(row), flush=True)
        del module, optimizer
    return int(not all(r["finite"] for r in record["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
