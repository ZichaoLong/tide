#!/usr/bin/env python3
"""Finite local additive comparison: shared matrix bank vs expanded snapshots.

Same independent K/V snapshot roots, q and weights. No graph scheduler, Q/out
projections, region allocation, optimizer or checkpoint work is included.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time
from durable_records import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("bank", "expanded"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mode", choices=("nograd", "grad-forward", "backward"), default="backward")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.length, args.width, args.heads, args.threads) < 1 or args.width % args.heads:
        parser.error("positive dimensions and width divisible by heads required")
    import torch
    from tidegraph.runtime import resolve_device
    from tidegraph.weighted.bank_attention import full
    from tidegraph.weighted.attention_mean import attention_mean
    from layer_tensor_baseline import numerics
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device, _ = resolve_device(args.device)
    policy = numerics(device)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    def sync():
        if device.type != "cpu":
            getattr(torch, device.type).synchronize(device)
    b, length, width, heads = args.batch, args.length, args.width, args.heads
    d, n = width//heads, b*length
    generator = torch.Generator().manual_seed(738)
    key = [row.clone() for row in torch.randn(b, length, width, generator=generator).to(device)]
    value = [row.clone() for row in torch.randn(b, length, width, generator=generator).to(device)]
    query = torch.randn(n, heads, d, generator=generator).to(device).requires_grad_()
    keys = [key[i][:j+1].detach().requires_grad_() for i in range(b) for j in range(length)]
    values = [value[i][:j+1].detach().requires_grad_() for i in range(b) for j in range(length)]
    weights = torch.tril(torch.ones(length, length, device=device)).repeat(b, 1).requires_grad_()
    def compute():
        if args.implementation == "bank":
            result = full(query, keys, values, weights)
            if result is None:
                raise RuntimeError("requested bank layout unexpectedly declined")
            return result
        def expand(rows):
            return torch.stack([torch.cat((row, row.new_zeros((length-len(row), width)))) for row in rows]).reshape(n, length, heads, d).transpose(1, 2)
        k, v = expand(keys), expand(values)
        logits = (query[:, :, None]*k).sum(-1)/d**.5
        positive = weights > 0
        safe = torch.where(positive, weights, 1)
        logits = (logits+safe.log()[:, None]).masked_fill(~positive[:, None], -torch.inf)
        return attention_mean(logits, v)
    report = dict(schema="tide-bank-attention-cost-v1", source=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), config=vars(args) | {"output_dir": str(out)},
        visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), numerical_policy=policy,
        scope="local weighted Full read after query projection; independent snapshots; no scheduler/region/optimizer",
        timing="one warmup then one measurement; local compute, declared roots VJP and sync; fixture construction excluded",
        strict_numerical="tested separately by test_bank_attention; not checked at measured scale", state="running")
    write_json(out/"result.json", report)
    try:
        for _ in range(2):
            sync(); begin = time.perf_counter()
            with torch.set_grad_enabled(args.mode != "nograd"):
                result = compute()
            sync(); forward = time.perf_counter()-begin
            backward = None
            if args.mode == "backward":
                begin = time.perf_counter()
                gradients = torch.autograd.grad(result, [query, weights, *keys, *values], torch.ones_like(result))
                sync(); backward = time.perf_counter()-begin
                if not all(torch.isfinite(t).all() for t in gradients):
                    raise ArithmeticError("nonfinite bank attention adjoint")
            if not torch.isfinite(result).all():
                raise ArithmeticError("nonfinite bank attention output")
            report["measurement"] = dict(forward_seconds=forward, backward_seconds=backward,
                                         total_seconds=forward+(backward or 0))
            del result
        report["state"] = "passed"
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out/"result.json", report)


if __name__ == "__main__":
    main()
