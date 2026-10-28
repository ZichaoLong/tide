#!/usr/bin/env python3
"""Run diagnose_weighted_workload with diagnostic FP64 outer adjoint joins.

Same CLI as diagnose_weighted_workload, restricted to --implementation python,
--dtype float32. This does not change public kernels, tolerances or optimizers.
Private saved VJP graphs still run through the normal engine. No speed claim.
"""
import json
from pathlib import Path
import sys
import torch
from durable_records import write_json
from weighted_join_probe import JoinProbe


def self_check(probe):
    x = torch.tensor(1., requires_grad=True)
    (1e8*x+x-1e8*x).backward()
    assert x.grad.item() == 1.  # Exact literal sum of three contributions.
    a = torch.tensor(.3, requires_grad=True)
    b = torch.tensor([.1, .2], requires_grad=True)
    y = ((a+b).sin()*a).sum()
    expected = torch.autograd.grad(y, (a, b), retain_graph=True)
    y.backward()
    for actual, reference in zip((a.grad, b.grad), expected):
        torch.testing.assert_close(actual, reference, atol=1e-7, rtol=1e-6)
    # Tensor and non-tensor arguments, independent None and connected zero.
    class Pair(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, scale, y):
            ctx.set_materialize_grads(False)
            ctx.scale = scale
            return x*scale, y*3
        @staticmethod
        def backward(ctx, gx, gy):
            return None if gx is None else gx*ctx.scale, None, None if gy is None else gy*3
    for coefficient in (1., 0.):
        x = torch.tensor(2., requires_grad=True)
        y = torch.tensor(3., requires_grad=True)
        first, second = Pair.apply(x, 2, y)
        (coefficient*first).backward()
        assert x.grad.item() == 2*coefficient and y.grad is None
    print(json.dumps(dict(state="passed", checks="literal cancellation, broadcast VJP, custom arguments, None/zero",
                          counts=dict(probe.counts))))


def main():
    probe = JoinProbe()
    if sys.argv[1:] == ["--self-check"]:
        original = torch.Tensor.backward
        with probe.install():
            self_check(probe)
        assert torch.Tensor.backward is original
        return
    # Parse the required boundaries before any tensors or output directories.
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--dtype");p.add_argument("--implementation");p.add_argument("--output-dir", type=Path)
    args, _ = p.parse_known_args()
    if "--help" not in sys.argv and (args.dtype != "float32" or args.implementation != "python" or args.output_dir is None):
        p.error("join diagnostic requires Python CPU FP32 and --output-dir")
    output = args.output_dir.resolve() if args.output_dir else None
    if output and output.exists():
        p.error("new diagnostic output directory required")
    with probe.install():
        import diagnose_weighted_workload
        try:
            diagnose_weighted_workload.main()
        finally:
            if output and output.is_dir():
                write_json(output/"join-probe.json", dict(torch=torch.__version__, counts=dict(probe.counts),
                    scope="diagnostic outer FP64 joins; local VJP and private subgraph dtype unchanged",
                    production=False, throughput=False))


if __name__ == "__main__":
    main()
