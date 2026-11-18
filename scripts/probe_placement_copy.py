"""Isolated first-order two-device copy/update probe; diagnostic, never timing.

Run each transport in a fresh admitted process. Stop after the first backend
error: later tensors from that process cannot diagnose an independent failure.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from torch import nn

from tide.placement import transfer_tensor
from tide.precision import precision_context
from scripts.job import atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs=2, default=["cpu", "cpu"])
    parser.add_argument("--transport", choices=("direct", "host_stage", "synchronized"), required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--phase-sync", action="store_true", help="diagnostic fences before backward/update")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.steps < 1:
        parser.error("use a new output and positive step count")
    if args.devices[0].startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.devices[0])
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(4); torch.manual_seed(1703)
    record = dict(source=os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        launch_blocking=os.environ.get("ASCEND_LAUNCH_BLOCKING"),
        interpretation="isolated operation/gradient/update diagnostic, not performance qualification",
        stages=[], observations=[], state="running")
    def stage(name):
        record["last_stage"] = name
        record["stages"].append(dict(stage=name, monotonic_seconds=time.monotonic()))
        atomic(args.output, record)
        print(json.dumps(dict(stage=name)), flush=True)
    def sync():
        for value in dict.fromkeys(args.devices):
            device = torch.device(value)
            if device.type in {"npu", "cuda"}:
                getattr(torch, device.type).synchronize(device)
    try:
        stage("construct")
        first = nn.Parameter((torch.randn(64, 64)/8).to(args.devices[0]))
        second = nn.Parameter((torch.randn(64, 64)/8).to(args.devices[1]))
        optimizer = torch.optim.AdamW([first, second], lr=1e-4, foreach=True)
        x_cpu = torch.randn(2, 5, 64)
        state = None
        for step in range(args.steps):
            stage(f"step.{step}.forward")
            optimizer.zero_grad(set_to_none=True)
            x = x_cpu.to(args.devices[0]).detach().requires_grad_()
            with precision_context(args.devices[0], args.precision):
                a = torch.tanh(x@first)
                b = transfer_tensor(a, args.devices[1], args.transport)
                b = b if state is None else b+state
                c = torch.sin(b@second)
                # The output stays with the destination just as an HB last layer.
                loss = c.float().square().mean()
            if args.phase_sync:
                sync()
            stage(f"step.{step}.backward")
            loss.backward()
            if args.phase_sync:
                sync()
            stage(f"step.{step}.optimizer")
            optimizer.step(); sync()
            state = c.detach()/8
            finite = all(bool(torch.isfinite(t).all()) for t in (c, first, second, first.grad, second.grad, x.grad))
            row = dict(step=step, loss=float(loss.detach()), finite=finite,
                       first_gradient_max=float(first.grad.abs().max()), second_gradient_max=float(second.grad.abs().max()))
            record["observations"].append(row)
            if not finite:
                raise ArithmeticError("nonfinite isolated copy/gradient/update")
            stage(f"step.{step}.complete")
        record["state"] = "completed"
    except Exception as error:
        import traceback
        traceback.print_exc()
        record.update(state="execution_failed", error=repr(error))
    atomic(args.output, record)
    return int(record["state"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
