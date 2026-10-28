#!/usr/bin/env python3
"""Bounded stock-Torch peer VJP probe, independent of every graph/local kernel.

This isolates runtime event handling. Passing this smaller graph does not clear
the weighted graph's asynchronous failure. No production transfer is replaced.
"""
import argparse
import os
from pathlib import Path
import sys
import torch
from durable_records import write_json
from source_identity import source_state


def chain(x, a, b):
    terms = []
    for _ in range(12):
        x = x.to(a.device)*a + .1
        terms.append(x.square().sum())
        x = x.to(b.device)*b - .05
        terms.append(x.square().sum())
    return sum(t.to(b.device) for t in terms)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", required=True)
    p.add_argument("--peer-device", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/"python"))
    from tidegraph.runtime import resolve_device, manifest, synchronize
    device, reason = resolve_device(args.device)
    peer, _ = resolve_device(args.peer_device)
    if device == peer or device.type != peer.type or device.type == "cpu":
        p.error("two distinct explicit accelerators required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    source, dirty = source_state(root)
    report = dict(schema="tide-stock-peer-diagnostic-v1", source=source, dirty=dirty,
        state="running", runtime=manifest(device, reason, torch.float32), peer=str(peer),
        launch_blocking=os.environ.get("ASCEND_LAUNCH_BLOCKING"),
        scope="12 stock elementwise/copy rounds; scalar roots from both devices; no graph executor/custom Function",
        phases=[])

    def phase(name):
        report["phases"].append(name)
        write_json(out/"result.json", report)

    try:
        initial = (torch.linspace(-.4, .5, 16), torch.linspace(.2, .7, 16), torch.linspace(.6, .9, 16))
        left = tuple(x.clone().requires_grad_() for x in initial)
        right = tuple(x.to(d).requires_grad_() for x, d in zip(initial, (device, device, peer)))
        expected = chain(*left)
        phase("before_stock_forward")
        actual = chain(*right)
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
        phase("stock_forward_passed")
        gradients = torch.autograd.grad(actual, right)
        phase("stock_vjp_returned")
        reference = torch.autograd.grad(expected, left)
        for x, y in zip(gradients, reference):
            torch.testing.assert_close(x.cpu(), y, rtol=1e-5, atol=1e-6)
        for d in (device, peer):
            synchronize(d)
        report["state"] = "passed"
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out/"result.json", report)


if __name__ == "__main__":
    main()
