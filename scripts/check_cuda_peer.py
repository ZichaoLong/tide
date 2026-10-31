#!/usr/bin/env python3
"""CUDA peer capability and bounded transfer/root probes, not model qualification.

Ordinary copies do not prove direct P2P. The explicit CPU-staged path is the
existing first-order diagnostic bridge, not a resident communication backend.
"""
import argparse
from contextlib import ExitStack
import ctypes
import json
from pathlib import Path
import sys
import torch
from durable_records import write_json
from source_identity import source_state


def capabilities(devices):
    # Reuse the runtime already loaded by the selected Torch, not a second SDK.
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if "/libcudart.so" in line}
    if len(paths) != 1:
        raise RuntimeError("expected one loaded CUDA runtime")
    path = paths.pop()
    runtime = ctypes.CDLL(path)
    runtime.cudaDeviceGetP2PAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                               ctypes.c_int, ctypes.c_int]
    result = []
    for source in devices:
        for target in devices:
            if source == target:
                continue
            row = dict(source=source, target=target,
                       torch_access=torch.cuda.can_device_access_peer(source, target))
            for attribute, name in ((2, "access_supported"), (3, "native_atomic_supported")):
                value = ctypes.c_int()
                code = runtime.cudaDeviceGetP2PAttribute(ctypes.byref(value), attribute, source, target)
                if code:
                    raise RuntimeError(f"P2P attribute query failed: {code}")
                row[name] = value.value
            if runtime.cudaSetDevice(source):
                raise RuntimeError("cannot select CUDA source")
            code = runtime.cudaDeviceEnablePeerAccess(target, 0)
            row["enable_peer_access_code"] = code
            runtime.cudaGetLastError()
            if code == 0 and runtime.cudaDeviceDisablePeerAccess(target):
                raise RuntimeError("cannot restore probe peer state")
            result.append(row)
    return dict(runtime_library=path, directions=result)


def probe(source, target, implementation, streams, core):
    from tidegraph.transfer import copy_rows
    from weighted_peer_bridge import _Copy
    src, dst = torch.device("cuda", source), torch.device("cuda", target)
    counts = dict(forward_calls=0, backward_calls=0, forward_bytes=0, backward_bytes=0)
    with ExitStack() as stack:
        if streams:
            stack.enter_context(torch.cuda.stream(streams[source]))
            stack.enter_context(torch.cuda.stream(streams[target]))
        initial = torch.arange(4096, dtype=torch.float32) / 4096
        a = initial.to(src).requires_grad_()
        b = (initial + .25).to(src).requires_grad_()
        if implementation == "python-ordinary":
            ys = copy_rows([a, b, a], dst)
        elif implementation == "native-ordinary":
            ys = core.copy_rows([a, b, a], str(dst), 8 * 1024**2)
        else:
            ys = [_Copy.apply(x, dst, x.dtype, counts) for x in (a, b, a)]
        for factor in (1., 0.):
            ga, gb = torch.autograd.grad((ys[0].sum() + ys[2].sum() * 2) * factor,
                                        (a, b), allow_unused=True, retain_graph=True)
            torch.cuda.synchronize(src)
            torch.cuda.synchronize(dst)
            assert gb is None and ga is not None
            torch.testing.assert_close(ga.cpu(), torch.full_like(initial, 3 * factor), atol=0, rtol=0)
        for actual, expected in zip(ys, (initial, initial + .25, initial)):
            torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    return dict(source=source, target=target, implementation=implementation,
                streams="nondefault" if streams else "default", elements=4096,
                independent_none_zero_alias_roots=True, exact=True, bridge_counts=counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", nargs="+", type=int, required=True, help="explicit logical CUDA indices")
    p.add_argument("--native-library")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "python"))
    from tidegraph.runtime import resolve_device, manifest
    from cuda_precision import ieee_policy
    if len(a.devices) < 2 or len(set(a.devices)) != len(a.devices):
        p.error("at least two distinct CUDA devices required")
    for index in a.devices:
        resolve_device(f"cuda:{index}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda", a.devices[0])
    policy = ieee_policy(device)
    core = None
    if a.native_library:
        from tidegraph.native_loader import load_native
        core = load_native(a.native_library, backend="cuda")
    source, dirty = source_state(root)
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(state="running", source=source, dirty=dirty, argv=sys.argv[1:],
        runtime=manifest(device, "explicit:cuda", torch.float32), cuda_precision=policy, cases=[],
        scope="capability queries and exact FP32 copy/independent VJP probes; no NCCL, model, resident or performance qualification")
    write_json(out / "result.json", report)
    try:
        report["capabilities"] = capabilities(a.devices)
        streams = {i: torch.cuda.Stream(device=i) for i in a.devices}
        implementations = ["python-ordinary", "python-cpu-staged"] + (["native-ordinary"] if core else [])
        for s in a.devices:
            for t in a.devices:
                if s == t:
                    continue
                for implementation in implementations:
                    for selected in (None, streams):
                        report["cases"].append(probe(s, t, implementation, selected, core))
                        write_json(out / "result.json", report)
        report["state"] = "passed"
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out / "result.json", report)
    print(json.dumps(dict(state=report["state"], cases=len(report["cases"]))))


if __name__ == "__main__":
    main()
