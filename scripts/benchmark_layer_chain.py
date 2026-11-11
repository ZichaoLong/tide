#!/usr/bin/env python3
"""Finite same-formula layer-chain cost anchor through the public default entry.

Run each implementation in a separate bounded service/process. The tensor
anchor has a narrower return/dependency domain; its gap is not equivalent
throughput or standard Transformer/Mamba parity. See docs/layer-chain-baseline.md.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from durable_records import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("kv", "ssd"), required=True)
    p.add_argument("--implementation", choices=("tensor", "python", "native"), required=True)
    p.add_argument("--interface", choices=("window", "records"), default="window")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--precision-policy", choices=("ieee", "vendor-default"), default="ieee")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--mode", choices=("inference", "grad-forward", "training"), default="training")
    p.add_argument("--native-library")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if min(a.batch, a.length, a.threads, a.width, a.depth, a.heads) < 1 or a.length > 256:
        p.error("positive dimensions/threads and L<=256 required")
    if a.width % a.heads:
        p.error("width must be divisible by heads")
    if a.implementation == "native" and not a.native_library:
        p.error("native requires an explicit matching library")
    import torch
    import tidegraph
    from tidegraph import Continuation, ExecutionPlacement
    from tidegraph.runtime import resolve_device
    from tidegraph.weighted.execution import eager_runner
    from tidegraph.experiment.finite import finite_tensors
    from layer_tensor_baseline import fixture, external, run, numerics
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    device, _ = resolve_device(a.device)
    numerical_policy = numerics(device, a.precision_policy)
    dtype = getattr(torch, a.dtype)
    def synchronize():
        if device.type != "cpu":
            getattr(torch, device.type).synchronize(device)
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema="tide-layer-chain-cost-v1", source=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), config=vars(a) | {"output_dir": str(out)},
        package=str(Path(tidegraph.__file__).resolve()), visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        numerical_policy=numerical_policy,
        affinity=sorted(os.sched_getaffinity(0)),
        scope="zero-left-state whole sequences; output content and terminal fields; tensor omits auxiliary Describe/graph records",
        strict_numerical="not_checked_at_measured_scale", recovery="outside_zero-left-state_anchor",
        timing="one reset warmup and one measured full update; input, transfer, scheduling, packing, loss, backward, finite checks, SGD and sync included; construction separate",
        execution="running", observations=[])
    write_json(out/"result.json", record)
    try:
        began = time.perf_counter()
        binding, spec = fixture(a.family, width=a.width, depth=a.depth, heads=a.heads, dtype=dtype, device=device)
        training = a.mode == "training"
        tracking = a.mode != "inference"
        parameters = [p for p in binding.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=1e-5, momentum=.3) if training else None
        initial = {n: t.detach().clone() for n, t in binding.model.state_dict().items()}
        record["parameters"] = sum(p.numel() for p in parameters)
        h = hashlib.sha256()
        for n, t in initial.items():
            h.update(n.encode()+b"\0"+t.cpu().numpy().tobytes())
        record["initial_parameters_sha256"] = h.hexdigest()
        runner = None if a.implementation == "tensor" else eager_runner(binding, a.implementation,
            ExecutionPlacement(preset="cpu" if device.type == "cpu" else "mixed-c", scoring_dtype=a.dtype),
            native_library=a.native_library, trace=False)
        if a.implementation == "native":
            record["native_build"] = json.loads((Path(a.native_library)/"build-manifest.json").read_text())
        synchronize()
        record["construction_seconds"] = time.perf_counter()-began
        for iteration in range(2):
            binding.model.load_state_dict(initial)
            if optimizer:
                optimizer.state.clear()
            synchronize()
            began = time.perf_counter()
            phases = {}
            def phase(name, fn):
                synchronize(); start = time.perf_counter()
                result = fn(); synchronize()
                phases[name] = time.perf_counter()-start
                return result
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            values = torch.randn(a.batch, a.length, a.width, dtype=dtype,
                                 generator=torch.Generator().manual_seed(17))*.1
            values = values.to(device).requires_grad_(tracking)
            with torch.set_grad_enabled(tracking):
                if runner is None:
                    y, state = phase("forward", lambda: run(binding, spec, values))
                    actual_work = dict(layers=a.depth, input_positions=a.batch*a.length,
                                       selected_events=a.batch*a.length*a.depth, implementation="tensor formulas")
                else:
                    def graph_forward():
                        if a.interface == "window":
                            return runner.run_positions(spec, values)
                        stop = a.length*spec.stride
                        return runner(Continuation(binding.graph.identity, a.batch), external(spec, values),
                                      stop, sealed_until=stop)
                    result = phase("forward", graph_forward)
                    # Canonical graph output order is time then sample; transpose
                    # only the returned content, without a CPU numeric replay.
                    if a.interface == "window":
                        y = result.readout(values)
                        record["effective_window_execution"] = dict(result.execution)
                    else:
                        y = torch.stack([payload[1:] for *_, payload in result.outputs]).reshape(
                            a.length, a.batch, a.width).transpose(0, 1)
                    state, actual_work = result.continuation, dict(result.stats)
                loss = y.square().mean()*.01
                phase("finite_forward", lambda: finite_tensors(dict(y=y, state=state, loss=loss), packing="packed"))
                if optimizer:
                    phase("backward", loss.backward)
                    phase("finite_gradients", lambda: finite_tensors([p.grad for p in parameters], packing="packed"))
                    phase("optimizer", optimizer.step)
                    phase("finite_parameters", lambda: finite_tensors(parameters, packing="packed"))
            row = dict(warmup=iteration == 0, phases=phases, loss=float(loss.detach()), actual_work=actual_work)
            del y, state, loss, values
            if runner is not None:
                del result
            synchronize()
            row["seconds"] = time.perf_counter()-began
            row["input_positions_per_second"] = a.batch*a.length/row["seconds"]
            if device.type != "cpu":
                api = getattr(torch, device.type)
                row["peak_allocated_bytes"] = api.max_memory_allocated(device)
                row["peak_reserved_bytes"] = api.max_memory_reserved(device)
            record["observations"].append(row)
            write_json(out/"result.json", record)
            print(json.dumps(row), flush=True)
        record["execution"] = "completed"
    except BaseException as error:
        record.update(execution="failed", error=repr(error))
        raise
    finally:
        write_json(out/"result.json", record)


if __name__ == "__main__":
    main()
