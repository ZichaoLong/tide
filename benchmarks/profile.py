"""Optional intrusive stage diagnosis: one warmup, then one observed step.

Run as python -m benchmarks.profile with the same config/device/batch/length/
threads arguments as benchmarks.run. Synchronization at every stage boundary
changes execution: these are diagnostic wall times, not normal throughput or
device kernel times. --cpu-operators adds a separate unwrapped CPU observation
containing operator counts/shapes and instrumented CPU times. Every step starts
from zero state; none of these CPU observations are NPU device kernel times.
"""
import argparse
import hashlib
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import torch
from tide import ModelConfig, TideModel


STAGES = ("Reception", "KV.prepare", "KV.finish", "SSD.prepare", "SSD.finish",
          "Description", "RegionNetwork", "region_control")


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def stage_observation(model, device, observations, layouts):
    """Temporarily wrap this model and its regional control call; always restore."""
    originals = []

    def wrap(owner, attribute, label):
        original = getattr(owner, attribute)

        def observed(*args, **kwargs):
            synchronize(device)
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                synchronize(device)
                observations[label]["seconds"] += time.perf_counter() - start
                observations[label]["calls"] += 1

        originals.append((owner, attribute, original, attribute in vars(owner)))
        setattr(owner, attribute, observed)

    def observe_packing(kind):
        module = importlib.import_module("tide.kv" if kind == "K" else "tide.ssd_memory")
        original = module.select_queries
        label = "KV" if kind == "K" else "SSD"

        def observed(*args, **kwargs):
            packed = original(*args, **kwargs)
            row = layouts.setdefault(label, dict(calls=0, selected_rows=0, padded_rows=0))
            row["calls"] += 1
            if packed is not None:
                row["selected_rows"] += packed.node.numel()
                row["padded_rows"] += packed.data.numel() // packed.data.shape[-1]
            return packed

        originals.append((module, "select_queries", original, True))
        module.select_queries = observed

    try:
        wrap(model.receive, "forward", "Reception")
        wrap(model.describe, "forward", "Description")
        for network in model.regional:
            wrap(network, "forward", "RegionNetwork")
        for kind, memory in model.memories.items():
            label = "KV" if kind == "K" else "SSD"
            wrap(memory, "prepare", label + ".prepare")
            wrap(memory, "finish", label + ".finish")
            observe_packing(kind)
        wrap(importlib.import_module("tide.model"), "region_control", "region_control")
        yield
    finally:
        for owner, attribute, original, owned in reversed(originals):
            if owned:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)


def step(model, inputs, optimizer, mode, device, stages=None, layouts=None):
    phases = {}

    def timed(name, operation):
        synchronize(device)
        start = time.perf_counter()
        result = operation()
        synchronize(device)
        phases[name] = time.perf_counter() - start
        return result

    synchronize(device)
    start = time.perf_counter()
    if optimizer is not None:
        timed("zero_grad", lambda: optimizer.zero_grad(set_to_none=True))
    with torch.set_grad_enabled(mode != "inference"):
        if stages is None:
            result = timed("forward", lambda: model(inputs))
        else:
            with stage_observation(model, device, stages, layouts):
                result = timed("forward", lambda: model(inputs))
        loss = timed("readout_and_loss", lambda: result.readout().float().square().mean())
    if optimizer is not None:
        timed("backward", loss.backward)
        timed("adamw", optimizer.step)
    synchronize(device)
    outer = time.perf_counter() - start
    if not bool(torch.isfinite(loss.detach())):
        raise ArithmeticError("non-finite diagnostic loss")
    return dict(outer_seconds=outer, phases_seconds=phases, loss=float(loss.detach()),
                outer_unattributed_seconds=outer - sum(phases.values()))


def persist(path, record):
    temporary = None
    payload = json.dumps(record, indent=2) + "\n"
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if path.read_text() != payload:
            raise OSError("diagnostic record verification failed")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--mode", choices=("inference", "grad_forward", "train"), default="train")
    parser.add_argument("--cpu-operators", action="store_true")
    parser.add_argument("--seed", type=int, default=1107)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.batch, args.length, args.threads) < 1:
        parser.error("batch, length and threads must be positive")
    if args.device.startswith("npu"):
        import torch_npu  # Registration only for an explicitly selected NPU.
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    if device.type=="npu":
        torch.npu.set_device(device)
    if args.cpu_operators and device.type != "cpu":
        parser.error("--cpu-operators is a separate CPU-only observation")
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a new observation path")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = ModelConfig.from_dict(json.loads(Path(args.config).read_text()))
    model = TideModel(config).to(device)
    model.train(args.mode != "inference")
    inputs_cpu = torch.randn(args.batch,args.length,config.input_dim or config.d,
                             generator=torch.Generator().manual_seed(args.seed+1))
    inputs = inputs_cpu.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True) if args.mode == "train" else None
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(state="running", source=source, harness_source=os.environ.get("TIDE_HARNESS_COMMIT",source),
                  arguments=vars(args), config=config.to_dict(),
                  torch=str(torch.__version__), precision="fp32", device=str(device),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  timing="intrusive synchronized wall time; prefill from zero through returned window",
                  device_kernel_times=False, warmup_steps=1, observed_steps=1,
                  parameters=sum(parameter.numel() for parameter in model.parameters()))
    record["input_sha256"]=hashlib.sha256(inputs_cpu.numpy().tobytes()).hexdigest()
    record["working_tree_dirty"] = (None if os.environ.get("TIDE_SOURCE_COMMIT") else bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src/tide", "benchmarks/profile.py"], text=True).strip()))
    if device.type == "npu":
        record["torch_npu"] = str(torch_npu.__version__)
        record["allow_hf32"] = {"matmul":torch.npu.matmul.allow_hf32,"conv":torch.npu.conv.allow_hf32}
    persist(output, record)
    try:
        step(model, inputs, optimizer, args.mode, device)
        stages = {label: dict(calls=0, seconds=0.0) for label in STAGES}
        layouts = {}
        observation = step(model, inputs, optimizer, args.mode, device, stages, layouts)
        observation["stages"] = stages
        observation["selected_query_packing"] = layouts
        observation["forward_unattributed_seconds"] = (observation["phases_seconds"]["forward"]
                                                       - sum(stage["seconds"] for stage in stages.values()))
        record["intrusive_observation"] = observation
        persist(output, record)
        if args.cpu_operators:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
                step(model, inputs, optimizer, args.mode, device)
            groups = [dict(operator=event.key, count=event.count, input_shapes=event.input_shapes,
                           self_cpu_time_us=event.self_cpu_time_total,cpu_time_us=event.cpu_time_total)
                      for event in profile.key_averages(group_by_input_shape=True)]
            record["cpu_operator_observation"] = dict(
                scope="separate unwrapped CPU step; counts/shapes and instrumented CPU times; not device kernel times",
                total_shape_groups=len(groups), groups=sorted(groups, key=lambda row: row["self_cpu_time_us"], reverse=True)[:100])
        record["state"] = "passed"
    except Exception as error:
        record.update(state="failed", error=repr(error))
        raise
    finally:
        persist(output, record)
    print(json.dumps(dict(state=record["state"], output=str(output), observation=record["intrusive_observation"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
