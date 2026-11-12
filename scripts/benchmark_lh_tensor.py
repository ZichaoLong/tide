#!/usr/bin/env python3
"""Repeatable direct → public column plan → default executor LH/HB comparison.

Each invocation measures one level/mode in its own process. The independent
direct anchor has dense Full work and a raw-state/output numerical projection;
public/default retain selected Full and public continuation responsibilities.
The result records these differences instead of calling their ratio overhead.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from durable_records import write_json


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", choices=("direct", "public", "default"), required=True)
    p.add_argument("--client", choices=("python", "native"), default="python")
    p.add_argument("--case", choices=("many-lh", "many-hb", "lh-1b", "lh-9b"), default="many-lh")
    p.add_argument("--width", type=int)
    p.add_argument("--depth", type=int)
    p.add_argument("--region-width", type=int)
    p.add_argument("--profile", choices=("baseline", "textbook"))
    p.add_argument("--plan", type=Path, help="explicit resolved workload instead of the named recipe")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--length", type=int, default=16)
    p.add_argument("--input-pattern", choices=("dense", "staggered-v1"), default="staggered-v1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--mode", choices=("inference", "grad-forward", "training"), default="training")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--native-library")
    p.add_argument("--max-materialize-bytes", type=int, default=128*1024**3)
    p.add_argument("--phase-timing", action="store_true", help="intrusive per-phase synchronization; separate diagnostic identity")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if min(a.batch, a.length, a.threads, a.max_materialize_bytes) < 1 or a.length > 256:
        p.error("positive dimensions/budget and L<=256 required")
    if a.level == "direct" and a.client != "python":
        p.error("independent direct formulas have a Python tensor implementation")
    if a.client == "native" and not a.native_library:
        p.error("native needs an explicit matching library")
    return a


def snapshot(model):
    """One CPU copy per exact aliased tensor, with hashes by public role name."""
    data, unique, digest = {}, {}, hashlib.sha256()
    for name, value in model.state_dict().items():
        key = (value.device, value.dtype, value.untyped_storage().data_ptr(), value.storage_offset(),
               tuple(value.shape), tuple(value.stride()))
        if key not in unique:
            unique[key] = value.detach().cpu().clone()
        data[name] = unique[key]
        digest.update(name.encode()+b"\0"+str((tuple(value.shape), value.dtype)).encode()+b"\0")
        digest.update(data[name].contiguous().numpy().tobytes())
    return data, digest.hexdigest(), sum(t.numel()*t.element_size() for t in unique.values())


def material(plan, dtype):
    import torch
    from tidegraph.planning.input_pattern import coordinates
    b, length, d = plan["samples"], plan["positions"], plan["width"]
    x = .2*torch.sin(torch.arange(d)*.19+.3*torch.arange(length)[:, None]+.1*torch.arange(b)[:, None, None])
    support = torch.zeros((len(plan["graph"]["inputs"]), b, length), dtype=torch.bool)
    for batch, position, port, _ in coordinates(plan):
        support[port, batch, position] = True
    return x.to(dtype), support


def ports(x, support):
    import torch
    from tidegraph.weighted.column_layout import SequenceLayout
    from tidegraph.weighted.message_columns import MessageColumns
    b, length, d = x.shape
    layout = SequenceLayout(torch.arange(b+1, dtype=torch.int64)*length,
                            torch.arange(length, dtype=torch.int64).repeat(b))
    result = {}
    for p in range(support.shape[0]):
        rows = support[p].flatten().nonzero().flatten()
        result[p] = MessageColumns(rows, x.new_ones(len(rows)), x.reshape(-1, d).index_select(0, rows.to(x.device)))
    return layout, result


def output_values(result, x, count):
    import torch
    values = []
    for p in range(count):
        row = result.outputs[p]
        weighted = row.strength[:, None]*row.content
        values.append(x.new_zeros((x.shape[0]*x.shape[1], x.shape[2])).index_copy(0, row.rows.to(x.device), weighted))
    return torch.stack(values).reshape(count, *x.shape)


def main():
    a = arguments()
    import torch
    import tidegraph
    from tidegraph.runtime import resolve_device
    from tidegraph import ExecutionPlacement
    from tidegraph.settle import SettleGraph
    from tidegraph.planning.materialize import materialize
    from tidegraph.planning.recipes import prepare
    from tidegraph.experiment.finite import finite_tensors
    from layer_tensor_baseline import numerics
    from lh_tensor_baseline import StaticKV
    from prefill_regression_config import CASES
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    device, _ = resolve_device(a.device)
    policy = numerics(device, "ieee")
    dtype = getattr(torch, a.dtype)
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    record = dict(schema="tide-lh-three-layer-cost-v1", source=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        package=str(Path(tidegraph.__file__).resolve()), affinity=sorted(os.sched_getaffinity(0)),
        device_visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), numerical_policy=policy,
        execution="running", observations=[], strict_numerical="not_checked_at_measured_scale",
        restore="not_checked_at_measured_scale", mathematical_projection="per-port cG; exact raw terminal fields retained",
        timing="one reset warmup, one measured step including input preparation/transfer, packing, forward, output loss, finite checks, backward, SGD, cleanup and synchronization; construction/reset excluded",
        comparison="direct uses dense Full plus masking and padded raw state; public/default selected Full; no complete independent event-root equivalence claim")
    write_json(out/"result.json", record)
    def sync():
        if device.type != "cpu":
            getattr(torch, device.type).synchronize(device)
    try:
        began = time.perf_counter()
        if a.plan:
            plan = json.loads(a.plan.read_text())
        else:
            recipe = dict(CASES[a.case], family="kv")
            for key, value in (("width", a.width), ("depth", a.depth), ("region_width", a.region_width), ("profile", a.profile)):
                if value is not None:
                    recipe[key] = value
            if a.width is not None:
                recipe.pop("target_parameters", None)
            plan, _ = prepare(recipe, samples=a.batch, positions=a.length, seed=7,
                              input_pattern=None if a.input_pattern == "dense" else a.input_pattern)
        binding, _, _ = materialize(plan, max_bytes=a.max_materialize_bytes, dtype=dtype, include_inputs=False)
        initial, digest, snapshot_bytes = snapshot(binding.model)
        sample_x, sample_support = material(plan, dtype)
        input_hash = hashlib.sha256(sample_x.numpy().tobytes()+sample_support.numpy().tobytes()).hexdigest()
        del sample_x, sample_support
        record.update(plan=plan, initial_parameters_sha256=digest, reset_snapshot_bytes=snapshot_bytes,
                      input_values_and_support_sha256=input_hash,
                      parameters=sum(p.numel() for p in binding.model.parameters() if p.requires_grad))
        binding.model.to(device)
        spec = SettleGraph(binding.graph, tuple(plan["graph"]["ranks"]), binding="ports", time_stride=plan["stride"])
        placement = ExecutionPlacement(preset="cpu" if device.type == "cpu" else "mixed-c", scoring_dtype=a.dtype)
        if a.level == "direct":
            runner = StaticKV(binding, spec)
        elif a.level == "public":
            from tidegraph.weighted.settle_columns import SettleColumnPlan
            runner = SettleColumnPlan(binding, spec, implementation=a.client, native_library=a.native_library, placement=placement)
        else:
            from tidegraph.weighted.execution import eager_runner
            runner = eager_runner(binding, a.client, placement, native_library=a.native_library, trace=False)
        if a.client == "native":
            record["native_build"] = json.loads((Path(a.native_library)/"build-manifest.json").read_text())
        parameters = [p for p in binding.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=1e-5, momentum=.3) if a.mode == "training" else None
        sync(); record["construction_seconds"] = time.perf_counter()-began
        for iteration in range(2):
            binding.model.load_state_dict(initial)
            if optimizer:
                optimizer.state.clear(); optimizer.zero_grad(set_to_none=True)
            sync(); began = time.perf_counter(); phases = {}
            def phase(name, fn):
                if a.phase_timing: sync()
                started = time.perf_counter()
                result = fn()
                if a.phase_timing: sync()
                phases[name] = time.perf_counter()-started
                return result
            x, support = material(plan, dtype)
            active_positions = int(support.any(0).sum())
            x = x.to(device).requires_grad_(a.mode != "inference")
            with torch.set_grad_enabled(a.mode != "inference"):
                if a.level == "direct":
                    result = phase("forward", lambda: runner(x, support=support.to(device)))
                    y = result["output_controls"][..., None]*result["output_contents"]
                    state = result["states"]
                else:
                    layout, payloads = ports(x, support)
                    result = phase("forward", lambda: runner.run_ports(layout, payloads) if a.level == "public" else
                        runner.run_ports(spec, layout, payloads, stop_position=plan["positions"]))
                    y = output_values(result, x, len(binding.graph.outputs))
                    state = result.states if a.level == "public" else result.continuation
                loss = y.square().sum()*1e-6
                phase("finite_forward", lambda: finite_tensors(dict(y=y, state=state, loss=loss), packing="packed"))
                if optimizer:
                    phase("backward", loss.backward)
                    phase("finite_gradients", lambda: finite_tensors([p.grad for p in parameters], packing="packed"))
                    phase("optimizer", optimizer.step)
                    phase("finite_parameters", lambda: finite_tensors(parameters, packing="packed"))
            if a.level == "direct":
                counts = result["counts"].cpu().tolist()
                work = dict(zip(("candidate_events", "selected_events", "visited_edges"), counts))
                work["physical_full_rows"] = result["physical_full_rows"]
                record["effective_execution"] = result["implementation"]
            else:
                work = dict(result.stats); work["physical_full_rows"] = work["selected_events"]
                record["effective_execution"] = result.execution if a.level == "default" else "public SettleColumnPlan"
            observed_loss = float(loss.detach())
            del y, state, loss, result, x
            if a.level != "direct": del layout, payloads
            sync(); seconds = time.perf_counter()-began
            row = dict(warmup=iteration == 0, seconds=seconds, input_positions=active_positions,
                       input_positions_per_second=active_positions/seconds, loss=observed_loss, actual_work=work)
            row["synchronized_phase_seconds" if a.phase_timing else "host_submission_phase_seconds"] = phases
            if device.type != "cpu":
                api = getattr(torch, device.type)
                row.update(peak_allocated_bytes=api.max_memory_allocated(device), peak_reserved_bytes=api.max_memory_reserved(device))
            record["observations"].append(row); write_json(out/"result.json", record)
            print(json.dumps(row), flush=True)
        record["execution"] = "completed"
    except BaseException as error:
        record.update(execution="failed", error=repr(error)); raise
    finally:
        write_json(out/"result.json", record)


if __name__ == "__main__":
    main()
