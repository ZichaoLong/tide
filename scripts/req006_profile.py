"""Opt-in scopes and trace accounting, separate from uninstrumented timing."""
from collections import Counter, defaultdict
from contextlib import contextmanager, ExitStack
from functools import wraps
import heapq
import json
from pathlib import Path
from unittest.mock import patch


@contextmanager
def scopes(runtime, model, optimizer, *, census=None):
    """Add labels without changing algorithms or synchronizing each stage."""
    import torch
    from tide_lab import direct_semantic, direct_semantic_memory, direct_semantic_ops, language
    import req006_runtime

    def labelled(name, function, *, roots=False):
        @wraps(function)
        def call(*args, **kwargs):
            if name == "backward" and census is not None:
                census.register(args[0], label="downstream/loss")
            with torch.profiler.record_function("req006::"+name):
                output = function(*args, **kwargs)
            if roots and census is not None:
                census.register(output, label="downstream/returned-logits")
            return output
        return call

    with ExitStack() as stack:
        for owner, name, label in (
            (direct_semantic.Aggregation, "forward", "aggregate"),
            (direct_semantic.Selector, "forward", "region"),
            (direct_semantic_memory.Memory, "forward", "memory"),
            (direct_semantic_memory.Memory, "kv", "kv"),
            (direct_semantic_ops.NodeLinear, "forward", "node_linear"),
            (direct_semantic, "allocation", "allocation"),
            (language.CausalBlock, "forward", "transformer_block"),
            (runtime.F, "cross_entropy", "loss"),
            (torch.Tensor, "backward", "backward"),
            (torch.nn.utils, "clip_grad_norm_", "clip"),
            (optimizer, "zero_grad", "zero_grad"),
            (optimizer, "step", "optimizer"),
            (req006_runtime, "batches_on_device", "input_upload"),
        ):
            stack.enter_context(patch.object(owner, name, labelled(label, getattr(owner, name))))
        stack.enter_context(patch.object(direct_semantic_ops._Allocation, "backward", staticmethod(
            labelled("allocation_vjp", direct_semantic_ops._Allocation.backward))))
        stack.enter_context(patch.object(model, "forward", labelled("model", model.forward, roots=True)))
        yield


def category(name):
    if name.startswith("autograd::engine::evaluate_function:"):
        return "autograd_execution"
    if name in {"aten::mm", "aten::bmm", "aten::addmm", "aten::matmul", "aten::linear"}:
        return "matrix_interfaces"
    if name.startswith(("aten::index", "aten::select", "aten::gather", "aten::scatter",
                        "aten::stack", "aten::cat", "aten::unbind")):
        return "index_and_assembly"
    if name == "aten::_local_scalar_dense" or "synchronize" in name.lower():
        return "scalar_or_synchronization"
    if name.startswith(("aten::to", "aten::_to_copy", "aten::copy_")):
        return "copies"
    return "other"


def summarize_trace(path):
    """CPU interface/engine events and actual hardware kernels remain distinct."""
    path = Path(path)
    document = json.loads(path.read_text())
    events = document if isinstance(document, list) else document["traceEvents"]
    # CANN exports decimal timestamps as JSON strings; Kineto uses numbers.
    # Normalize working records only, preserving the raw trace and its hash.
    complete = [dict(e, ts=float(e["ts"]), dur=float(e.get("dur", 0)))
                for e in events if e.get("ph") == "X" and "ts" in e]
    hardware = {e["pid"] for e in events if e.get("name") == "process_name"
                and e.get("args", {}).get("name") == "Ascend Hardware"}
    ranges, by_thread = defaultdict(list), defaultdict(list)
    for event in complete:
        thread = (event.get("pid"), event.get("tid"))
        if event.get("name", "").startswith("req006::"):
            ranges[thread].append(event)
        elif event.get("name", "").startswith(("aten::", "autograd::engine::")):
            by_thread[thread].append(event)
    operators = defaultdict(lambda: dict(calls=0, inclusive_cpu_us=0., shapes=Counter()))
    categories = Counter()
    for thread, rows in by_thread.items():
        pending = sorted(ranges[thread], key=lambda e: (e["ts"], -float(e.get("dur", 0))))
        active, next_range = [], 0
        for event in sorted(rows, key=lambda e: e["ts"]):
            timestamp = event["ts"]
            while next_range < len(pending) and pending[next_range]["ts"] <= timestamp:
                scope = pending[next_range]
                heapq.heappush(active, (-scope["ts"], float(scope.get("dur", 0)),
                                       scope["name"], scope["ts"]+float(scope.get("dur", 0))))
                next_range += 1
            while active and active[0][3] < timestamp:
                heapq.heappop(active)
            stage = active[0][2] if active else "unattributed"
            name = event["name"]
            row = operators[stage, name]
            row["calls"] += 1
            row["inclusive_cpu_us"] += max(0., float(event.get("dur", 0)))
            args = event.get("args", {})
            shapes = args.get("Input Dims", args.get("Input Shapes", "unrecorded"))
            row["shapes"][json.dumps(shapes, sort_keys=True)] += 1
            categories[category(name)] += 1
    kernels = [e for e in complete if "kernel" in str(e.get("cat", "")).lower()
               or e.get("pid") in hardware and str(e.get("args", {}).get("Task Type", "")).startswith("KERNEL")]
    durations = Counter()
    for event in kernels:
        durations[event["name"]] += max(0., float(event.get("dur", 0)))
    intervals = sorted((e["ts"], e["ts"]+max(0., float(e.get("dur", 0)))) for e in kernels)
    union, end = 0., float("-inf")
    for begin, stop in intervals:
        union += max(0., stop-max(begin, end)); end = max(end, stop)
    return dict(schema="tide-req006-trace-summary-v1", trace_bytes=path.stat().st_size,
        operator_categories=dict(categories),
        operators=[dict(stage=stage, operator=name, category=category(name), **row)
                   for (stage, name), row in sorted(operators.items())],
        kernel_events=len(kernels), kernel_types=dict(Counter(
            e.get("args", {}).get("Task Type", "device kernel") for e in kernels)),
        kernel_names=dict(Counter(e["name"] for e in kernels)),
        kernel_inclusive_us=dict(durations), kernel_union_us=union,
        phases_inclusive_us={name: sum(float(e.get("dur", 0)) for e in complete if e.get("name") == name)
                             for name in sorted({e["name"] for rows in ranges.values() for e in rows})},
        observed_host_fallback_events=[e.get("name") for e in events if any(
            text in e.get("name", "").lower() for text in ("cpu_fallback", "fallback_to_cpu", "fallback to cpu"))],
        scope="CPU interface events grouped by nearest same-thread labelled range at start; "
              "worker events may be unattributed. Inclusive ranges/operators overlap. "
              "matmul and nested mm/bmm are not separate mathematical work. Device kernel "
              "durations/union are not exclusive critical-path attribution. AICPU is not host fallback.")


def profile_call(call, device, output):
    import torch
    if device.type == "npu":
        import torch_npu
        profiler = torch_npu.profiler
        activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
    else:
        profiler = torch.profiler
        activities = [profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(profiler.ProfilerActivity.CUDA)
    with profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as recording:
        result = call()
    recording.export_chrome_trace(str(output))
    return result


if __name__ == "__main__":
    # Offline recovery of an exported trace never restarts the measured job or
    # overwrites its failure status. The new parser/source identity is explicit.
    import argparse
    import hashlib
    from durable_records import write_json
    from source_identity import source_state
    parser = argparse.ArgumentParser(description="Summarize an existing REQ-006 trace without running a model")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve prior analysis")
    source, dirty = source_state(Path(__file__).resolve().parents[1])
    result = summarize_trace(args.trace)
    result.update(analysis_source=source, analysis_dirty=dirty, trace_path=str(args.trace.resolve()),
                  trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest())
    write_json(args.output, result)
