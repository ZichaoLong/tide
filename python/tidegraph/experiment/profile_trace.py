"""Offline device timeline summary; no runtime or throughput claim."""
from collections import Counter
import hashlib
import json

def inspect_trace(path):
    from .profile_sync import summarize_cuda_sync
    document = json.loads(path.read_text())
    events = document if isinstance(document, list) else document["traceEvents"]
    hardware = {e["pid"] for e in events if e.get("name") == "process_name"
                and e.get("args", {}).get("name") == "Ascend Hardware"}
    kernels = [e for e in events if e.get("ph") == "X" and
               ("kernel" in str(e.get("cat", "")).lower() or
                e.get("pid") in hardware and str(e.get("args", {}).get("Task Type", "")).startswith("KERNEL"))]
    if not kernels:
        raise RuntimeError("profile has no actual accelerator kernels")
    fallback = [e.get("name") for e in events if any(x in str(e.get("name", "")).lower()
                for x in ("cpu_fallback", "fallback_to_cpu", "fallback to cpu"))]
    if fallback:
        raise RuntimeError("unexpected host fallback events: "+repr(fallback[:10]))
    transfers = Counter(e.get("name") for e in events if any(x in str(e.get("name", "")).lower()
                        for x in ("memcpy", "mem_copy", "_local_scalar_dense")))
    matrices = Counter()
    for event in events:
        if event.get("ph") == "X" and event.get("name") in {"aten::mm", "aten::bmm", "aten::addmm"}:
            args = event.get("args", {})
            shapes = args.get("Input Dims", args.get("Input Shapes", "unrecorded"))
            matrices[(event["name"], json.dumps(shapes, sort_keys=True))] += 1
    def durations(rows):
        total = Counter()
        for e in rows:
            if e.get("ph") == "X":
                total[e.get("name", "unknown")] += max(0., float(e.get("dur", 0.)))
        return dict(total.most_common())
    intervals = sorted((float(e["ts"]), float(e["ts"])+max(0., float(e.get("dur", 0.)))) for e in kernels)
    union, end = 0., float("-inf")
    for start, stop in intervals:
        union += max(0., stop-max(start, end)); end = max(end, stop)
    return dict(matrix_shapes=[dict(operator=name, input_shapes=json.loads(shape), calls=count)
                               for (name, shape), count in sorted(matrices.items())],
                kernel_events=len(kernels), kernel_names=dict(Counter(e["name"] for e in kernels)),
                kernel_types=dict(Counter(e.get("args", {}).get("Task Type", "cuda") for e in kernels)),
                transfer_and_scalar_events=dict(transfers), observed_host_fallback_events=fallback,
                kernel_duration_us=durations(kernels), kernel_union_us=union,
                phases_inclusive_us=durations(e for e in events if str(e.get("name", "")).startswith("tide.")),
                transfer_and_scalar_inclusive_us=durations(e for e in events if e.get("name") in transfers),
                host_operators_inclusive_us=durations(e for e in events if e.get("cat") == "cpu_op"),
                cuda_host_synchronization=summarize_cuda_sync(events),
                duration_caveat="inclusive host events overlap; kernel union is covered device time, not exclusive critical-path attribution",
                trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

