"""Finite FP32 FFN formula/VJP and separate device-trace witness for W.

Use the project frozen launcher for NPU. This checks local FFNs with a few
owners at d192/F768 and d768/F3072; it is neither 9B training nor throughput.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback
import warnings


def literal_ffn(torch, x, tensors, groups, ids):
    """Independent owner-wise equations; no production norm or linear helper."""
    def project(name, row, owner):
        result = row @ tensors[name + ".weight"][groups[owner]].T
        if name + ".right" in tensors:
            result = result + (row @ tensors[name + ".right"][owner].T) @ tensors[name + ".left"][owner].T
        return result
    outputs = []
    for row, owner in zip(x, ids):
        z = row / torch.sqrt((row * row).mean(-1, keepdim=True) + 1e-5)
        gate = project("gate", z, owner)
        up = project("up", z, owner)
        outputs.append(row + project("down", gate * torch.sigmoid(gate) * up, owner))
    return torch.stack(outputs)


def trace_summary(path):
    payload = json.loads(Path(path).read_text())
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    categories, critical, tracks = Counter(), [], []
    for event in events:
        category = str(event.get("cat", "unspecified"))
        if event.get("ph") == "M" and event.get("name") in {"process_name", "thread_name"}:
            tracks.append({key: event.get(key) for key in ("name", "pid", "tid", "args")})
        if event.get("ph") == "X":
            categories[category] += 1
            if "groupedmatmul" in str(event.get("name", "")).lower().replace("_", ""):
                critical.append({key: event.get(key) for key in ("name", "cat", "pid", "tid", "ts", "dur", "args")})
    raw = Path(path).read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                categories=dict(categories), grouped_matmul_events=critical, tracks=tracks,
                device_verification="inspect named device tracks and GroupedMatmul events together; CPU API rows alone do not prove device execution",
                timing_boundary="separate profiled replay; not normal timing, additive wall shares or a speed comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shapes", nargs="+", choices=("tiny", "medium", "reference"), default=["medium", "reference"])
    parser.add_argument("--sharing", nargs="+", choices=("independent", "region_shared"), default=["independent", "region_shared"])
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--seed", type=int, default=467)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output identity")
    # Import after argument parsing: help remains independent of Torch/vendor.
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from scripts.probe_grouped_ffn import atomic, comparison, tensor_record
    from tide.parameters import FeedForward, NodeOwners
    from tide.precision import precision_context, precision_policy
    from benchmarks.parameter_gathers import ParameterGatherAudit
    from benchmarks.costs import state_hash, summarize_trace
    from benchmarks.inference import source_identity
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    record = dict(schema="tide-grouped-ffn-formula-vjp-v1", state="running",
                  observed=datetime.now(timezone.utc).isoformat(), source=source_identity(),
                  arguments=vars(args) | {"output": str(args.output)}, torch=torch.__version__,
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  precision=precision_policy(device, "fp32"), cases=[], failures=[],
                  scope="few-owner local FFN, first-order only; not model/private-state/checkpoint or throughput qualification",
                  thresholds=dict(values=dict(atol=2e-6, rtol=2e-6), gradients=dict(atol=6e-6, rtol=6e-6)),
                  trace_scope="at most one separate full FFN forward/backward replay, no dispatch audit inside the device profiler")
    if device.type == "npu":
        record.update(torch_npu=torch_npu.__version__, device_name=torch.npu.get_device_name(device),
                      allow_hf32=dict(matmul=torch.npu.matmul.allow_hf32, conv=torch.npu.conv.allow_hf32))
    def sync():
        if device.type == "npu":
            torch.npu.synchronize(device)
    class Witness(TorchDispatchMode):
        def __init__(self, module):
            super().__init__()
            self.banks = {(str(p.device), p.untyped_storage().data_ptr()): name for name, p in module.named_parameters()}
            self.rows = []
            self.copies = []
        def descriptor(self, value):
            return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype), device=str(value.device),
                        parameter_storage=self.banks.get((str(value.device), value.untyped_storage().data_ptr())))
        def __torch_dispatch__(self, func, types, arguments=(), kwargs=None):
            options = kwargs or {}
            output = func(*arguments, **options)
            if "npu_grouped_matmul" in str(func):
                self.rows.append(dict(operator=str(func), x=[self.descriptor(x) for x in arguments[0]],
                    weight=[self.descriptor(x) for x in arguments[1]], group_type=options.get("group_type"),
                    group_list=self.descriptor(options["group_list"]), outputs=[self.descriptor(x) for x in output]))
            if str(func) in {"aten.clone.default", "aten._to_copy.default", "aten.contiguous.default"} and arguments:
                item = arguments[0]
                if isinstance(item, torch.Tensor) and self.descriptor(item)["parameter_storage"] is not None:
                    self.copies.append(dict(operator=str(func), input=self.descriptor(item)))
            return output
    atomic(args.output, record)
    try:
        chosen = [(shape, sharing) for shape in args.shapes for sharing in args.sharing]
        for index, (shape, sharing) in enumerate(chosen):
            torch.manual_seed(args.seed + index)
            width, hidden = {"tiny": (16, 32), "medium": (192, 768), "reference": (768, 3072)}[shape]
            groups = None if sharing == "independent" else [0, 1, 0, 1]
            host_groups = list(range(4)) if groups is None else groups
            ids = [3, 0, 3, 1]
            cpu_model = FeedForward(4, width, groups, rank=8, hidden_features=hidden, weight_access="gather")
            initial = {name: value.detach().clone() for name, value in cpu_model.state_dict().items()}
            leaves = {name: p.detach().clone().requires_grad_() for name, p in cpu_model.named_parameters()}
            cpu_input = (.1 * torch.randn(4, 1, 2, width)).requires_grad_()
            root = .1 * torch.sin(torch.arange(cpu_input.numel(), dtype=torch.float32)).reshape_as(cpu_input)
            expected = literal_ffn(torch, cpu_input, leaves, host_groups, ids)
            reference_grad = torch.autograd.grad(expected, (cpu_input, *leaves.values()), root)
            expected, reference_grad = expected.detach(), [value.detach() for value in reference_grad]
            row = dict(shape=shape, sharing=sharing, width=width, hidden=hidden, owner_ids=ids,
                       groups=host_groups, input=tensor_record(cpu_input), initial_parameters_sha256=state_hash(initial),
                       parameters=sum(p.numel() for p in cpu_model.parameters()), paths={})
            record["cases"].append(row)
            atomic(args.output, record)
            for access in ("gather", "grouped"):
                model = FeedForward(4, width, groups, rank=8, hidden_features=hidden, weight_access=access)
                model.load_state_dict(initial)
                model = model.to(device)
                x = cpu_input.detach().to(device, copy=True).requires_grad_()
                owners = NodeOwners(torch.tensor(ids, dtype=torch.int64, device=device), (0, 1, 2, 3))
                witness = Witness(model)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with ParameterGatherAudit(model) as gather_audit, witness, precision_context(device, "fp32"):
                        output = model(x, owners)
                        gradients = torch.autograd.grad(output, (x, *model.parameters()), root.to(device))
                    sync()
                values = comparison(output, expected, atol=2e-6, rtol=2e-6)
                grad = {name: comparison(actual, reference, atol=6e-6, rtol=6e-6)
                        for name, actual, reference in zip(("input", *leaves), gradients, reference_grad)}
                result = dict(output=values, gradients=grad, output_identity=tensor_record(output),
                              parameter_gathers=gather_audit.report(), grouped_dispatch=witness.rows,
                              parameter_layout_copies=witness.copies,
                              warnings=[dict(category=type(w.message).__name__, message=str(w.message)) for w in caught])
                row["paths"][access] = result
                bad = [name for name, item in dict(output=values, **grad).items() if item["state"] != "within_threshold"]
                if bad:
                    record["failures"].append(dict(shape=shape, sharing=sharing, access=access, fields=bad))
                if access == "grouped" and gather_audit.report()["output_bytes"]:
                    raise AssertionError("grouped FFN materialized an indexed Parameter")
                if access == "grouped" and witness.copies:
                    raise AssertionError("grouped FFN copied a Parameter layout at the ATen surface")
                if device.type == "npu" and access == "grouped":
                    if not witness.rows or {r["group_type"] for r in witness.rows} != {0, 2}:
                        raise AssertionError("grouped FFN did not witness both forward/input and weight adjoint native operations")
                    if not all(r["weight"][0]["parameter_storage"] is not None for r in witness.rows if r["group_type"] == 0):
                        raise AssertionError("native group0 must read an original Parameter bank/view")
                atomic(args.output, record)
                if args.trace and access == "grouped" and index == len(chosen) - 1:
                    trace_path = args.output.with_suffix(".trace.json")
                    profiler = torch_npu.profiler if device.type == "npu" else torch.profiler
                    activities = [profiler.ProfilerActivity.CPU]
                    if device.type == "npu":
                        activities.append(profiler.ProfilerActivity.NPU)
                    with profiler.profile(activities=activities, record_shapes=True) as profile:
                        with precision_context(device, "fp32"):
                            replay = model(x, owners)
                            torch.autograd.grad(replay, (x, *model.parameters()), root.to(device))
                        sync()
                    profile.export_chrome_trace(str(trace_path))
                    record["trace"] = trace_summary(trace_path)
                    record["trace"]["category_summary"] = summarize_trace(trace_path)
                    atomic(args.output, record)
                del model, x, output, gradients
            del cpu_model, initial, leaves, cpu_input, expected, reference_grad
        record["state"] = "completed_with_numerical_failures" if record["failures"] else "within_threshold"
        atomic(args.output, record)
        print(json.dumps(dict(state=record["state"], cases=len(record["cases"]), failures=record["failures"], output=str(args.output))))
        return int(bool(record["failures"]))
    except Exception as error:
        record.update(state="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        atomic(args.output, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
