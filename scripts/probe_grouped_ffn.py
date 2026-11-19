"""Finite FP32 grouped-matmul capability probe, independent of Tide modules.

Run only through the project's device admission/finite-job launcher. This is
operation/gradient evidence, not a performance benchmark or FFN qualification.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
import warnings


def atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, stage = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(stage):
            os.unlink(stage)


def tensor_record(tensor):
    raw = tensor.detach().cpu().contiguous()
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                dtype=str(tensor.dtype), device=str(tensor.device),
                sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest())


def comparison(actual, expected, *, atol, rtol):
    import torch
    actual = actual.detach().cpu()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return dict(state="shape_or_dtype_mismatch", actual=tensor_record(actual), expected=tensor_record(expected))
    difference = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
    scaled = difference / (atol + rtol * expected.abs())
    return dict(state="within_threshold" if finite and bool((scaled <= 1).all()) else "outside_threshold",
                finite=finite, atol=atol, rtol=rtol,
                max_abs=float(difference.max()) if difference.numel() else 0.,
                max_scaled=float(scaled.max()) if scaled.numel() else 0.,
                exact=bool(torch.equal(actual, expected)))


def metadata_or_error(reader):
    """Optional schema/registration introspection must not block operations."""
    try:
        return reader()
    except Exception as error:
        return dict(state="metadata_unavailable", error_type=type(error).__name__, error=str(error))


def fixture(torch, generator, owners, groups=4, input_width=32, output_width=48):
    """Literal owner-wise formulas; shared-owner gradients sum outer products."""
    rows = len(owners)
    x = .1 * torch.randn(rows, input_width, generator=generator)
    weight = .1 * torch.randn(groups, output_width, input_width, generator=generator)
    dy = .1 * torch.randn(rows, output_width, generator=generator)
    order = sorted(range(rows), key=lambda row: owners[row])
    counts = [owners.count(group) for group in range(groups)]
    ends = [sum(counts[:group + 1]) for group in range(groups)]
    expected_y = torch.stack([weight[owner] @ x[row] for row, owner in enumerate(owners)])
    expected_dx = torch.stack([dy[row] @ weight[owner] for row, owner in enumerate(owners)])
    expected_dw = torch.zeros_like(weight)
    for row, owner in enumerate(owners):
        expected_dw[owner] += torch.outer(dy[row], x[row])
    return dict(x=x[order], weight=weight, dy=dy[order], ends=torch.tensor(ends, dtype=torch.int64),
                expected_y=expected_y[order], expected_dx=expected_dx[order], expected_dw=expected_dw,
                owners=owners, order=order, counts=counts)


def cpu_formula_check(torch, data):
    x = data["x"].detach().clone().requires_grad_()
    weight = data["weight"].detach().clone().requires_grad_()
    sorted_owners = [data["owners"][row] for row in data["order"]]
    y = torch.stack([weight[owner] @ x[row] for row, owner in enumerate(sorted_owners)])
    dx, dw = torch.autograd.grad(y, (x, weight), data["dy"])
    return dict(output=comparison(y, data["expected_y"], atol=2e-6, rtol=2e-6),
                input_gradient=comparison(dx, data["expected_dx"], atol=6e-6, rtol=6e-6),
                weight_gradient=comparison(dw, data["expected_dw"], atol=6e-6, rtol=6e-6))


def observe(torch, device, operation, expected, *, gradient=False):
    report = dict(state="running", boundary="explicit device synchronization before and after this operation; no throughput claim")
    caught = []
    try:
        torch.npu.synchronize(device)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = operation()
            torch.npu.synchronize(device)
        report["operation_seconds"] = time.perf_counter() - started
        outputs = output if isinstance(output, (list, tuple)) else (output,)
        report["outputs"] = [tensor_record(tensor) for tensor in outputs]
        report["comparisons"] = [comparison(tensor, reference, atol=6e-6 if gradient else 2e-6,
                                            rtol=6e-6 if gradient else 2e-6)
                                  for tensor, reference in zip(outputs, expected)]
        report["native_device_outputs"] = all(tensor.device == device for tensor in outputs)
        report["state"] = ("within_threshold" if len(outputs) == len(expected) and report["native_device_outputs"]
                           and all(row["state"] == "within_threshold" for row in report["comparisons"])
                           else "comparison_failed")
    except Exception as error:
        report.update(state="operation_failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        try:
            torch.npu.synchronize(device)
            report["synchronization_after_error"] = "completed"
        except Exception as sync_error:
            report["synchronization_after_error"] = repr(sync_error)
    report["warnings"] = [dict(category=type(item.message).__name__, message=str(item.message)) for item in caught]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    record = dict(schema="tide-grouped-ffn-capability-probe-v1", state="starting",
                  observed=datetime.now(timezone.utc).isoformat(), arguments=vars(args) | {"output": str(args.output)},
                  source=os.environ.get("TIDE_SOURCE_COMMIT"),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), cases=[],
                  limits=["Capability probe only, no normal timing or whole-model performance qualification.",
                          "Dispatch placement and output device do not by themselves exclude an internal vendor fallback.",
                          "Native autograd and explicit grouped adjoint operations are tested independently.",
                          "No owner-indexed weight tensor, dense adapter, or weight contiguous copy is created.",
                          "Thresholds retain NodeLinear local FP32 value 2e-6 and gradient 6e-6 atol/rtol."])
    atomic(args.output, record)
    try:
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        record["torch"] = torch.__version__
        if not args.reference_only:
            import torch_npu
            device = torch.device(args.device)
            if device.type != "npu":
                raise ValueError("device execution requires an explicit NPU; use --reference-only for CPU formulas")
            torch.npu.set_device(device)
            torch.npu.matmul.allow_hf32 = False
            torch.npu.conv.allow_hf32 = False
            record.update(torch_npu=torch_npu.__version__, device=str(device),
                          device_name=torch.npu.get_device_name(device),
                          precision=dict(input="float32", weight="float32", output="float32", autocast=False,
                                         matmul_allow_hf32=torch.npu.matmul.allow_hf32, conv_allow_hf32=torch.npu.conv.allow_hf32),
                          schema_default=metadata_or_error(lambda: str(torch.ops.npu.npu_grouped_matmul.default._schema)),
                          schema_list=metadata_or_error(lambda: str(torch.ops.npu.npu_grouped_matmul.List._schema)),
                          dispatch_registration=metadata_or_error(lambda: torch._C._dispatch_dump_table("npu::npu_grouped_matmul")))
        generator = torch.Generator().manual_seed(args.seed)
        for label, owners in (("repeated_shared_owner_zero_middle", [2, 0, 2, 3, 0, 2, 3, 2, 0, 2]),
                              ("singleton_with_zero_prefix_and_tail", [1])):
            data = fixture(torch, generator, owners)
            case = dict(label=label, owners=owners, sort_order=data["order"], counts=data["counts"],
                        group_list=data["ends"].tolist(), inputs={key: tensor_record(data[key]) for key in ("x", "weight", "dy", "ends")},
                        cpu_formula_check=cpu_formula_check(torch, data), operations={})
            record["cases"].append(case)
            atomic(args.output, record)
            if any(value["state"] != "within_threshold" for value in case["cpu_formula_check"].values()):
                raise RuntimeError("independent CPU formula/adjoint self-check failed")
            if args.reference_only:
                continue
            x, weight, dy, ends = (data[key].to(device) for key in ("x", "weight", "dy", "ends"))
            transposed_weight = weight.transpose(-1, -2)
            case["device_inputs"] = {key: tensor_record(value) for key, value in
                                     (("x", x), ("weight", weight), ("weight_transpose_view", transposed_weight), ("dy", dy), ("group_list", ends))}
            case["weight_transpose_shares_storage"] = transposed_weight.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
            def grouped(a, b, group_type):
                with torch.autocast(device.type, enabled=False):
                    return torch_npu.npu_grouped_matmul([a], [b], group_list=ends,
                        split_item=3, group_type=group_type, group_list_type=0)[0]
            actions = (
                ("forward_group0", lambda: grouped(x, transposed_weight, 0), [data["expected_y"]], False),
                ("input_adjoint_group0", lambda: grouped(dy, weight, 0), [data["expected_dx"]], True),
                ("weight_adjoint_group2", lambda: grouped(dy.T, x, 2), [data["expected_dw"]], True),
            )
            for name, operation, expected, gradient in actions:
                case["operations"][name] = observe(torch, device, operation, expected, gradient=gradient)
                atomic(args.output, record)
            def native_autograd():
                left = x.detach().requires_grad_()
                bank = weight.detach().requires_grad_()
                output = grouped(left, bank.transpose(-1, -2), 0)
                case["native_output_requires_grad"] = output.requires_grad
                case["native_output_grad_fn"] = type(output.grad_fn).__name__ if output.grad_fn else None
                return torch.autograd.grad(output, (left, bank), dy)
            case["operations"]["native_autograd"] = observe(torch, device, native_autograd,
                [data["expected_dx"], data["expected_dw"]], gradient=True)
            atomic(args.output, record)
        record["state"] = "completed_reference_only" if args.reference_only else "completed_probe"
        record["operation_states"] = {case["label"]: {name: value["state"] for name, value in case["operations"].items()}
                                      for case in record["cases"]}
        atomic(args.output, record)
        print(json.dumps(dict(state=record["state"], output=str(args.output), operations=record["operation_states"])))
        return 0
    except Exception as error:
        record.update(state="probe_setup_failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        atomic(args.output, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
