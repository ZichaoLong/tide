"""Two fixed d768 FFN output discrepancies; bounded same-data diagnosis.

No thresholds or production code change. Root owns the frozen <=5 minute NPU
job. CPU FP64 is an independent diagnostic root, not a training precision.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback


def file_identity(path):
    data = Path(path).read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def detailed_compare(torch, actual, reference, gradient=False):
    """Evaluate cross-dtype error in FP64, retaining original tolerance values."""
    a, b = actual.detach().cpu().double(), reference.detach().cpu().double()
    tolerance = 6e-6 if gradient else 2e-6
    if a.shape != b.shape:
        return dict(state="shape_mismatch", actual=list(a.shape), reference=list(b.shape))
    delta = (a-b).abs()
    scale = delta/(tolerance+tolerance*b.abs())
    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    worst = int(scale.flatten().argmax()) if scale.numel() else 0
    return dict(state="within_threshold" if finite and bool((scale <= 1).all()) else "outside_threshold",
                atol=tolerance, rtol=tolerance, finite=finite, max_abs=float(delta.max()),
                max_scaled=float(scale.max()), failed_elements=int((scale > 1).sum()), elements=a.numel(),
                worst_flat_index=worst, worst_actual=float(a.flatten()[worst]), worst_reference=float(b.flatten()[worst]),
                actual_dtype=str(actual.dtype), reference_dtype=str(reference.dtype), exact=bool(torch.equal(a, b)),
                comparison_arithmetic="CPU FP64 difference/scaling of the retained values; no tolerance widening")


def literal_projection(torch, x, tensors, groups, ids, name):
    outputs = []
    for row, owner in zip(x, ids):
        value = row @ tensors[name+".weight"][groups[owner]].T
        if name+".right" in tensors:
            value = value + (row @ tensors[name+".right"][owner].T) @ tensors[name+".left"][owner].T
        outputs.append(value)
    return torch.stack(outputs)


def literal_stages(torch, x, tensors, groups, ids):
    # Match the original oracle's exact per-owner operation shapes/order.
    def project(name, row, owner):
        result = row @ tensors[name+".weight"][groups[owner]].T
        if name+".right" in tensors:
            result = result + (row @ tensors[name+".right"][owner].T) @ tensors[name+".left"][owner].T
        return result
    stages = {key:[] for key in ("norm_gate", "norm_up", "gate", "up", "product", "down", "output")}
    for row, owner in zip(x, ids):
        z = row/torch.sqrt((row*row).mean(-1, keepdim=True)+1e-5)
        gate, up = project("gate",z,owner), project("up",z,owner)
        product = gate*torch.sigmoid(gate)*up
        down = project("down",product,owner)
        for key,value in dict(norm_gate=z,norm_up=z,gate=gate,up=up,product=product,down=down,output=row+down).items():
            stages[key].append(value)
    return {key:torch.stack(values) for key,values in stages.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output identity")
    import torch
    from scripts.probe_grouped_ffn import atomic, comparison, tensor_record
    from scripts.check_grouped_ffn import literal_ffn
    from benchmarks.costs import state_hash
    from benchmarks.inference import source_identity
    from tide.parameters import FeedForward, NodeOwners, norm
    from tide.precision import precision_context, precision_policy
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not args.device.startswith("npu"):
        raise ValueError("this fixed follow-up requires an explicitly admitted NPU")
    import torch_npu
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    original = json.loads(args.qualification.read_text())
    source = source_identity()
    if source["tide_content_sha256"] != original["source"]["tide_content_sha256"]:
        raise ValueError("production tree differs from the original qualification; do not relabel this as fixed-source diagnosis")
    record = dict(schema="tide-grouped-ffn-bounded-output-diagnosis-v1", state="running",
        observed=datetime.now(timezone.utc).isoformat(), source=source, qualification=file_identity(args.qualification),
        arguments=vars(args) | {"qualification": str(args.qualification), "output": str(args.output)},
        torch=torch.__version__, torch_npu=torch_npu.__version__, device_name=torch.npu.get_device_name(device),
        physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), precision=precision_policy(device, "fp32"),
        allow_hf32=dict(matmul=torch.npu.matmul.allow_hf32, conv=torch.npu.conv.allow_hf32), cases=[],
        scope=["Only original d768/F3072 independent/shared fixtures at seeds469/470.",
               "Original CPU FP32 qualification failures remain authoritative records; this adds narrower comparisons.",
               "CPU FP64 converts the original FP32 input/parameters exactly; no new FP64 training qualification.",
               "All gradient checks here are input VJPs. Original all-parameter gradient results are retained, not rerun.",
               "Production FFN hooks capture actual intermediates; conditional common-input probes are separately labelled.",
               "No timing, throughput, new device trace, or model/private-history equivalence claim."])
    payload = {}
    atomic(args.output, record)
    def cpu(value):
        return value.detach().cpu().clone()
    def report_pair(a, b):
        return {key:detailed_compare(torch, a[key], b[key], gradient=key.endswith("vjp")) for key in a if key in b}
    def literal_capture(x, tensors, groups, ids, root):
        stages = literal_stages(torch, x, tensors, groups, ids)
        stages["input_vjp"] = torch.autograd.grad(stages["output"], x, root)[0]
        return {key:cpu(value) for key,value in stages.items()}
    def production_capture(model, x, owners, root):
        stages, hooks = {}, []
        for name in ("gate", "up", "down"):
            module = getattr(model, name)
            before = "product" if name == "down" else "norm_"+name
            def before_hook(_, arguments, key=before):
                stages[key] = arguments[0]
            def after_hook(_, arguments, output, key=name):
                stages[key] = output
            hooks.append(module.register_forward_pre_hook(before_hook))
            hooks.append(module.register_forward_hook(after_hook))
        try:
            with precision_context(x.device, "fp32"):
                stages["output"] = model(x, owners)
                stages["input_vjp"] = torch.autograd.grad(stages["output"], x, root)[0]
            if x.device.type == "npu":
                torch.npu.synchronize(x.device)
            return {key:cpu(value) for key,value in stages.items()}
        finally:
            for handle in hooks:
                handle.remove()
    try:
        for sharing, seed in (("independent", 469), ("region_shared", 470)):
            prior = next(case for case in original["cases"] if case["shape"] == "reference" and case["sharing"] == sharing)
            torch.manual_seed(seed)
            groups = None if sharing == "independent" else [0, 1, 0, 1]
            host_groups = list(range(4)) if groups is None else groups
            ids = [3, 0, 3, 1]
            model = FeedForward(4, 768, groups, rank=8, hidden_features=3072, weight_access="gather")
            initial = {name:value.detach().clone() for name,value in model.state_dict().items()}
            # Preserve the original constructor-to-randn RNG order exactly.
            original_x = .1*torch.randn(4, 1, 2, 768)
            root = .1*torch.sin(torch.arange(original_x.numel(), dtype=torch.float32)).reshape_as(original_x)
            input_id, parameter_hash = tensor_record(original_x), state_hash(initial)
            if input_id["sha256"] != prior["input"]["sha256"] or parameter_hash != prior["initial_parameters_sha256"]:
                raise AssertionError("original input/parameter bytes did not reproduce")
            row = dict(sharing=sharing, seed=seed, input=input_id, initial_parameters_sha256=parameter_hash,
                original_input_and_parameter_bytes_match=True, original_failures={key:value["output"] for key,value in prior["paths"].items()},
                original_all_parameter_gradient_max_scaled={key:max(g["max_scaled"] for g in value["gradients"].values()) for key,value in prior["paths"].items()})
            record["cases"].append(row)
            atomic(args.output, record)
            tensors32 = {name:p.detach() for name,p in model.named_parameters()}
            tensors64 = {name:p.double() for name,p in tensors32.items()}
            captured = {}
            captured["cpu_formula_fp32"] = literal_capture(original_x.clone().requires_grad_(), tensors32, host_groups, ids, root)
            original_formula = literal_ffn(torch, original_x, tensors32, host_groups, ids)
            if not torch.equal(original_formula, captured["cpu_formula_fp32"]["output"]):
                raise AssertionError("stage capture changed the original CPU independent formula output")
            captured["cpu_formula_fp64"] = literal_capture(original_x.double().requires_grad_(), tensors64, host_groups, ids, root.double())
            model.requires_grad_(False)
            captured["cpu_production_fp32"] = production_capture(model, original_x.clone().requires_grad_(),
                NodeOwners(torch.tensor(ids), (0, 1, 2, 3)), root)
            models, owners = {}, NodeOwners(torch.tensor(ids, dtype=torch.int64, device=device), (0, 1, 2, 3))
            for access in ("gather", "grouped"):
                current = FeedForward(4, 768, groups, rank=8, hidden_features=3072, weight_access=access)
                current.load_state_dict(initial)
                models[access] = current.requires_grad_(False).to(device)
                captured["npu_"+access] = production_capture(models[access], original_x.to(device).requires_grad_(), owners, root.to(device))
            row["against_fp64_formula"] = {key:report_pair(value, captured["cpu_formula_fp64"]) for key,value in captured.items() if key != "cpu_formula_fp64"}
            row["same_npu_grouped_vs_gather"] = report_pair(captured["npu_grouped"], captured["npu_gather"])
            row["original_fp32_comparison_replay"] = {access:comparison(captured["npu_"+access]["output"], captured["cpu_formula_fp32"]["output"], atol=2e-6, rtol=2e-6) for access in models}
            row["npu_output_hash_matches_prior"] = {access:tensor_record(captured["npu_"+access]["output"])["sha256"] == prior["paths"][access]["output_identity"]["sha256"] for access in models}
            # Fixed CPU64-root boundary values, rounded once to FP32, isolate
            # projection differences from upstream norm and gate/product error.
            z_common = captured["cpu_formula_fp64"]["norm_gate"].float()
            h_common = captured["cpu_formula_fp64"]["product"].float()
            conditional = {}
            with torch.no_grad(), precision_context(device, "fp32"):
                for access, current in models.items():
                    gate, up = current.gate(z_common.to(device), owners), current.up(z_common.to(device), owners)
                    h = torch.nn.functional.silu(gate)*up
                    down = current.down(h, owners)
                    conditional[access] = dict(gate=cpu(gate), up=cpu(up), product=cpu(h), down=cpu(down),
                        down_from_common_product=cpu(current.down(h_common.to(device), owners)))
                torch.npu.synchronize(device)
            row["conditional_common_inputs"] = dict(definition="fixed norm/product from independent FP64 root rounded to FP32; these are local projection probes, not whole-path qualification",
                grouped_vs_gather=report_pair(conditional["grouped"], conditional["gather"]),
                down_against_fp64_common_input={access:detailed_compare(torch, values["down_from_common_product"],
                    literal_projection(torch, h_common.double(), tensors64, host_groups, ids, "down")) for access,values in conditional.items()})
            local = {}
            for label, operation in (("production_rsqrt", lambda x:x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)),
                                     ("literal_sqrt_div", lambda x:x/torch.sqrt((x*x).mean(-1,keepdim=True)+1e-5))):
                x = original_x.to(device).requires_grad_()
                value = operation(x)
                vjp = torch.autograd.grad(value, x, root.to(device))[0]
                local[label] = dict(output=cpu(value), input_vjp=cpu(vjp))
            x64 = original_x.double().requires_grad_()
            norm64 = x64/torch.sqrt((x64*x64).mean(-1,keepdim=True)+1e-5)
            local["norm_fp64"] = dict(output=cpu(norm64), input_vjp=cpu(torch.autograd.grad(norm64,x64,root.double())[0]))
            gate_common = captured["cpu_formula_fp64"]["gate"].float()
            up_common = captured["cpu_formula_fp64"]["up"].float()
            local_root = .1*torch.cos(torch.arange(gate_common.numel(),dtype=torch.float32)).reshape_as(gate_common)
            for label, silu in (("native_silu_product", torch.nn.functional.silu), ("sigmoid_product", lambda g:g*torch.sigmoid(g))):
                gate = gate_common.to(device).requires_grad_()
                up = up_common.to(device).requires_grad_()
                value = silu(gate)*up
                dg, du = torch.autograd.grad(value, (gate,up), local_root.to(device))
                local[label] = dict(output=cpu(value), gate_vjp=cpu(dg), up_vjp=cpu(du))
            gate64, up64 = gate_common.double().requires_grad_(), up_common.double().requires_grad_()
            value64 = gate64*torch.sigmoid(gate64)*up64
            dg64, du64 = torch.autograd.grad(value64, (gate64,up64), local_root.double())
            local["silu_product_fp64"] = dict(output=cpu(value64), gate_vjp=cpu(dg64), up_vjp=cpu(du64))
            row["local_norm"] = dict(production_vs_literal=report_pair(local["production_rsqrt"],local["literal_sqrt_div"]),
                production_vs_fp64=report_pair(local["production_rsqrt"],local["norm_fp64"]),
                literal_vs_fp64=report_pair(local["literal_sqrt_div"],local["norm_fp64"]))
            row["local_silu_product"] = dict(definition="same FP32 gate/up values, obtained by rounding FP64-root intermediates once",
                native_vs_sigmoid=report_pair(local["native_silu_product"],local["sigmoid_product"]),
                native_vs_fp64=report_pair(local["native_silu_product"],local["silu_product_fp64"]),
                sigmoid_vs_fp64=report_pair(local["sigmoid_product"],local["silu_product_fp64"]))
            payload[sharing] = dict(input=original_x, root=root, stages=captured, conditional=conditional,
                                    common_norm=z_common, common_product=h_common, local=local)
            atomic(args.output, record)
            del model, models, tensors32, tensors64, initial, captured, conditional, local
        tensor_bytes = sum(t.numel()*t.element_size() for case in payload.values() for branch in (case["stages"],case["conditional"],case["local"]) for view in branch.values() for t in view.values())
        if tensor_bytes >= 1024**3:
            raise AssertionError("bounded stage payload exceeded1GiB")
        payload_path = args.output.with_suffix(".tensors.pt")
        descriptor, stage = tempfile.mkstemp(prefix=payload_path.name+".", dir=payload_path.parent)
        with os.fdopen(descriptor,"wb") as stream:
            torch.save(payload,stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage,payload_path)
        directory = os.open(payload_path.parent,os.O_DIRECTORY)
        os.fsync(directory)
        os.close(directory)
        record.update(state="completed_bounded_diagnosis", retained_tensors=file_identity(payload_path),
                      stage_tensor_bytes=tensor_bytes, retained_original_qualification_failures=original["failures"])
        if record["retained_tensors"]["bytes"] >= 1024**3:
            raise AssertionError("serialized stage payload exceeded1GiB")
        atomic(args.output,record)
        print(json.dumps(dict(state=record["state"],output=str(args.output),payload_bytes=record["retained_tensors"]["bytes"])))
        return 0
    except Exception as error:
        record.update(state="failed",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        atomic(args.output,record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
