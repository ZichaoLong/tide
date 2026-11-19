"""Bounded repeated-owner versus restored-owner backward/AdamW diagnosis.

Four forwards/backwards use unchanged parameters, private state and input in
the order original, restored, original, restored. Then each owner performs one
actual AdamW update using its second gradient. Scalar FP64 AdamW predictions
diagnose selected differing coordinates; they do not replace device execution.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import torch

from tide import ModelConfig, TideModel, detach_state, load_checkpoint, save_checkpoint
from tide.precision import precision_context, precision_policy
from scripts.job import atomic
from scripts.qualify_precision import comparison, cpu_tree, fields_of, file_sha256, fingerprint


TOLERANCE = {"atol": 1e-6, "rtol": 2e-3}


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def adamw_scalar(parameter, gradient, first, second, step, *, lr, betas, eps, weight_decay):
    """Ordinary AdamW in scalar Python/FP64 for diagnosis of one coordinate."""
    b1, b2 = betas
    first = b1*first + (1-b1)*gradient
    second = b2*second + (1-b2)*gradient*gradient
    count = step + 1
    denominator = math.sqrt(second) / math.sqrt(1-b2**count) + eps
    updated = parameter*(1-lr*weight_decay) - lr/(1-b1**count)*first/denominator
    return {"parameter": updated, "first_moment": first, "second_moment": second, "denominator": denominator}


def backward_snapshot(model, state, value, precision):
    model.zero_grad(set_to_none=True)
    x = value.detach().clone().requires_grad_()
    with precision_context(value.device, precision):
        result = model(x, state)
        loss = result.readout().float().square().mean()
    loss.backward()
    synchronize(value.device)
    return cpu_tree({"result": fields_of(result), "input_gradient": x.grad,
                     "gradients": {name: p.grad for name, p in model.named_parameters()}, "loss": loss})


def compare_backwards(candidate, reference):
    return {"forward_state": comparison(candidate["result"], reference["result"], TOLERANCE),
            "input_gradient": comparison(candidate["input_gradient"], reference["input_gradient"], {"atol": 1e-6, "rtol": 1e-2}),
            "parameter_gradients": comparison(candidate["gradients"], reference["gradients"], TOLERANCE)}


def layouts(model, device):
    vendor = None
    if device.type == "npu":
        import torch_npu
        vendor = getattr(torch_npu, "get_npu_format", None)
    result = {}
    for name, parameter in model.named_parameters():
        if name == "inputs.D.weight" or name.endswith("regional.0.second.bias") or name in {
                "regional.0.second.bias", "regional.0.v.weight", "describe.encode.weight", "memories.K.kv.weight", "memories.S.input.weight"}:
            result[name] = {"shape": list(parameter.shape), "stride": list(parameter.stride()), "dtype": str(parameter.dtype),
                            "contiguous": parameter.is_contiguous(), "storage_offset": parameter.storage_offset()}
            if vendor:
                result[name]["npu_format"] = int(vendor(parameter))
    return result


def run_case(cfg, name, *, device, precision, batch, length, steps, checkpoint):
    seed = 31415 + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    torch.manual_seed(seed)
    model = TideModel(cfg)
    x_cpu = torch.randn(batch, length, cfg.input_dim or cfg.d)
    record = {"case": name, "config": cfg.to_dict(), "seed": seed,
              "initial_parameter_sha256": fingerprint(model.state_dict()), "input_sha256": fingerprint(x_cpu)}
    model, value = model.to(device), x_cpu.to(device)
    # Match the qualification's initial grad-enabled forward/backward and
    # inference calls before constructing the optimizer and continuing updates.
    backward_snapshot(model, None, value, precision)
    with torch.no_grad(), precision_context(device, precision):
        model(value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)
    state, losses = None, []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with precision_context(device, precision):
            result = model(value, state)
            loss = result.readout().float().square().mean()
        loss.backward()
        optimizer.step()
        state = detach_state(result.state)
        losses.append(float(loss.detach()))
    synchronize(device)
    save_checkpoint(checkpoint, model, state=state, optimizer=optimizer)
    restored, saved, slots = load_checkpoint(checkpoint, device=device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4, foreach=True)
    restored_optimizer.load_state_dict(slots)
    record["warmup_update_losses"] = losses
    record["checkpoint_sha256"] = file_sha256(checkpoint)
    record["saved_data"] = comparison({"parameters": restored.state_dict(), "state": saved, "optimizer": restored_optimizer.state_dict()},
                                       {"parameters": model.state_dict(), "state": state, "optimizer": optimizer.state_dict()}, TOLERANCE)
    record["layouts"] = {"original": layouts(model, device), "restored": layouts(restored, device)}
    snapshots = {}
    for label, subject, carry in (("original1", model, state), ("restored1", restored, saved),
                                   ("original2", model, state), ("restored2", restored, saved)):
        snapshots[label] = backward_snapshot(subject, carry, value, precision)
    record["backwards"] = {
        "same_original_owner": compare_backwards(snapshots["original2"], snapshots["original1"]),
        "same_restored_owner": compare_backwards(snapshots["restored2"], snapshots["restored1"]),
        "first_cross_owner": compare_backwards(snapshots["restored1"], snapshots["original1"]),
        "second_cross_owner": compare_backwards(snapshots["restored2"], snapshots["original2"]),
    }
    before = cpu_tree(model.state_dict())
    moments = {name: cpu_tree(optimizer.state[p]) for name, p in model.named_parameters() if p in optimizer.state}
    group = optimizer.param_groups[0]
    hyper = {name: group[name] for name in ("lr", "betas", "eps", "weight_decay")}
    optimizer.step()
    synchronize(device)
    restored_optimizer.step()
    synchronize(device)
    a, b = cpu_tree(model.state_dict()), cpu_tree(restored.state_dict())
    record["actual_updated_parameters"] = comparison(b, a, TOLERANCE)
    record["actual_updated_optimizer"] = comparison(restored_optimizer.state_dict(), optimizer.state_dict(), TOLERANCE)
    candidates = []
    for key in moments:
        if a[key].numel() and snapshots["original2"]["gradients"][key] is not None and snapshots["restored2"]["gradients"][key] is not None:
            delta = (a[key].double()-b[key].double()).abs().flatten()
            scaled = delta/(TOLERANCE["atol"] + TOLERANCE["rtol"]*a[key].double().abs().flatten())
            index = int(scaled.argmax())
            if float(delta[index]):
                candidates.append((float(scaled[index]), key, index))
    witnesses = []
    for scaled, key, index in sorted(candidates, reverse=True)[:8]:
        old = float(before[key].flatten()[index])
        m = float(moments[key]["exp_avg"].flatten()[index])
        v = float(moments[key]["exp_avg_sq"].flatten()[index])
        count = float(moments[key]["step"])
        g1 = float(snapshots["original2"]["gradients"][key].flatten()[index])
        g2 = float(snapshots["restored2"]["gradients"][key].flatten()[index])
        predicted1 = adamw_scalar(old, g1, m, v, count, **hyper)
        predicted2 = adamw_scalar(old, g2, m, v, count, **hyper)
        witnesses.append({"parameter": key, "flat_index": index, "scaled_update_error": scaled,
                          "original_gradient": g1, "restored_gradient": g2, "gradient_difference": g2-g1,
                          "both_gradient_magnitudes_at_most_optimizer_eps": max(abs(g1), abs(g2)) <= hyper["eps"],
                          "parameter_before": old, "first_moment_before": m, "second_moment_before": v, "completed_steps": count,
                          "predicted_original": predicted1, "predicted_restored": predicted2,
                          "predicted_parameter_difference": predicted2["parameter"]-predicted1["parameter"],
                          "actual_original_parameter": float(a[key].flatten()[index]), "actual_restored_parameter": float(b[key].flatten()[index]),
                          "actual_parameter_difference": float(b[key].double().flatten()[index]-a[key].double().flatten()[index])})
    record["adamw"] = hyper
    record["coordinate_witnesses"] = witnesses
    numerical = [record["saved_data"], record["actual_updated_parameters"], record["actual_updated_optimizer"]]
    numerical += [entry for pair in record["backwards"].values() for entry in pair.values()]
    bad_data = not record["saved_data"]["exact"]
    metadata = any(row["metadata_failures"] for row in numerical)
    nonfinite = any(row["nonfinite"] for row in numerical)
    forward_exact = all(pair["forward_state"]["exact"] for pair in record["backwards"].values())
    record["state"] = ("nonfinite" if nonfinite else "structural_failed" if bad_data or metadata else
                       "finite_deviation" if any(row["threshold_failures"] for row in numerical) else "within_threshold")
    record["all_forward_state_repeats_exact"] = forward_exact
    record["scope"] = "same source/device, unchanged weights/state/input for four backward calls; no optimizer update between them; one actual update per owner afterwards"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--configs", nargs="+", default=["fc-k", "fc-s"])
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.batch, args.length, args.steps, args.threads) < 1 or args.width < 0:
        parser.error("new output, positive counts, and nonnegative width are required")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    torch.set_num_threads(args.threads)
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report = {"source": source, "policy": precision_policy(device, args.precision), "torch": str(torch.__version__),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "physical_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "intrusive_diagnostic": True, "thresholds": TOLERANCE, "rows": []}
    for name in args.configs:
        path = Path(name) if name.endswith(".json") else Path("configs") / (name + ".json")
        row = {"case": name, "state": "running"}
        report["rows"].append(row)
        atomic(args.output, report)
        try:
            config = json.loads(path.read_text())
            if args.width:
                config["d"] = args.width
            with tempfile.TemporaryDirectory(prefix="restore-witness-", dir=args.output.parent) as temporary:
                row.update(run_case(ModelConfig.from_dict(config), path.stem, device=device, precision=args.precision,
                                    batch=args.batch, length=args.length, steps=args.steps, checkpoint=Path(temporary) / "state.pt"))
        except Exception as error:
            import traceback
            traceback.print_exc()
            row.update(state="execution_failed", error=repr(error))
        atomic(args.output, report)
        print(json.dumps({"case": name, "state": row["state"]}), flush=True)
    return int(any(row["state"] in {"execution_failed", "structural_failed", "nonfinite"} for row in report["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
