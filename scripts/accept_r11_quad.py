"""Qualify the released foundation on the bound r11 Exchange/K A3 failure and its NPU continuation."""
import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

from accept_r09_shannon import owners, update


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS = ROOT / "docs/evidence/pilot-r3-r11-quad-reception-vjp.json"
ROUND = ROOT / ".local/pilot-r3/campaign-r11"


def read(path):
    return json.loads(Path(path).read_text())


def checked(reference):
    from pilotlab.io import sha256
    path = ROOT / reference["path"]
    if sha256(path) != reference["sha256"]:
        raise ValueError(f"qualification input changed: {path}")
    return path


def capture_update(model, opt, device, task, cap, control):
    import torch
    import torch.nn.functional as F
    from pilotlab import worker3 as w
    from pilotlab.worker import compare_tree, tree_map
    from tide.precision import precision_context

    model.load_state_dict(cap["model"])
    opt.load_state_dict(cap["optimizer"])
    state = tree_map(cap["private_state"], lambda x: x.to(device))
    for group in opt.param_groups:
        group["lr"] = w.learning_rate(task, cap["completed_tokens"] + cap["targets"].numel())
    model.train()
    opt.zero_grad(set_to_none=True)
    with precision_context(device, task["precision"]):
        result = model(cap["inputs"].to(device), state, record_work=False)
        loss = F.cross_entropy(result.logits.float().flatten(0, 1), cap["targets"].to(device).flatten())
        aux = getattr(result, "auxiliary_loss", None)
        objective = loss if aux is None else loss + task["auxiliary_loss_weight"] * aux
    objective.backward()
    gradients = w.cpu_snapshot({name: p.grad for name, p in model.named_parameters()})
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), task["grad_clip"], error_if_nonfinite=True, foreach=False)
    opt.step()
    w.synchronize(device)
    actual = dict(parameter_gradients_before_clip=gradients, model_after_step=w.cpu_snapshot(model.state_dict()),
                  optimizer_after_step=w.cpu_snapshot(opt.state_dict()), private_state=w.cpu_snapshot(result.state))
    reference = torch.load(checked(control["scaled_control_raw_state"]), map_location="cpu", weights_only=False)
    ref_report = read(checked(control["unchanged_graph_scaled_control_report"]))
    comparisons = {key: compare_tree(reference[key], value) for key, value in actual.items()}
    logits_equal = w.digest_tree(result.logits) == ref_report["logits_sha256"]
    same_roots = sum(g is not None for g in gradients.values()) == ref_report["gradients"]
    # Cross-entropy reduces over the batch. Identical logits can differ by a
    # final FP32 rounding between NPU reductions. Check both against FP64.
    logits = result.logits.detach().float().flatten(0, 1)
    targets = cap["targets"].flatten()
    loss_fp64 = float(F.cross_entropy(logits.cpu().double(), targets))
    target_device = targets.to(device)
    repeated = [float(F.cross_entropy(logits, target_device)) for _ in range(8)]
    scalar_losses = [float(loss.detach()), ref_report["loss"], *repeated]
    close_loss = all(math.isclose(value, loss_fp64, abs_tol=1e-5, rel_tol=1e-4) for value in scalar_losses)
    passed = (all(c["within_tolerance"] for c in comparisons.values()) and same_roots and logits_equal
              and close_loss
              and math.isclose(float(norm), ref_report["grad_norm"], rel_tol=1e-4, abs_tol=1e-5))
    return dict(passed=passed, comparison=comparisons, logits_bitwise_equal=logits_equal,
                gradient_roots_equal=same_roots, loss=float(loss.detach()), grad_norm=float(norm),
                loss_fp64=loss_fp64, scaled_control_loss=ref_report["loss"], repeated_loss_same_logits=repeated,
                loss_bitwise_equal=float(loss.detach()) == ref_report["loss"], all_losses_within_fp32_tolerance=close_loss)


def replay(args, e, worker_source, data, capture_check, suffix, *, model_config=None):
    import torch
    import tide
    from pilotlab import worker3 as w
    from pilotlab.worker import compare_tree
    from pilotlab.data3 import DocumentStore3

    old = read(checked(e["task"]))
    result = read(checked(e["worker_result"]))
    cp = result["checkpoint"]
    if cp != e["formal_checkpoint"] or e["original_failure_capture"] != result["failure_capture"]:
        raise ValueError("fault inputs disagree with the original result")
    checked(cp)
    cap = torch.load(checked(result["failure_capture"]), map_location="cpu", weights_only=False)
    fault = e["stage"]["failed_update"]
    if cap["task"] != old or cap["completed_steps"] + 1 != fault or not cap["update_in_progress"]:
        raise ValueError("capture lost its fault step or task binding")
    base = args.foundation_src.resolve() / "tide"
    if Path(tide.__file__).resolve().parent != base or Path(w.__file__).resolve().parent != worker_source / "pilotlab":
        raise ValueError("training did not import the two frozen trees")
    files = {str(p.relative_to(base)): w.sha256(p) for p in sorted(base.rglob("*.py"))}
    if w.canonical_digest(files) != args.source_digest:
        raise ValueError("foundation differs from the implementation commit")
    task = copy.deepcopy(old)
    task.pop("stage_a_migration", None)
    task["id"] += suffix
    if model_config:
        task["model"]["config"].update(model_config)
    original_files = old["source_identity"]["source_files"]
    task["source_identity"]["source_files"] = {
        p: h for p, h in original_files.items() if Path(p).is_relative_to(worker_source)
    } | {str(base / p): h for p, h in files.items()}
    w.validate_source_identity(task, data)
    device = w.initialize(args.device, task)
    model, opt = owners(task, DocumentStore3(data), device)
    record = dict(pid=os.getpid(), device=str(device), physical_device=os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
                  files_digest=args.source_digest, actual_module=str(tide.__file__), fault_step=fault)
    save_step, end_step = fault + 4, fault + 5
    if args.phase == "capture":
        record.update(capture_check(model, opt, device, task, cap))
    elif args.phase == "direct":
        saved, origin = w.load_checkpoint(cp["path"], old, device, cp["sha256"])
        if origin["fallback_used"] or (saved["steps"], saved["state_sha256"]) != (cp["steps"], cp["state_sha256"]):
            raise ValueError("official checkpoint failed exact load")
        state, stream = w.restore_owners(saved, model, opt, DocumentStore3(data), old, device)
        tokens = saved["tokens"]
        record.update(official_checkpoint=cp, original_state_loaded_exact=True, original_error=result["error"], updates=[])
        for step in range(saved["steps"] + 1, end_step + 1):
            batch = stream.next()
            x, y, reset = batch
            if step == fault:
                if not torch.equal(x, cap["inputs"]) or not torch.equal(y, cap["targets"]):
                    raise ValueError("formal fault batch differs on replay")
                record["fault_batch_matched"] = True
            if step == end_step:
                state, trees, info = update(model, opt, state, batch, task, device, tokens)
                w.atomic_torch_save(args.output / "expected.pt", trees)
            else:
                value = w.train_update(model, opt, x, y, None if reset else state, device, task, tokens)
                w.synchronize(device)
                state = value["state"]
                info = {k: value[k] for k in ("loss", "objective", "grad_norm", "lr")}
                if not all(math.isfinite(v) for v in info.values()):
                    raise ArithmeticError(f"nonfinite update at {step}")
            tokens += y.numel()
            record["updates"].append(dict(info, step=step, tokens=tokens))
            if step == save_step:
                payload = w.checkpoint_payload(model, opt, state, stream, task, device, tokens, step, saved["initialization"])
                record["new_checkpoint"] = w.save_checkpoint(args.output, payload)
        record.update(stream_sha256=w.digest_tree(stream.state_dict()), rng_sha256=w.digest_tree(w.rng_state(device)),
                      passed=record["fault_batch_matched"])
    else:
        direct = read(args.output / f"{device.type}-direct.json")
        ref = direct["new_checkpoint"]
        loaded, origin = w.load_checkpoint(ref["path"], task, device, ref["sha256"])
        if (direct["pid"] == os.getpid() or origin["fallback_used"] or loaded["steps"] != save_step
                or loaded["state_sha256"] != ref["state_sha256"]):
            raise ValueError("fresh owner did not load the exact post-fault state")
        state, stream = w.restore_owners(loaded, model, opt, DocumentStore3(data), task, device)
        _, actual, info = update(model, opt, state, stream.next(), task, device, loaded["tokens"])
        expected = torch.load(args.output / "expected.pt", map_location="cpu", weights_only=False)
        comparisons = {k: compare_tree(v, actual[k]) for k, v in expected.items()}
        prior = direct["updates"][-1]
        scalars = {k: abs(info[k] - prior[k]) for k in ("loss", "objective", "grad_norm")}
        discrete = (all(info[k] == prior[k] for k in ("input_sha256", "reset", "counts", "gradient_tensors", "lr"))
                    and w.digest_tree(stream.state_dict()) == direct["stream_sha256"]
                    and w.digest_tree(w.rng_state(device)) == direct["rng_sha256"])
        record.update(step=end_step, loaded_state_exact=True, checkpoint_sha256=origin["sha256"], discrete_exact=discrete,
                      scalar_abs_diffs=scalars, comparison=comparisons, passed=discrete and all(
                          x["within_tolerance"] for x in comparisons.values()) and all(
                          math.isclose(info[k], prior[k], abs_tol=1e-5, rel_tol=1e-4) for k in scalars))
    w.atomic_json(args.output / f"{device.type}-{args.phase}.json", record)
    if not record["passed"]:
        raise ArithmeticError(f"{args.phase} outside existing FP32 acceptance")


def qualify(args):
    e = read(DIAGNOSIS)
    def check(model, opt, device, task, cap):
        return capture_update(model, opt, device, task, cap, e["prototype"]["single_step"][device.type])
    data = Path(read(ROUND / "campaign-plan.json")["config"]["data"])
    replay(args, e, ROUND / "source", data, check, "-quad-qualification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("capture", "direct", "restored"), required=True)
    parser.add_argument("--device", choices=("cpu", "npu:0"), required=True)
    parser.add_argument("--foundation-src", type=Path, required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(ROUND / "source"), str(args.foundation_src.resolve())]
    qualify(args)
