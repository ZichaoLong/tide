"""Replay the captured regional-prior VJP and checkpoint against a frozen foundation."""
import argparse
import math
from pathlib import Path
import sys

from accept_r11_quad import ROOT, checked, read, replay


def capture_gradients(model, task, capture, device, dtype):
    import torch
    from pilotlab.worker import tree_map

    model.to(dtype=dtype)
    model.load_state_dict(capture["model"])
    model.train()
    state = tree_map(capture["private_state"], lambda x: x.to(
        device=device, dtype=dtype if x.is_floating_point() else None))
    out = model(capture["inputs"].to(device), state)
    loss = torch.nn.functional.cross_entropy(out.logits.flatten(0, 1), capture["targets"].to(device).flatten())
    auxiliary = getattr(out, "auxiliary_loss", None)
    objective = loss if auxiliary is None else loss + task["auxiliary_loss_weight"] * auxiliary
    objective.backward()
    gradients = {n:p.grad.detach().cpu().double() for n,p in model.named_parameters() if p.grad is not None}
    return out, gradients, float(objective.detach())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("reference","capture","direct","restored"), required=True)
    p.add_argument("--device", choices=("cpu","npu:0"), required=True)
    p.add_argument("--foundation-src", type=Path, required=True)
    p.add_argument("--source-digest", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    args = p.parse_args()
    diag = read(ROOT/"docs/evidence/pilot-r3-r21-regional-key-prior-vjp.json")
    plan = read(ROOT/".local/pilot-r3/campaign-r21/campaign-plan.json")
    cfg = plan["config"]
    source, original = Path(cfg["worker_source"]), Path(cfg["worker_foundation"])/"src"
    data = Path(cfg["data"])
    base = original if args.phase == "reference" else args.foundation_src.resolve()
    sys.path[:0] = [str(source), str(base)]
    import torch
    from pilotlab import worker3 as w
    from pilotlab.prepare_resume3 import reference
    from pilotlab.worker import compare_tree
    from pilotlab.models3 import build_model

    result = read(checked(diag["result"]))
    old = reference(checked(diag["result"]).parent/"task.json")
    task = read(checked(old))
    args.reference_dir.mkdir(parents=True, exist_ok=True)
    raw = args.reference_dir/"reference-gradients.pt"
    receipt = args.reference_dir/"reference.json"
    if args.phase == "reference":
        if args.device != "cpu":
            raise ValueError("original FP64 reference runs on CPU")
        w.validate_source_identity(task, data)
        device = w.initialize("cpu", task)
        model = build_model(task["model"], read(data/"manifest.json")["vocab_size"], task["init_recipe"])
        cap = torch.load(checked(result["failure_capture"]), map_location="cpu", weights_only=False)
        output, gradients, objective = capture_gradients(model, task, cap, device, torch.float64)
        w.atomic_torch_save(raw, gradients)
        w.atomic_json(receipt, dict(capture=result["failure_capture"], gradients=reference(raw),
                                    objective=objective, logits_sha256=w.digest_tree(output.logits)))
        return

    def check(model, optimizer, device, task, cap):
        optimizer.load_state_dict(cap["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = w.learning_rate(task, cap["completed_tokens"] + cap["targets"].numel())
        reference_data = read(receipt)
        if reference_data["capture"] != result["failure_capture"]:
            raise ValueError("FP64 gradients belong to another capture")
        out, gradients, objective = capture_gradients(model, task, cap, device, torch.float32)
        expected = torch.load(checked(reference_data["gradients"]), weights_only=True, map_location="cpu")
        comparison = compare_tree(expected, gradients)
        prefix = device.type+"-original.json"
        control = read(checked(next(r for r in diag["artifacts"] if Path(r["path"]).name == prefix)))
        logits_equal = w.digest_tree(out.logits) == control["logits_hash"]
        state_equal = w.digest_tree(out.state) == control["private_state_hash"]
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),task["grad_clip"],error_if_nonfinite=True,foreach=False))
        optimizer.step()
        w.synchronize(device)
        finite = all(bool(t.isfinite().all()) for _,t in w.leaves((model.state_dict(),optimizer.state_dict()))
                     if isinstance(t,torch.Tensor) and t.is_floating_point())
        return dict(passed=finite and comparison["within_tolerance"] and logits_equal and state_equal
                    and math.isclose(objective, reference_data["objective"], abs_tol=1e-5, rel_tol=1e-4),
                    logits_bitwise_equal=logits_equal,private_state_bitwise_equal=state_equal,
                    gradient_fp64=comparison,grad_norm=norm,objective=objective,after_adamw_finite=finite)

    evidence=dict(task=old,worker_result=diag["result"],formal_checkpoint=result["checkpoint"],
                  original_failure_capture=result["failure_capture"],stage={"failed_update":diag["attempted_step"]})
    args.output.mkdir(parents=True, exist_ok=True)
    replay(args,evidence,source,data,check,"-regional-prior-qualification")


if __name__ == "__main__":
    main()
