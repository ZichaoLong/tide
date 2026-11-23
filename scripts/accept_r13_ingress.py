"""Validate timed ingress on the r13 failing batch and checkpoint with frozen source trees."""
import argparse
import math
from pathlib import Path
import sys

from accept_r11_quad import ROOT, checked, read, replay


def load_captured_state(model, cap, device, dtype=None):
    from pilotlab.worker import tree_map
    model.load_state_dict(cap["model"])
    if dtype is not None:
        model.to(dtype=dtype)
    model.train()
    return tree_map(cap["private_state"], lambda t: t.to(
        device=device, dtype=dtype if t.is_floating_point() else None))


def fp64_reference(output, old, cap, data):
    import torch
    from pilotlab import worker3 as w
    from pilotlab.data3 import DocumentStore3
    from pilotlab.models3 import build_model

    w.validate_source_identity(old, data)
    device = w.initialize("cpu", old)
    model = build_model(old["model"], DocumentStore3(data).vocab_size, old["init_recipe"])
    state = load_captured_state(model, cap, device, torch.float64)
    logits = model(cap["inputs"], state, record_work=False).logits
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), cap["targets"].flatten())
    loss.backward()
    gradients = {name:p.grad.detach() for name,p in model.named_parameters() if p.grad is not None}
    if not all(bool(g.isfinite().all()) for g in gradients.values()):
        raise ArithmeticError("nonfinite FP64 reference gradient")
    path = output / "fp64-gradients.pt"
    w.atomic_torch_save(path, gradients)
    w.atomic_json(output / "cpu-reference.json", dict(
        gradient_file={"path":str(path),"sha256":w.sha256(path)}, roots=len(gradients),
        objective=float(loss.detach()), capture_task=cap["task"]["id"]))


def capture_update(model, opt, device, task, cap, control, output):
    import torch
    from pilotlab import worker3 as w
    from pilotlab.worker import compare_tree
    from tide.precision import precision_context
    from tide import observe_shannon_support
    from dataclasses import asdict

    state = load_captured_state(model, cap, device)
    opt.load_state_dict(cap["optimizer"])
    lr = w.learning_rate(task, cap["completed_tokens"] + cap["targets"].numel())
    for group in opt.param_groups:
        group["lr"] = lr
    opt.zero_grad(set_to_none=True)
    with observe_shannon_support() as support:
        with precision_context(device, task["precision"]):
            out = model(cap["inputs"].to(device), state, record_work=False)
            loss = torch.nn.functional.cross_entropy(out.logits.float().flatten(0, 1), cap["targets"].to(device).flatten())
            objective = loss if getattr(out, "auxiliary_loss", None) is None else loss + task["auxiliary_loss_weight"]*out.auxiliary_loss
        objective.backward()
    gradients = {n:p.grad.detach().cpu().double() for n,p in model.named_parameters() if p.grad is not None}
    ref = read(output / "cpu-reference.json")
    if ref["capture_task"] != cap["task"]["id"]:
        raise ValueError("FP64 reference belongs to another task")
    expected = torch.load(checked(ref["gradient_file"]), map_location="cpu", weights_only=True)
    gradient_check = compare_tree(expected, gradients)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), task["grad_clip"], error_if_nonfinite=True, foreach=False))
    opt.step()
    w.synchronize(device)
    finite = all(bool(t.isfinite().all()) for _,t in w.leaves((model.state_dict(), opt.state_dict()))
                 if isinstance(t, torch.Tensor) and t.is_floating_point())
    logits = w.digest_tree(out.logits)
    state_digest = w.digest_tree(out.state)
    counts = asdict(support)
    passed = (set(expected) == set(gradients) and gradient_check["within_tolerance"] and finite
              and math.isfinite(norm) and logits == control["baseline_logits_sha256"]
              and state_digest == control["baseline_nextstate_sha256"] and not any(counts.values())
              and math.isclose(float(loss.detach()), ref["objective"], abs_tol=1e-5, rel_tol=1e-4))
    return dict(passed=passed, loss=float(loss.detach()), grad_norm=norm, roots=len(gradients),
                gradient_fp64=gradient_check, logits_sha256=logits, next_state_sha256=state_digest,
                logits_bitwise_equal=logits==control["baseline_logits_sha256"],
                private_state_bitwise_equal=state_digest==control["baseline_nextstate_sha256"],
                next_parameters_and_adamw_finite=finite, support=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("reference", "capture", "direct", "restored"))
    parser.add_argument("--device", required=True, choices=("cpu", "npu:0"))
    parser.add_argument("--foundation-src", type=Path, required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    e = read(ROOT / "docs/evidence/pilot-r3-r13-timed-ingress-vjp.json")
    run = ROOT / ".local/pilot-r3/campaign-r13"
    cfg = read(run / "campaign-plan.json")["config"]
    worker_source = Path(cfg["worker_source"])
    original_src = Path(cfg["worker_foundation"]) / "src"
    base = original_src if args.phase == "reference" else args.foundation_src.resolve()
    sys.path[:0] = [str(worker_source), str(base)]
    import tide
    import torch
    from pilotlab import worker3
    if (Path(tide.__file__).resolve().parent != base/"tide"
            or Path(worker3.__file__).resolve().parent != worker_source/"pilotlab"):
        raise ValueError("frozen source not loaded")
    meta = e["task"]
    old = read(checked(meta["definition"]))
    orig = read(checked(meta["result"]))
    if orig["failure_capture"]["sha256"] != meta["capture"]["sha256"]:
        raise ValueError("failure capture is not bound to the original result")
    if args.phase == "reference":
        cap = torch.load(checked(orig["failure_capture"]), map_location="cpu", weights_only=False)
        if cap["task"] != old or cap["resumable"] or cap["completed_steps"]+1 != meta["attempted_step"]:
            raise ValueError("wrong fault state")
        fp64_reference(args.output, old, cap, Path(cfg["data"]))
        return
    ck = orig["checkpoint"]
    if any(ck[k] != e["verified_checkpoint"][k] for k in ("path","steps","tokens","sha256","state_sha256")):
        raise ValueError("original checkpoint differs from inspected generation")
    evidence = dict(task=meta["definition"], worker_result=meta["result"], formal_checkpoint=ck,
                    original_failure_capture=orig["failure_capture"], stage={"failed_update":meta["attempted_step"]})
    def compare(model, opt, device, task, cap):
        return capture_update(model, opt, device, task, cap,
                              e["probes"]["runtime_only_clock_proof"][device.type], args.output)
    replay(args, evidence, worker_source, Path(cfg["data"]), compare, "-timed-ingress-qualification")


if __name__ == "__main__":
    main()
