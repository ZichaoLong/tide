"""Offline comparison of the legacy storage policy on the captured r09 batch.

The batch adapter implements only the two-interior-candidate, unit-total
face witnessed in this capture. The rest of the model uses its frozen source.
Extract the historical python/tidegraph package from its locked Git object
and pass its python directory with --legacy-python.
"""

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch.autograd.function import once_differentiable


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/pilot-r3-r09-shannon-underflow.json"


def reject(mask, message):
    if bool(mask.any()):
        raise ArithmeticError(message)


class BinaryFloor(torch.autograd.Function):
    """Witness face with T=1, two finite scores and both controls interior."""

    @staticmethod
    def forward(ctx, scores, total, policy):
        relative = scores - scores.amax(-1, keepdim=True)
        log_p = relative - torch.logsumexp(relative, -1, keepdim=True)
        stored, lost = policy.repair(log_p.exp(), 1 + log_p, torch.ones_like(log_p),
                                     scores.new_tensor(1), total, total, reject)
        if not bool(((total == 1) & (lost.sum(-1) == 1)).all()):
            raise ValueError("row outside the unit-total, one-lost-sender witness")
        ctx.save_for_backward(log_p)
        ctx.policy = policy
        return stored

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (log_p,) = ctx.saved_tensors
        product = ctx.policy.signed_exp_product
        difference = grad[:, 0] - grad[:, 1]
        # Both scores are free even when the strong control rounds to its cap.
        ds0 = product(log_p.sum(-1), difference)
        weak_first = log_p[:, 0] < log_p[:, 1]
        anchor = torch.where(weak_first, grad[:, 1], grad[:, 0])
        centered = torch.where(weak_first, difference, -difference)
        dtotal = anchor + product(log_p.amin(-1), centered)
        if not bool((ds0.isfinite() & dtotal.isfinite()).all()):
            raise ArithmeticError("binary witness VJP outside the finite range")
        return torch.stack((ds0, -ds0), -1), dtotal, None


def decimal_weak(row):
    with localcontext() as context:
        context.prec = 120
        a, b = map(Decimal.from_float, row["scores"])
        return 1 / (1 + abs(a - b).exp())


def row_comparison(rows, device, legacy, policy):
    def tensor(values):
        return torch.tensor(values, device=device, dtype=torch.float32)

    scores = tensor([r["scores"] for r in rows]).requires_grad_()
    caps = tensor([r["upper"] for r in rows])
    raw = tensor([r["fp32_allocation"] for r in rows])
    total = tensor([r["high_total"] for r in rows]).requires_grad_()
    weak = scores.argmin(-1, keepdim=True)
    root = torch.zeros_like(scores).scatter_(1, weak, 1e30)
    assert all(r["low_total"] == r["high_total"] == 1 and r["lower"] == [0, 0]
               and r["upper"] == [1, 1] for r in rows)
    options = dict(regularizer="ent", ent_alpha=1., support_policy=policy.FLOOR)
    old = legacy.allocate_batch(list(scores.unbind()), list(caps.unbind()),
                                list(total.unbind()), list(total.unbind()), **options)
    old_c = torch.stack([a.controls for a in old])
    old_ds = torch.autograd.grad(old_c, scores, root)[0]
    old_weak = old_c.gather(1, weak).squeeze(-1).tolist()
    log_p = scores.detach().log_softmax(-1)
    repaired, lost = policy.repair(raw, 1 + log_p, caps, tensor(1), total, total, reject)
    adapted = BinaryFloor.apply(scores, total, policy)
    assert torch.equal(repaired, adapted)
    ds, dt = torch.autograd.grad(adapted, (scores, total), root)
    ds_weak = ds.gather(1, weak).squeeze(-1).tolist()
    ds_strong = ds.gather(1, 1 - weak).squeeze(-1).tolist()
    delta = float(policy.smallest(scores).item())
    g = root.max().item()
    with localcontext() as context:
        context.prec = 120
        probabilities = [decimal_weak(r) for r in rows]
        cotangent = Decimal.from_float(g)
        slopes = [float(p * (1 - p) * cotangent) for p in probabilities]
        total_slopes = [float(p * cotangent) for p in probabilities]
        exact = [float(p) for p in probabilities]

    def error(computed, expected):
        return max(abs(a - b) / abs(b) for a, b in zip(computed, expected))

    # Existing positive subnormal, zero cap and endpoint cases have independent
    # checks. A whole slack row lost to rounding must exceed the mass budget.
    control = tensor([[0, 2 * delta, 1]])
    keep, mask = policy.repair(control, tensor([[-120, -102, 1]]),
                               tensor([[0, 1, 1]]), tensor(1), tensor([1]), tensor([1]), reject)
    endpoints = [legacy.allocate(tensor([-110, 0]), tensor(cc), bound, bound, **options).controls.tolist()
                 for cc, bound in (([0, 1], .5), ([1, 1], 0.), ([1, 1], 2.))]
    try:
        policy.repair(tensor([[0, 0]]), tensor([[-999, -999]]), tensor([[1, 1]]),
                      tensor(1), tensor([0]), tensor([2]), reject)
    except ArithmeticError as err:
        budget_refusal = str(err)
    else:
        raise AssertionError("all-underflow row passed")

    from tide.weighted_mean import stable_weighted_mean
    mass = tensor([delta]).requires_grad_()
    value = tensor([[.25]]).requires_grad_()
    mean = stable_weighted_mean(mass, value)
    dm, dv = torch.autograd.grad(mean.sum(), (mass, value))
    return {
        "rows": len(rows), "floor": delta, "floor_count": int(lost.sum()),
        "floored_coordinates_match_witness": lost.nonzero()[:, -1].tolist() == [r["index"][-1] for r in rows]
        and bool((lost.sum(-1) == 1).all()),
        "all_stored_weak_equal_floor": bool((repaired.gather(1, weak) == delta).all()),
        "unchanged_other_coordinates": torch.equal(raw[~lost], repaired[~lost]),
        "stored_to_decimal_weak_ratio_range": [delta / max(exact), delta / min(exact)],
        "absolute_error_range": [delta - max(exact), delta - min(exact)],
        "legacy_whole_allocator": {
            "floor_count": sum(int(a.support_repairs.sum()) for a in old),
            "weak_range": [min(old_weak), max(old_weak)],
            "zero_score_gradient_rows": int((old_ds == 0).all(-1).sum()),
        },
        "root_on_weak_coordinate": g,
        "max_relative_error_vs_decimal": {"weak_score": error(ds_weak, slopes),
                                           "strong_score": error(ds_strong, [-s for s in slopes]),
                                           "total": error(dt.tolist(), total_slopes)},
        "representative": {"index": rows[0]["index"], "scores": rows[0]["scores"],
                           "decimal_weak": exact[0], "stored_weak": delta,
                           "stored_to_decimal_ratio": delta / exact[0],
                           "old_allocator_values": old_c[0].tolist(), "old_allocator_score_vjp": old_ds[0].tolist(),
                           "binary_score_vjp": ds[0].tolist(), "decimal_weak_score_vjp": slopes[0]},
        "endpoint_controls_zero_cap_zero_total_all_caps": endpoints,
        "existing_subnormal_and_zero_cap_preserved": torch.equal(keep, control) and not bool(mask.any()),
        "all_underflow_refusal": budget_refusal,
        "frozen_reception_singleton": {"mass": mass.item(), "output": mean.item(),
                                        "mass_grad": dm.item(), "value_grad": dv.item()},
    }


def captured_batch(capture, device, alloc, policy):
    from pilotlab.models3 import build_model
    from pilotlab.worker import synchronize, tree_map
    from tide.precision import precision_context

    original = alloc._Constrained.apply
    calls, repaired_calls = 0, []

    def dispatch(scores, mu, lower, upper, low_total, high_total, kind):
        nonlocal calls
        if kind == "shannon":
            calls += 1
        args = mu, lower, upper, low_total, high_total, kind
        if kind != "shannon" or scores.shape[-1] != 2:
            return original(scores, *args)
        unit = (low_total == 1) & (high_total == 1)
        unit = unit & ((lower == 0) & (upper == 1)).all(-1, keepdim=True)
        gap = (scores[..., 0].detach() - scores[..., 1].detach()).abs()
        rows = unit.squeeze(-1) & (gap > 150 * math.log(2))
        if not bool(rows.any()):
            return original(scores, *args)
        with torch.no_grad():
            # In the capture the unconstrained total exceeds one. The current
            # contract assigns the total derivative to high_total on this face.
            c0 = alloc._response(scores, 0, lower, upper, kind)
            head, tail = alloc._compensated_sum(c0)
            diff, err = alloc._two_sum(head, -high_total)
            if not bool(((diff + (err + tail))[rows] > 0).all()):
                raise ValueError("witness requires the active upper total")
            repaired_calls.append({"call": calls, "indices": rows.nonzero().tolist()})
        # Keep the batch shape and original solver for every other row.
        ordinary = original(torch.where(rows[..., None], 0, scores), *args)
        result = ordinary.clone()
        target = high_total.expand(*scores.shape[:-1], 1).squeeze(-1)
        result[rows] = BinaryFloor.apply(scores[rows], target[rows], policy)
        return result

    task = capture["task"]
    model = build_model(task["model"], capture["model"]["embedding.weight"].shape[0], task["init_recipe"]).to(device)
    model.load_state_dict(capture["model"])
    model.train()
    state = tree_map(capture["private_state"], lambda t: t.to(device))
    with patch.object(alloc._Constrained, "apply", dispatch), precision_context(device, task["precision"]):
        output = model(capture["inputs"].to(device), state)
        loss = torch.nn.functional.cross_entropy(output.logits.float().flatten(0, 1),
                                                 capture["targets"].to(device).flatten())
        if getattr(output, "auxiliary_loss", None) is not None:
            loss = loss + task.get("auxiliary_loss_weight", 1.) * output.auxiliary_loss
        loss.backward()
        synchronize(device)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return {"objective": loss.item(), "finite_loss": bool(loss.isfinite()),
            "all_logits_finite": bool(output.logits.isfinite().all()),
            "gradient_tensor_count": len(grads), "all_grad_finite": all(bool(g.isfinite().all()) for g in grads),
            "shannon_calls": calls, "repaired_calls": repaired_calls, "optimizer_steps": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-python", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "npu:0"), required=True)
    args = parser.parse_args()
    record = json.loads(EVIDENCE.read_text())
    frozen = ROOT / ".local/pilot-r3/campaign-r08/frozen"
    sys.path[:0] = [str(args.legacy_python.resolve()), str(frozen / "foundation/src"), str(frozen / "source")]
    from pilotlab.worker3 import initialize
    import tide.allocation as alloc
    import tidegraph.weighted as legacy
    from tidegraph.weighted import support_policy

    ref = record["failure"]["capture"]
    capture_path = ROOT / ref["path"]
    if hashlib.sha256(capture_path.read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError("capture hash mismatch")
    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    device = initialize(args.device, dict(capture["task"], cpu_threads=4))
    rows = record["numeric_witness"]["all_failing_coordinates"][device.type]
    result = {"device": args.device, "physical_id": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "row_comparison": row_comparison(rows, device, legacy, support_policy),
              "captured_batch": captured_batch(capture, device, alloc, support_policy)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
