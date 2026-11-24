"""Compare completed positive-floor and weak-log arms on the same device and prefix."""
import argparse
import json
from pathlib import Path

import torch

from pilotlab.io import atomic_json, sha256


def gradient_difference(old, new):
    if old.keys() != new.keys():
        raise ValueError("the policies have different sets of connected parameter gradients")
    squared_reference = squared_change = maximum = 0.0
    for name, value in old.items():
        other = new[name]
        if value.shape != other.shape or not (value.isfinite().all() and other.isfinite().all()):
            raise ValueError(f"invalid gradient pair for {name}")
        reference = value.double()
        difference = other.double() - reference
        squared_reference += reference.square().sum().item()
        squared_change += difference.square().sum().item()
        maximum = max(maximum, difference.abs().max().item())
    return {
        "connected_parameter_tensors": len(old),
        "l2": squared_change ** 0.5,
        "relative_l2_to_floor": (squared_change / squared_reference) ** 0.5 if squared_reference else None,
        "max_abs": maximum,
    }


def compare_pair(plan_path):
    plan = json.loads(plan_path.read_bytes())
    plan_hash = sha256(plan_path)
    arms = {}
    for policy in ("positive-floor-v1", "weak-log-v1"):
        result = json.loads((plan_path.parent / policy / "result.json").read_bytes())
        if (result["status"] != "completed" or result["policy"] != policy
                or result["plan_sha256"] != plan_hash or result["device"].split(":")[0] != plan["device"]):
            raise ValueError(f"invalid result identity for {plan_path} {policy}")
        arms[policy] = result
    floor, weak = arms.values()
    gradient_point = floor.get("gradient_point", "before_clip")
    if weak.get("gradient_point", "before_clip") != gradient_point:
        raise ValueError(f"gradient measurement points differ in {plan_path}")
    for key in ("device", "physical_card", "cpu_affinity", "torch", "threads"):
        if floor[key] != weak[key]:
            raise ValueError(f"mismatched {key} in {plan_path}")
    expected = [("failure", 0)]
    expected.extend(("train", i) for i in range(plan["train_batches"]))
    expected.extend(("validation", i) for i in range(plan["validation_batches"]))
    pairs = []
    for position, old, new in zip(expected, floor["rows"], weak["rows"], strict=True):
        if (any((row["case"], row["index"]) != position for row in (old, new))
                or old["input_sha256"] != new["input_sha256"] or old["tokens"] != new["tokens"]):
            raise ValueError(f"unmatched batch {position} in {plan_path}")
        a, b = [torch.load(row["outputs"], weights_only=True, map_location="cpu") for row in (old, new)]
        if (a["logits"].shape != b["logits"].shape or a["nll"].numel() != old["tokens"]
                or b["nll"].numel() != new["tokens"]):
            raise ValueError(f"output shape mismatch at {position}")
        delta = b["nll"].double() - a["nll"].double()
        if not delta.isfinite().all():
            raise ValueError(f"nonfinite token loss difference at {position}")
        observations = {}
        for policy, row in zip(arms, (old, new)):
            observations[policy] = {k: row[k] for k in (
                "loss", "timed_seconds", "inside_observer_seconds", "strengths", "shannon_forward_calls") if k in row}
            if position[0] != "validation":
                observations[policy].update(grad_norm=row["grad_norm"], backward_divisor=row["backward_divisor"])
        pair = {
            "case": position[0], "index": position[1], "input_sha256": old["input_sha256"],
            "tokens": old["tokens"], "arms": observations,
            "mean_token_nll_delta": delta.mean().item(),
            "max_abs_token_nll_delta": delta.abs().max().item(),
            "max_abs_logit_delta": (b["logits"] - a["logits"]).abs().max().item(),
            "top1_changed": (a["argmax"] != b["argmax"]).sum().item(),
        }
        if position[0] != "validation":
            pair["gradient_" + gradient_point] = gradient_difference(a["grads"], b["grads"])
        pairs.append(pair)
    return {
        "plan_sha256": plan_hash, "source_identity": plan["source_identity"], "driver": plan["driver"],
        "old_result": plan["old_result"], "device": floor["device"], "physical_card": floor["physical_card"],
        "cpu_affinity": floor["cpu_affinity"], "threads": floor["threads"], "torch": floor["torch"],
        "order": plan["order"],
        "peak_bytes": {p: {k: v for k, v in a["memory"].items() if "peak" in k} for p, a in arms.items()},
        "pairs": pairs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plans", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    atomic_json(args.output, {str(p): compare_pair(p) for p in args.plans})
