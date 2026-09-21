"""Explicit LH post-selection activation/normalization backbones."""
import torch

PROFILES = {f"lh-{act}-{norm}-v1": (act, norm)
            for act in ("relu", "silu", "identity") for norm in ("identity", "rms", "layer")}


def initialize(weights):
    _, norm = PROFILES[weights.full_kind]
    if norm != "identity":
        weights.extra["lh_norm_weight"] = torch.nn.Parameter(torch.ones_like(weights.bias))
    if norm == "layer":
        weights.extra["lh_norm_bias"] = torch.nn.Parameter(torch.zeros_like(weights.bias))


def validate(weights):
    _, norm = PROFILES[weights.full_kind]
    names = [] if norm == "identity" else ["lh_norm_weight"]
    if norm == "layer":
        names.append("lh_norm_bias")
    for name in names:
        p = weights.extra.get(name)
        if (p is None or p.shape != weights.bias.shape or p.dtype != weights.bias.dtype
                or p.device != weights.bias.device or not torch.isfinite(p).all()):
            raise ValueError("invalid LH Full normalization parameter")


def fresh(weights, comparison):
    act, norm = PROFILES[weights.full_kind]
    value = comparison.relu() if act == "relu" else torch.nn.functional.silu(comparison) if act == "silu" else comparison
    shape = (len(weights.bias),)
    if norm == "rms":
        return torch.nn.functional.rms_norm(value, shape, weights.extra["lh_norm_weight"], eps=1e-7)
    if norm == "layer":
        return torch.nn.functional.layer_norm(value, shape, weights.extra["lh_norm_weight"],
                                              weights.extra["lh_norm_bias"], eps=1e-5)
    return value
