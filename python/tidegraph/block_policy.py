"""Policy validation shared by Python clients, outside semantic identity."""
from .aggregate import SourceAggregate
from .full import ProjectionEmit


def validate(model, *, packed=True, full_autograd="replay", aggregate_autograd="replay"):
    if type(packed) is not bool:
        raise ValueError("packed must be bool")
    for label, policy, cls, field in (("Full", full_autograd, ProjectionEmit, "full_program"),
                                      ("Aggregate", aggregate_autograd, SourceAggregate, "aggregate_program")):
        if policy not in {"replay", "batched"}:
            raise ValueError(f"unknown {label} autograd policy")
        if policy == "batched":
            if not packed:
                raise ValueError(f"batched {label} autograd requires packed execution")
            programs = [getattr(w, field) for w in model.nodes]
            supported = (all(p.batched_autograd for p in programs) if label == "Aggregate"
                         else all(type(p) is cls for p in programs))
            if not supported:
                raise ValueError(f"{label} program has no batched autograd implementation")
