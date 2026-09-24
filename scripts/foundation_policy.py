"""Versioned benchmark policy resolution; never forward kernel fields to Options."""
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Schedule:
    workers: int = 1
    packed: bool = True
    prefill: bool = True
    full_autograd: str = "replay"
    aggregate_autograd: str = "replay"
    packed_sources: bool = False
    batch_next: bool = False
    parallel_regions: bool = False
    compact_events: bool = False
    defer_state_release: bool = False


@dataclass(frozen=True)
class Kernels:
    attention_packing: str = "exact"
    fiber_pooling: str = "event"
    fiber_cache: str = "cloned"
    attention_layout: str = "event"


SCHEDULE_FIELDS = set(Schedule.__dataclass_fields__)
KERNEL_FIELDS = set(Kernels.__dataclass_fields__)
FIELDS = SCHEDULE_FIELDS | KERNEL_FIELDS | {"projection_layout"}
ALGORITHMS = {"python-stream", "python-stream-packed", "python-frontier", "python-chain", "python-diamond",
              "python-settle", "python-layered", "python-layered-block", "python-chain-block",
              "native-stream", "native-frontier", "native-chain", "native-diamond", "native-settle",
              "native-settle-stream", "encoded-frontier", "encoded-stream"}


def resolve(config, variant, workers, overrides=None):
    base, preset = variant, "baseline"
    for suffix in ("-optimized", "-scalar", "-step", "-single"):
        if base.endswith(suffix):
            base, preset = base[:-len(suffix)], suffix[1:]
            break
    if base not in ALGORITHMS:
        raise ValueError("unknown v2 execution variant: "+variant)
    python = base.startswith("python")
    scalar = base in {"python-stream", "python-layered"}
    stream = base in {"python-stream", "python-stream-packed", "native-stream", "encoded-stream", "native-settle-stream"}
    fiber = "fiber-attention" in config["module"]
    family = config["graph"]
    if ("settle" in base or "layered" in base or base.startswith("encoded")) and family != "settle":
        raise ValueError("Settle variant requires SettleGraph")
    if family == "pdg" and ("frontier" in base or "diamond" in base or "chain" in base):
        # PDG can be acyclic, but this version's PDG rows intentionally exercise feedback.
        raise ValueError("v2 cyclic PDG workload requires streaming")
    if family == "settle" and base in {"native-stream", "native-frontier", "native-chain", "native-diamond", "python-frontier", "python-chain", "python-diamond"}:
        raise ValueError("choose a Settle frontend or explicit encoded variant")
    s = Schedule(workers=1 if python else workers, packed=not scalar and preset != "scalar",
                 prefill=not scalar and not stream and preset != "step")
    k, layout = Kernels(), "input"
    if preset == "optimized":
        if scalar:
            raise ValueError("scalar reference has no optimized policy")
        s = replace(s, full_autograd="batched", aggregate_autograd="batched")
        if not python:
            s = replace(s, packed_sources=True, batch_next=True, parallel_regions=True,
                        compact_events=True, defer_state_release=True)
            if fiber:
                k, layout = Kernels("single", "csr", "owned", "head"), "linear"
    if preset == "single":
        k = replace(k, attention_packing="single")
    overrides = {} if overrides is None else dict(overrides)
    if set(overrides)-FIELDS:
        raise ValueError("unknown execution option: "+str(sorted(set(overrides)-FIELDS)))
    s = replace(s, **{n: v for n, v in overrides.items() if n in SCHEDULE_FIELDS})
    k = replace(k, **{n: v for n, v in overrides.items() if n in KERNEL_FIELDS})
    layout = overrides.get("projection_layout", layout)
    for name, value in asdict(s).items():
        if name == "workers":
            if type(value) is not int or value < 1:
                raise ValueError("positive integer workers required")
        elif name.endswith("autograd"):
            if value not in {"replay", "batched"}:
                raise ValueError("unknown autograd policy")
        elif type(value) is not bool:
            raise ValueError(name+" must be bool")
    for name, allowed in {"attention_packing": {"exact", "single"}, "fiber_pooling": {"event", "csr"},
                          "fiber_cache": {"cloned", "owned"}, "attention_layout": {"event", "head"}}.items():
        if getattr(k, name) not in allowed:
            raise ValueError("unknown kernel policy: "+name)
    if layout not in {"input", "linear"}:
        raise ValueError("unknown projection layout")
    if not s.packed and (s.full_autograd == "batched" or s.aggregate_autograd == "batched" or s.packed_sources or s.batch_next):
        raise ValueError("batched VJP/transport requires packed execution")
    if s.defer_state_release and not s.compact_events:
        raise ValueError("deferred state release requires compact events")
    if stream and s.prefill:
        raise ValueError("streaming has no time prefill; choose a frontier/block schedule")
    if scalar and (s.packed or s.prefill or s.full_autograd != "replay" or s.aggregate_autograd != "replay"):
        raise ValueError("scalar reference does not implement packed policies")
    if not fiber and (k != Kernels() or layout != "input"):
        raise ValueError("fiber policies require same-fiber attention, not event Attention")
    if python and (s.workers != 1 or s.packed_sources or s.batch_next or s.parallel_regions
                   or s.compact_events or s.defer_state_release or k != Kernels()):
        raise ValueError("requested native-only policy has no Python implementation")
    return dict(schema="tide-execution-policy-v2", variant=variant, algorithm=base,
                requested=dict(preset=preset, worker_limit=workers, overrides=overrides),
                resolved=dict(schedule=asdict(s), kernels=asdict(k), model=dict(projection_layout=layout),
                              application=dict(head_workers=0, reason="graph-only workload has no vocabulary head")),
                scope=dict(node_parallel="not implemented in Python" if python else "independent node work only",
                           batch_counters="maximum submitted owners/rows; state kernels may regroup; inspect calls/scalar steps/operator work",
                           prefill="causal guards checked per node; actual sequence counters required",
                           training="first-order; state/Read retain semantic replay"))


def actual_paths(policy, stats, training):
    return dict(policy=policy, grad_enabled=training,
                max_state_sequence=stats.get("max_state_sequence", 0),
                max_state_batch=stats.get("max_state_batch", stats.get("max_node_batch", 0)),
                max_full_batch=stats.get("max_full_batch", 0),
                state_sequence_calls=stats.get("state_sequence_calls", 0),
                state_scalar_sequence_steps=stats.get("state_scalar_sequence_steps", 0),
                fallback_counters={k: v for k, v in stats.items() if "fallback" in k or k.startswith("state_prefill_")},
                replay_counters={k: v for k, v in stats.items() if "replay" in k},
                batch_counters={k: v for k, v in stats.items() if "batch" in k})
