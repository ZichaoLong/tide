"""Finite full-Reception comparisons with fixed parameters and real mass roots.

This includes projections, residual/source transforms and aggregation. It is
not a full graph benchmark. Normal timings exclude the separate operator audit
and numerical comparisons. Training includes real AdamW updates; a new module
and optimizer start each candidate from the same frozen CPU parameter fixture.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time

import torch
from torch.utils.checkpoint import checkpoint

from tide.parameters import NodeOwners
from tide.precision import ArithmeticAudit, precision_context, precision_policy
from tide.reception import Reception
from benchmarks.attention_sites import dispatch_observation, synchronize
from benchmarks.continuous import nested_comparison
from scripts.job import atomic
from scripts.qualify_precision import THRESHOLDS, cpu_tree, fingerprint


PROFILES = ("source_residual", "incoming_light_64", "incoming_light_128", "incoming_wide")
CANDIDATES = {"separate": ("separate", False), "fused": ("fused", False),
              "masked": ("separate", True), "fused_masked": ("fused", True)}


def make_fixture(profile, case, *, nodes, batch, length, edges, d, sharing, seed=181119):
    """Static absence and genuine zero mass have intentionally different roots."""
    groups = [0] * nodes if sharing == "shared" else None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        module = Reception([{"DD", "HD", "ext"}] * nodes, d, profile, groups, rank=8)
    rng = torch.Generator().manual_seed(seed+1)
    content = torch.randn(nodes, batch, length, edges, d, generator=rng)
    raw_mass = .2 + torch.rand(nodes, batch, length, edges, generator=rng)
    counts = torch.full((nodes,), edges)
    if case != "positive":
        counts = (edges - torch.arange(nodes) % max(2, edges)).clamp_min(1)
    membership = torch.arange(edges)[None] < counts[:, None]
    if case == "zero":
        raw_mass[..., ::3] = 0
        raw_mass.reshape(-1, edges)[::7] = 0
    routed = raw_mass * membership[:, None, None]
    fixture = dict(content=content, raw_mass=raw_mass, membership=membership,
                   classes=torch.arange(edges)[None].expand(nodes, -1) % 3,
                   cotangent=torch.randn(nodes, batch, length, d, generator=rng),
                   parameters=module.state_dict())
    spec = dict(profile=profile, case=case, nodes=nodes, batch=batch, length=length, edges=edges,
                d=d, sharing=sharing, structural_edges=int(membership.sum()),
                padded_edges=int((~membership).sum()), positive_messages=int((routed > 0).sum()),
                genuine_zero_messages=int(((raw_mass == 0) & membership[:, None, None]).sum()),
                physical_message_slots=nodes*batch*length*edges,
                unique_parameters=sum(p.numel() for p in module.parameters()))
    return fixture, spec


def evaluate(fixture, spec, *, device, precision, mode, candidate, backend, recompute=False,
             warmup=2, steps=5, learning_rate=1e-4):
    qkv_backend, mask_padding = CANDIDATES[candidate]
    nodes, d = spec["nodes"], spec["d"]
    module = Reception([{"DD", "HD", "ext"}] * nodes, d, spec["profile"],
                       [0] * nodes if spec["sharing"] == "shared" else None,
                       rank=8, backend=backend, qkv_backend=qkv_backend).to(device)
    module.load_state_dict(fixture["parameters"])
    grad = mode != "inference"
    backward = mode in {"forward_backward", "training"}
    content = fixture["content"].to(device).detach().requires_grad_(grad)
    raw_mass = fixture["raw_mass"].to(device).detach().requires_grad_(grad)
    membership, classes = (fixture[key].to(device) for key in ("membership", "classes"))
    owners = NodeOwners(torch.arange(nodes, device=device), tuple(range(nodes)))
    seed = fixture["cotangent"].to(device)
    roots = dict(content=content, raw_mass=raw_mass, **dict(module.named_parameters()))
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, foreach=False) if mode == "training" else None
    effective_recompute = recompute and grad

    def clear():
        for root in roots.values():
            root.grad = None

    def compute():
        clear()
        with torch.set_grad_enabled(grad), precision_context(device, precision):
            # Graph routing always eliminates nonexistent sender roots, even
            # for the older unmasked Attention path. Comparisons must not
            # mistake padded-slot mass derivatives for real edge derivatives.
            mass = raw_mass * membership[:, None, None]
            args = (content, mass, classes, owners, membership if mask_padding else None)
            total, aggregate = (checkpoint(module, *args, use_reentrant=False) if effective_recompute else module(*args))
            if backward:
                loss = (aggregate.float()*seed).mean() + .01*total.float().mean()
        if backward:
            loss.backward()
        if optimizer is not None:
            optimizer.step()
        synchronize(device)
        return total, aggregate

    for _ in range(warmup):
        compute()
    clear()
    synchronize(device)
    before = allocated = reserved = None
    if device.type in {"cuda", "npu"}:
        api = getattr(torch, device.type)
        before = api.memory_allocated(device)
        api.reset_peak_memory_stats(device)
    samples = []
    for _ in range(steps):
        start = time.perf_counter()
        compute()
        samples.append(time.perf_counter()-start)
    if device.type in {"cuda", "npu"}:
        allocated, reserved = api.max_memory_allocated(device), api.max_memory_reserved(device)
    with dispatch_observation(device) as dispatch, ArithmeticAudit() as audit:
        result = compute()
    snapshot = cpu_tree(dict(result=result, gradients={name: value.grad for name, value in roots.items()},
                             parameters=module.state_dict(),
                             optimizer=None if optimizer is None else optimizer.state_dict()))
    if spec["profile"] == "source_residual":
        dispatch = dict(effective="absent", native_api=None)
    gradients = {name: dict(present=value.grad is not None,
                           dtype=None if value.grad is None else str(value.grad.dtype),
                           finite=None if value.grad is None else bool(torch.isfinite(value.grad).all()))
                 for name, value in roots.items()}
    seconds = statistics.median(samples)
    record = dict(requested_qkv_backend=qkv_backend, effective_qkv_backend=module.effective_qkv_backend,
                  mask_padding=mask_padding, recompute_requested=recompute,
                  recompute_effective=effective_recompute, requested_attention_backend=backend,
                  seconds=samples, median_seconds=seconds,
                  reception_node_positions_per_second=nodes*spec["batch"]*spec["length"]/seconds,
                  throughput_definition="receiver node x sample x position per second; not backbone input tokens",
                  device_allocated_before_samples=before, peak_device_allocated_bytes=allocated,
                  peak_device_reserved_bytes=reserved,
                  process_peak_rss_kib_cumulative=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  training_updates_including_warmup_and_audit=warmup+steps+1 if optimizer is not None else 0,
                  snapshot_sha256=fingerprint(snapshot), parameter_storage_dtype="torch.float32",
                  input_storage_dtype="torch.float32", prior_dtype="torch.float32",
                  audit=dict(scope="one separate post-timing forward and requested backward/update",
                             attention=dispatch, operators=audit.report(), gradients=gradients,
                             result_dtypes=[str(x.dtype) for x in result],
                             finite=all(bool(torch.isfinite(x).all()) for x in result),
                             hardware_accumulation_inferred=False))
    self_check = nested_comparison(snapshot, snapshot, **THRESHOLDS[precision])
    connected = not backward or all(row["present"] for row in gradients.values())
    record["state"] = "completed" if self_check["finite"] and connected else "nonfinite_or_disconnected"
    return record, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES[1:]))
    parser.add_argument("--cases", nargs="+", choices=("positive", "padded", "zero"), default=["padded", "zero"])
    parser.add_argument("--precisions", nargs="+", choices=("fp32", "bf16"), default=["fp32"])
    parser.add_argument("--modes", nargs="+", choices=("inference", "grad_forward", "forward_backward", "training"),
                        default=["inference", "training"])
    parser.add_argument("--candidates", nargs="+", choices=tuple(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--backends", nargs="+", choices=("auto", "math", "tiled", "sdpa", "npu_fusion"), default=["auto"])
    parser.add_argument("--sharing", choices=("independent", "shared"), default="shared")
    parser.add_argument("--checkpoint-reception", action="store_true")
    for name, default in (("nodes", 8), ("batch", 8), ("length", 128), ("edges", 8), ("d", 128),
                          ("threads", 4), ("steps", 5)):
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output identity")
    if min(args.nodes, args.batch, args.length, args.edges, args.d, args.threads, args.steps) < 1 or args.warmup < 0:
        parser.error("counts must be positive and warmup nonnegative")
    if args.d % 64:
        parser.error("d must be divisible by 64")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    torch.set_num_threads(args.threads)
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    candidates = ["separate"] + [name for name in dict.fromkeys(args.candidates) if name != "separate"]
    record = dict(source=source, harness_source=os.environ.get("TIDE_HARNESS_COMMIT", source),
                  arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  torch=str(torch.__version__), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  threads=torch.get_num_threads(), physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unmanaged observation; shared host"),
                  boundary="resident inputs, static routed mass, full Reception, output and device sync; grad modes build graph; backward adds mean loss/clear-grad/backward; training includes AdamW",
                  numerical_boundary="paired same-profile/case/precision/mode/backend/recompute separate-projection reference; true zero prior roots retained, structural roots absent on every path",
                  memory_scope="peak allocated/reserved after warmup; reserved may retain earlier allocator blocks; CPU RSS is cumulative high-water mark",
                  precision_policies={precision: precision_policy(device, precision) for precision in args.precisions}, rows=[])
    if device.type == "npu":
        record.update(torch_npu=str(torch_npu.__version__), allow_hf32=False)
    for profile in args.profiles:
        for case in args.cases:
            fixture, spec = make_fixture(profile, case, nodes=args.nodes, batch=args.batch, length=args.length,
                                         edges=args.edges, d=args.d, sharing=args.sharing)
            identity = fingerprint(fixture)
            for precision in args.precisions:
                for mode in args.modes:
                    for backend in args.backends:
                        reference = None
                        for candidate in candidates:
                            row = dict(profile=profile, case=case, precision=precision, mode=mode,
                                       candidate=candidate, requested_attention_backend=backend,
                                       shape=spec, canonical_fixture_sha256=identity,
                                       state="running", host_load_before=os.getloadavg())
                            record["rows"].append(row)
                            atomic(args.output, record)
                            try:
                                observation, snapshot = evaluate(fixture, spec, device=device, precision=precision,
                                    mode=mode, candidate=candidate, backend=backend, recompute=args.checkpoint_reception,
                                    warmup=args.warmup, steps=args.steps)
                                row.update(observation)
                                if candidate == "separate":
                                    reference = snapshot
                                if reference is not None:
                                    comparison = nested_comparison(snapshot, reference, **THRESHOLDS[precision])
                                    row["vs_separate"] = comparison
                                    if row["state"] == "completed":
                                        row["state"] = ("structural_failed" if comparison["metadata_mismatches"] else
                                                        "within_threshold" if comparison["within_threshold"] else "finite_deviation")
                                del snapshot
                            except Exception as error:
                                import traceback
                                traceback.print_exc()
                                row.update(state="execution_failed", error=repr(error))
                            row["host_load_after"] = os.getloadavg()
                            atomic(args.output, record)
                            print(json.dumps({key: row[key] for key in ("profile", "case", "precision", "mode", "candidate", "state")}), flush=True)
    return int(any(row["state"] != "within_threshold" for row in record["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
