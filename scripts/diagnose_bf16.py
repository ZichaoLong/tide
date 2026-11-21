"""Bounded first-support-divergence witnesses for BF16 regional decisions.

capture runs actual model trajectories and saves detached regional inputs.
analyze is CPU-only: it replays the rounded inputs in FP64 and solves quad
independently with Decimal breakpoint intervals. Neither supplies a trajectory
to model execution, clips support, nor changes a qualification threshold.
"""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import copy
from decimal import Decimal, localcontext
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess

import torch

from tide import ModelConfig, TideModel
from tide.precision import precision_context
from scripts.job import atomic
from scripts.qualify_precision import atomic_tensor, comparison, cpu_tree, fingerprint, THRESHOLDS


@contextmanager
def capture_regions(model, entries, attention):
    module = importlib.import_module("tide.model")
    attention_module = importlib.import_module("tide.attention")
    original_control, original_native = module.region_control, attention_module._native_attention
    regions = [i for i, group in enumerate(model.region_group) if group >= 0]
    current = []
    aliases = []
    def native(*args, **kwargs):
        if current:
            current[-1]["effective"] = args[5]
        return original_native(*args, **kwargs)
    def wrap(site, original):
        def call(q, k, v, *args, **kwargs):
            weights = kwargs.get("weights", args[0] if args else None)
            backend = kwargs.get("backend", "auto")
            row = {"site": site, "requested": backend, "q_dtype": str(q.dtype),
                   "trainable_prior": weights is not None and weights.requires_grad,
                   "zero_prior": weights is not None and bool((weights == 0).any()),
                   "effective": "tiled" if backend in {"auto", "tiled"} else backend}
            current.append(row)
            try:
                result = original(q, k, v, *args, **kwargs)
                attention.append(row)
                return result
            finally:
                current.pop()
        return call
    def control(score, received, roles, budget, profile, partition, *args, **kwargs):
        result = original_control(score, received, roles, budget, profile, partition, *args, **kwargs)
        region = regions[len(entries)]
        entries.append(cpu_tree({"region": region, "layer": model.graph.nodes[model.graph.regions[region][0]].layer,
                                 "nodes": model.graph.regions[region], "score": score, "received": received,
                                 "roles": roles, "budget": budget, "profile": profile, "partition": partition,
                                 "control": result[0] if isinstance(result, tuple) else result}))
        return result
    module.region_control = control
    attention_module._native_attention = native
    for site in ("kv", "reception", "region"):
        local = importlib.import_module("tide." + site)
        aliases.append((local, local.weighted_attention))
        local.weighted_attention = wrap(site, local.weighted_attention)
    try:
        yield
    finally:
        module.region_control = original_control
        attention_module._native_attention = original_native
        for local, original in aliases:
            local.weighted_attention = original


def decimal_quad(score, received, roles, budget, partition):
    """Independent forward solve on exact decimal images of recorded floats."""
    with localcontext() as context:
        context.prec = 80
        zero, one = Decimal(0), Decimal(1)
        scores = [Decimal.from_float(float(x)) for x in score]
        reception = [Decimal.from_float(float(x)) for x in received]
        w = [max(zero, min(one, x)) for x in reception]
        mu = [max(zero, min(one, 2*x)) for x in reception]
        hubs = sum(bool(x) for x in roles)
        lower = [x/4 if hub else zero for x, hub in zip(mu, roles)]
        if hubs == len(scores) and partition == "singleton":
            return {"control": [float(x) for x in mu], "multiplier": None, "margin": [None]*len(scores)}
        k = Decimal(max(1, (hubs+1)//2) if hubs == len(scores) else budget + max(1, (hubs+1)//2) if hubs else budget)
        low = max(sum(lower), min(one, sum(w), k, sum(mu)))
        high = min(k, sum(mu))
        def response(multiplier):
            return [max(lo, min(hi, s-multiplier)) for s, lo, hi in zip(scores, lower, mu)]
        c = response(zero)
        total = sum(c)
        multiplier = zero
        if not low <= total <= high:
            target = low if total < low else high
            events = sorted(set(s-bound for s, lo, hi in zip(scores, lower, mu) if lo < hi for bound in (lo, hi)))
            if target == sum(lower):
                multiplier = max(events, default=zero)
                c = list(lower)
            elif target == sum(mu):
                multiplier = min(events, default=zero)
                c = list(mu)
            else:
                found = False
                for left, right in zip(events, events[1:]):
                    at_left = response(left)
                    if sum(at_left) == target:
                        multiplier, c, found = left, at_left, True
                        break
                    at_right = response(right)
                    if sum(at_right) == target:
                        multiplier, c, found = right, at_right, True
                        break
                    if sum(at_left) > target > sum(at_right):
                        middle = (left+right)/2
                        middle_c = response(middle)
                        slope = sum(lo < x < hi for lo, x, hi in zip(lower, middle_c, mu))
                        multiplier = middle + (sum(middle_c)-target)/slope
                        c, found = response(multiplier), True
                        break
                if not found:
                    raise ArithmeticError("independent Decimal interval did not locate a feasible quad solution")
        margin = [float(s-multiplier-lo) for s, lo in zip(scores, lower)]
        return {"control": [float(x) for x in c], "multiplier": float(multiplier), "margin": margin,
                "total": float(sum(c)), "low_total": float(low), "high_total": float(high)}


def replay(row, batch, time):
    from tide.region import region_control
    score = row["score"][:, batch, time].double()
    received = row["received"][:, batch, time].double()
    if row["profile"] != "quad":
        raise ValueError("this bounded independent diagnostic currently selects quad witnesses")
    decimal = decimal_quad(score.tolist(), received.tolist(), row["roles"].tolist(), row["budget"], row["partition"])
    with torch.no_grad():
        double = region_control(score[:, None, None], received[:, None, None], row["roles"], row["budget"],
                                row["profile"], row["partition"])[:, 0, 0]
    actual = row["control"][:, batch, time].double()
    oracle = torch.tensor(decimal["control"], dtype=torch.float64)
    return {"scores": score.tolist(), "received": received.tolist(), "production": actual.tolist(),
            "fp64_replay": double.tolist(), "decimal": decimal,
            "production_decimal_max_abs": float((actual-oracle).abs().max()),
            "production_decimal_support_mismatches": int(((actual > 0) != (oracle > 0)).sum()),
            "fp64_decimal_max_abs": float((double-oracle).abs().max()),
            "fp64_decimal_support_mismatches": int(((double > 0) != (oracle > 0)).sum())}


def compare_trace(left, right):
    stages = []
    for a, b in zip(left["regions"], right["regions"]):
        if a["region"] != b["region"] or a["score"].shape != b["score"].shape:
            raise ValueError("captured region schedules disagree")
        support = (a["control"] > 0) != (b["control"] > 0)
        row = {"region": a["region"], "layer": a["layer"],
               "score_max_abs": float((a["score"].double()-b["score"].double()).abs().max()),
               "received_max_abs": float((a["received"].double()-b["received"].double()).abs().max()),
               "control_max_abs": float((a["control"].double()-b["control"].double()).abs().max()),
               "support_mismatches": int(support.sum())}
        stages.append(row)
    first_support = next((i for i, row in enumerate(stages) if row["support_mismatches"]), None)
    witnesses = []
    if first_support is not None:
        a, b = left["regions"][first_support], right["regions"][first_support]
        mask = (a["control"] > 0) != (b["control"] > 0)
        positions = sorted(set((batch, time) for _, batch, time in mask.nonzero().tolist()))[:4]
        for batch, time in positions:
            ar, br = replay(a, batch, time), replay(b, batch, time)
            oracle_match = ar["production_decimal_support_mismatches"] == br["production_decimal_support_mismatches"] == 0
            witnesses.append({"batch": batch, "time_index": time, "reference": ar, "candidate": br,
                              "diagnosis": "recorded_score_or_reception_changes_cross_support_boundary" if oracle_match else
                                           "production_allocator_support_differs_from_exact_rounded_input_oracle"})
    return {"reference": left["name"], "candidate": right["name"], "stages": stages,
            "first_score_difference": next((r for r in stages if r["score_max_abs"]), None),
            "first_support_difference": None if first_support is None else stages[first_support], "witnesses": witnesses,
            "final": comparison(right["result"], left["result"], THRESHOLDS["bf16"]),
            "readout": comparison(right["result"]["readout"], left["result"]["readout"], THRESHOLDS["bf16"]),
            "raw_values": comparison(right["result"]["values"], left["result"]["values"], THRESHOLDS["bf16"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--device", default="cpu")
    capture.add_argument("--config", required=True)
    capture.add_argument("--batch", type=int, default=2)
    capture.add_argument("--length", type=int, default=5)
    capture.add_argument("--threads", type=int, default=4)
    capture.add_argument("--tiled", action="store_true")
    capture.add_argument("--fp64", action="store_true")
    capture.add_argument("--trace", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--trace", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output identity")
    if args.command == "analyze":
        torch.set_num_threads(4)
        trace = torch.load(args.trace, weights_only=False, map_location="cpu")
        runs = {r["name"]: r for r in trace["runs"]}
        pairs = [("fp32_grad", "bf16_grad"), ("bf16_grad", "bf16_inference"),
                 ("fp64_grad", "fp32_grad"), ("fp64_grad", "bf16_grad"),
                 ("bf16_tiled_grad", "bf16_tiled_inference"), ("bf16_grad", "bf16_tiled_grad")]
        report = {"capture": trace["identity"], "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                  "scope": "first differing region, up to four sample/time rows; forward allocation only; original failures unchanged",
                  "comparisons": [compare_trace(runs[a], runs[b]) for a, b in pairs if a in runs and b in runs]}
        atomic(args.output, report)
        print(json.dumps([{k: row[k] for k in ("reference", "candidate", "first_support_difference")}
                          for row in report["comparisons"]]), flush=True)
        return 0
    if args.trace.exists():
        parser.error("choose a new trace identity")
    if args.fp64 and args.device != "cpu":
        parser.error("FP64 model replay is a separate CPU diagnostic")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    torch.set_num_threads(args.threads)
    path = Path(args.config)
    cfg = ModelConfig.from_dict(json.loads(path.read_text()))
    seed = 31415 + int(hashlib.sha256(path.stem.encode()).hexdigest()[:8], 16)
    torch.manual_seed(seed)
    original = TideModel(cfg)
    x = torch.randn(args.batch, args.length, cfg.input_dim or cfg.d)
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    identity = {"source": source, "config": cfg.to_dict(), "seed": seed, "device": str(device),
                "initial_parameter_sha256": fingerprint(original.state_dict()), "input_sha256": fingerprint(x),
                "torch": str(torch.__version__), "physical_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": args.threads,
                "intrusive": "regional tensor copies and attention dispatch audit; no timing claim"}
    runs = []
    modes = ["fp32_grad", "bf16_grad", "bf16_inference"]
    if args.fp64:
        modes.append("fp64_grad")
    if args.tiled:
        modes += ["bf16_tiled_grad", "bf16_tiled_inference"]
    for mode in modes:
        model = copy.deepcopy(original).to(device=device, dtype=torch.float64 if mode.startswith("fp64") else torch.float32)
        if "tiled" in mode:
            for module in model.modules():
                if hasattr(module, "backend"):
                    module.backend = "tiled"
        value = x.to(device=device, dtype=next(model.parameters()).dtype)
        regions, attention = [], []
        context = nullcontext() if mode.startswith("fp64") else precision_context(device, mode.split("_")[0])
        with torch.set_grad_enabled(mode.endswith("grad")), context, capture_regions(model, regions, attention):
            result = model(value)
        counts = Counter(tuple(sorted(row.items())) for row in attention)
        runs.append({"name": mode, "regions": regions, "result": cpu_tree({"values": result.values, "strengths": result.strengths,
                                                                            "state": result.state, "readout": result.readout()}),
                     "attention": [{**dict(key), "calls": count} for key, count in counts.items()]})
        print(json.dumps({"captured": mode, "regions": len(regions)}), flush=True)
        del result, model
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    atomic_tensor(args.trace, {"identity": identity, "runs": runs})
    atomic(args.output, {"identity": identity, "trace": str(args.trace),
                         "runs": [{"name": r["name"], "attention": r["attention"]} for r in runs]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
