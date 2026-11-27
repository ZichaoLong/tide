"""Adjudicate the finite early Stage A cohort from its bound observations and old seed rankings."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from statistics import mean

from .gate3 import MI
from .hbm_profile3 import reservation
from .prepare_resume3 import reference
from .round3 import DIAGNOSTIC_SEEDS, TUNING_SEEDS
from .scheduler3 import evaluation_bound
from .source_migration3 import read_bound


def seed_ranking(ref, keep):
    decision = read_bound(ref)
    groups = defaultdict(dict)
    for obs in decision["observations"]:
        if obs["eligible"]:
            vals = groups[obs["recipe_id"]]
            if obs["seed"] in vals or not math.isfinite(obs["nll"]):
                raise ValueError("invalid historical selection observation")
            vals[obs["seed"]] = obs["nll"]
    valid = {rid: [scores[s] for s in TUNING_SEEDS] for rid, scores in groups.items()
             if set(scores) == set(TUNING_SEEDS)}
    order = sorted(valid, key=lambda rid: (mean(valid[rid]), rid))
    if (len(order) <= keep or [row["recipe_id"] for row in decision["ranked"]] != order
            or [row["recipe_id"] for row in decision["selected"]] != order[:keep]
            or any(row["mean_nll"] != mean(valid[row["recipe_id"]]) for row in decision["ranked"])):
        raise ValueError("historical ranking does not reproduce")
    seed_orders = [sorted(valid, key=lambda rid: (valid[rid][idx], rid)) for idx in range(len(TUNING_SEEDS))]
    margins = [valid[o[keep]][i] - valid[o[keep-1]][i] for i, o in enumerate(seed_orders)]
    return dict(decision=ref, order=order, selected=order[:keep], seed_orders=seed_orders,
                boundary_margin=mean(valid[order[keep]])-mean(valid[order[keep-1]]),
                seed_boundary_margins=margins, stable=all(set(o[:keep]) == set(order[:keep]) for o in seed_orders)
                and min(margins) > 0)


def review_observations(mid, entries, reports, missing):
    from scripts.deferred_source_probe import compare

    sources = {}
    observed = {}
    per_batch = {}
    refs = []
    for key, item in entries.items():
        if key.rsplit("/", 1)[0] != mid:
            continue
        plan = read_bound(item["plan"])
        directory = Path(item.get("observations") or Path(item["plan"]["path"]).parent / "observations")
        report_ref = reference(directory / "comparison.json")
        report = read_bound(report_ref)
        arm = "weak_log" if "weak_log" in plan["arms"] else "reference"
        if (plan["model"] != mid or report["status"] != "observations_recorded"
                or report["model"] != mid or report["plan_sha256"] != item["plan"]["sha256"]
                or reports[key] != item["job"]):
            raise ValueError(f"unbound review: {key}")
        sources.setdefault(arm, plan["arms"][arm])
        if plan["arms"][arm] != sources[arm]:
            raise ValueError("target worker changed between backends")
        raw_refs = {a: reference(directory / (a + ".json")) for a in plan["arms"]}
        arms = {a: read_bound(r) for a, r in raw_refs.items()}
        keys = {case["key"] for case in plan["cases"]}
        if (len(keys) != len(plan["cases"]) or any(set(rows) != keys for rows in arms.values())
                or set(report["comparisons"]) != set(arms)-{"historical"}
                or any(set(rows) != keys for rows in report["comparisons"].values())):
            raise ValueError("partial review coverage")
        refs.append(dict(job=item["job"], plan=item["plan"], comparison=report_ref, observations=raw_refs))
        for c in plan["cases"]:
            k = c["key"]
            t = read_bound(c["task"])
            hist, new = arms["historical"][k], arms[arm][k]
            for a, comparisons in report["comparisons"].items():
                if compare(hist, arms[a][k]) != comparisons[k]:
                    raise ValueError("comparison does not reproduce from observations")
            outcome = report["comparisons"][arm][k]
            if (new["outcome"] != "updated" or new["task"] != t["id"] or new["case"] != k
                    or new["batch_tokens"] != c["tokens_per_arm"] or t["model_id"] != mid or k in observed
                    or not all(math.isfinite(new["metrics"][f]) for f in ("loss", "objective", "grad_norm", "lr"))
                    or outcome["outcome"] not in ("sample_exact", "sample_changed", "update_error_to_updated")):
                raise ValueError(f"target did not produce a comparable finite update: {k}")
            observed[k] = dict(phase=t["phase"], recipe=t.get("recipe_id"), seed=t["seed"],
                               task=t["id"], start=c["start"], comparison=outcome)
            sizes = per_batch.setdefault(str(t["batch_size"]), dict(hbm=0, rss=0, historical_hbm_request=0))
            mem = new["memory"]
            sizes["rss"] = max(sizes["rss"], mem["cpu_rss_peak_bytes"])
            if t["device"].startswith("npu"):
                sizes["hbm"] = max(sizes["hbm"], mem["npu_peak_reserved_bytes"], mem["npu_peak_allocated_bytes"])
                sizes["historical_hbm_request"] = max(sizes["historical_hbm_request"], t.get("reserved_hbm_bytes", 0))
    if len(sources) != 1 or not observed:
        raise ValueError(f"missing target for {mid}")
    arm, source = next(iter(sources.items()))
    return arm, source, observed, per_batch, refs, [r["task"] for r in missing if r["model"] == mid]


def disposition(subject, rankings, observations, missing):
    gap = subject["next_review"]
    candidates = [f"r{i}" for i in range(6)]
    if gap in ("incomplete_capacity_profile", "initialization", "complete_a1_comparison"):
        return "six_recipe_selection", candidates, "No complete six-recipe historical A1 decision at equal exposure."
    first = rankings["a1"]
    if not first["stable"]:
        return "six_recipe_selection", candidates, "Historical A1 finalists differ between the two seeds."
    finalists = first["selected"]
    if gap == "complete_a2_comparison":
        return "compare_finalists", finalists, "The two A1 finalists are stable; A2 lacks a complete comparison."
    second = rankings["a2"]
    winner = second["selected"][0]
    if gap == "complete_a3_confirmation":
        if (second["stable"] and {r["phase"] for r in observations.values()} == {"a3"}
                and {(r["recipe"], r["seed"]) for r in observations.values()} ==
                {(winner, s) for s in TUNING_SEEDS} and
                all(r["comparison"]["outcome"] == "sample_exact" for r in observations.values())):
            return "confirm_historical_winner", [winner], "Both old ranks are stable and both A3 successor samples match; complete fresh confirmation."
        return "compare_finalists", finalists, "A3 is incomplete and target confirmation updates differ or repair a historical failure."
    if gap != "source_equivalence_on_winning_chain_and_competing_recipes":
        raise ValueError("unknown historical gap")
    points = defaultdict(list)
    for r in observations.values():
        points[r["phase"], r["recipe"], r["seed"]].append(r["comparison"]["outcome"])
    exact = {key for key, outcomes in points.items() if all(o == "sample_exact" for o in outcomes)}
    # Check both sides of each selection boundary, including the strongest A1 exclusion on either seed.
    boundary = set(finalists) | {order[2] for order in first["seed_orders"]} | {first["order"][2]}
    required = {(stage, rid, seed) for stage, recipes in (("a1", boundary), ("a2", finalists), ("a3", [winner]))
                for rid in recipes for seed in TUNING_SEEDS}
    a1 = [r["comparison"] for r in observations.values() if r["phase"] == "a1"]
    finite_losers = (len(a1) == 12 and all(v["outcome"] in {"sample_exact", "sample_changed"}
                     and v["equal"]["activity"] for v in a1)
                     and max(abs(v["delta"]["loss"]) for v in a1) < min(first["seed_boundary_margins"]))
    if second["stable"] and not missing and required <= exact and finite_losers:
        return "carry_recipe", [winner], "Historical selection retained as a recipe recommendation; both seed boundaries and A3 sampled exactly. Other A1 updates can differ."
    reason = "Reconstruct the missing A2 comparisons from fresh seed parents." if missing else "Decisive target updates differ from the old trajectory."
    return "compare_finalists", finalists, reason


def phase_counts(mode, gap, recipes):
    if mode == "carry_recipe":
        return {}
    return dict(capacity=3 if gap == "incomplete_capacity_profile" else 0, init=len(DIAGNOSTIC_SEEDS),
                a1=len(recipes)*len(TUNING_SEEDS), a2=min(2,len(recipes))*len(TUNING_SEEDS), a3=len(TUNING_SEEDS))


def evaluation_forecast(counts):
    total = 0
    for phase, target, start, size in (("init", 32768, 0, 256), ("a1", MI//4, 0, 16384),
                                       ("a2", MI, MI//4, 16384), ("a3", 2*MI, MI, 16384)):
        total += counts.get(phase, 0) * evaluation_bound(dict(target_tokens=target, evaluation_start_tokens=start,
                                                eval_interval_tokens=0 if phase=="init" else MI, eval_tokens=size))
    return total


def summarize_observations(observed):
    stages = defaultdict(list)
    for row in observed.values():
        stages[row["phase"]].append(row)
    result = {}
    for phase, rows in stages.items():
        comparisons = [r["comparison"] for r in rows]
        result[phase] = dict(counts=dict(Counter(c["outcome"] for c in comparisons)),
                             starts=dict(Counter(r["start"] for r in rows)),
                             different_fields=dict(Counter(field for c in comparisons
                                 for field, equal in c.get("equal", {}).items() if not equal)),
                             max_abs_delta={k:max((abs(c["delta"][k]) for c in comparisons if "delta" in c), default=None)
                                            for k in ("loss","objective","grad_norm","lr")})
    return result


def reviewed_requests(per_batch, resource, factor, padding):
    """Use target NPU samples, historical full-training peaks and original declarations."""
    result = {}
    for batch in set(per_batch) | set(resource.get("npu_peak_bytes_by_batch", {})):
        sample = per_batch.get(batch, {})
        peak = max(sample.get("hbm", 0), resource.get("npu_peak_bytes_by_batch", {}).get(batch, 0))
        result[batch] = dict(hbm=max(reservation(peak, factor, padding), sample.get("historical_hbm_request", 0)),
                             rss=max(reservation(sample.get("rss", 0), factor, padding),
                                     resource.get("rss_peak_bytes_by_batch", {}).get(batch, 0)), peak=peak)
    return result
