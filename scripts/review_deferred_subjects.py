"""Rebuild the 74 subject dispositions from the original results and completed source probes."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from pilotlab.deferred_review3 import (disposition, evaluation_forecast, phase_counts, reviewed_requests,
                                      review_observations, seed_ranking, summarize_observations)
from pilotlab.gate3 import MI, choose_recipes, eligible
from pilotlab.hbm_profile3 import HbmProfile
from pilotlab.io import sha256
from pilotlab.prepare_resume3 import reference
from pilotlab.round3 import DIAGNOSTIC_SEEDS, TUNING_SEEDS
from pilotlab.scheduler3 import freeze_json, read
from pilotlab.source_migration3 import read_bound
from scripts.review_io import write_review_json


def original_result(directory, tid, row):
    task_ref = reference(row.get("task_path") or directory / "tasks" / tid / "task.json")
    result_ref = dict(path=row["result_path"], sha256=row["result_sha256"])
    task, result = read_bound(task_ref), read_bound(result_ref)
    if (result["task_sha256"] != task_ref["sha256"] or result["id"] != tid or task["id"] != tid
            or result["model_id"] != task["model_id"] or result["source_identity"] != task["source_identity"]):
        raise ValueError(f"unbound historical result: {tid}")
    return task, result, dict(task=task_ref, result=result_ref)


def verify_rankings(subject, history):
    ranks = {}
    for phase, old in subject.get("replayed_rankings", {}).items():
        keep = 2 if phase == "a1" else 1
        decision = read_bound(old["decision"])
        tasks, results = [], {}
        for obs in decision["observations"]:
            task, result, _ = history[obs["id"]]
            if task["model_id"] != subject["model_id"] or task["phase"] != phase:
                raise ValueError("selection inputs belong to another subject or phase")
            tasks.append(task)
            results[task["id"]] = result
        if choose_recipes(tasks, results, TUNING_SEEDS, keep=keep) != decision:
            raise ValueError("selection does not reproduce from the bound worker results")
        ranks[phase] = seed_ranking(old["decision"], keep)
    return ranks


def verify_confirmations(subject, history):
    verified = []
    for cp in subject.get("confirmed_old_a3", []):
        tid = f'{subject["model_id"]}-a3-{cp["seed"]}-{subject["old_recipe"]}'
        task, result, refs = history[tid]
        end = result["checkpoint"]
        if (refs["result"]["sha256"] != cp["result_sha256"] or eligible(task, result, require_resume=True)
                or Path(end["path"]).resolve() != Path(cp["path"]).resolve() or end["sha256"] != cp["sha256"]
                or sha256(end["path"]) != cp["sha256"]):
            raise ValueError("historical A3 checkpoint or confirmation changed: " + tid)
        verified.append(dict(seed=cp["seed"], checkpoint=dict(path=end["path"], sha256=end["sha256"]), **refs))
    if subject.get("old_recipe") and sorted(v["seed"] for v in verified) != TUNING_SEEDS:
        raise ValueError("old selection lacks both A3 checkpoints")
    return verified


def scheduler_history(refs, mids):
    history, prior, peaks = {}, Counter(), {}
    for name in ("r01_scheduler", "r07_scheduler"):
        state = read_bound(refs[name])
        directory = Path(refs[name]["path"]).resolve().parent
        for tid, row in state["tasks"].items():
            mid = row["model_id"]
            if mid not in mids:
                continue
            if row.get("evaluation_budget_key") == mid:
                prior[mid] += row.get("evaluation_reserved_tokens", 0)
            if row["status"] != "completed" or not row.get("result_sha256"):
                continue
            task, result, files = original_result(directory, tid, row)
            history[tid] = task, result, files
            mem = result.get("device_memory") or {}
            size = peaks.setdefault(mid, {}).setdefault(str(task["batch_size"]), dict(rss=0, hbm=0, historical_hbm_request=0))
            size["rss"] = max(size["rss"], mem.get("cpu_rss_peak_bytes", 0))
            if result["device"].startswith("npu"):
                size["hbm"] = max(size["hbm"], *(mem.get(k, 0) for k in (
                    "npu_peak_reserved_bytes", "npu_peak_allocated_bytes", "npu_startup_peak_reserved_bytes",
                    "npu_startup_peak_allocated_bytes")))
                size["historical_hbm_request"] = max(size["historical_hbm_request"], task.get("reserved_hbm_bytes", 0))
    return history, prior, peaks


def completed_probes(launch, jobs, matrix):
    original = {row["model"] + "/" + row["backend"]: row["plan"] for row in matrix["jobs"]}
    if original != {k: j["plan"] for k, j in jobs.items()} or len({j["job"] for j in jobs.values()}) != len(jobs):
        raise ValueError("final review jobs do not cover the finite matrix")
    root = Path(launch["runq_root"]) / "queue.sqlite3"
    verified, envelopes = {}, {}
    with closing(sqlite3.connect(root.as_uri() + "?mode=ro", uri=True)) as db:
        for key, entry in jobs.items():
            mid, backend = key.rsplit("/", 1)
            row = db.execute("SELECT state,rc,backend,spec FROM jobs WHERE id=?", (entry["job"],)).fetchone()
            if not row or row[:3] != ("completed", 0, backend):
                raise ValueError("source probe did not complete: " + key)
            spec = json.loads(row[3])
            if (spec["task_id"] != "early-review-" + key.replace("/", "-") or spec["owner"] != launch["output"]
                    or spec["purpose"] != "deferred_source_review"):
                raise ValueError("source probe queue identity changed")
            verified[key] = entry["job"]
            if backend == "npu":
                envelopes[mid] = spec["hbm"]
    return verified, envelopes


def make_review(launch):
    binding = launch["deferred_source_review"]
    matrix = read_bound(binding["matrix"])
    inventory = read_bound(matrix["inventory"])
    jobs = read_bound(binding["submissions"])["jobs"]
    active_ref = {"path": launch["plan_path"], "sha256": launch["plan_sha256"]}
    main = read_bound(active_ref)
    original = read_bound(inventory["references"]["original_plan"])
    subjects = {s["id"]: s for s in original["subjects"]}
    mids = {s["model_id"] for s in inventory["subjects"]}
    active = {s["id"] for s in main["subjects"]}
    if (len(mids) != 74 or len(active) != 270 or set(subjects) != mids | active or mids & active
            or inventory["seeds"] != dict(initialization_diagnostics=DIAGNOSTIC_SEEDS, tuning=TUNING_SEEDS)
            or matrix["deadline"] != main["config"]["deadline_unix"] or matrix["official_owner"] != launch["output"]):
        raise ValueError("cohort, seeds or original campaign changed")
    verified, envelopes = completed_probes(launch, jobs, matrix)
    resources = read_bound(inventory["references"]["r07_model_resources"])
    history, prior, peaks = scheduler_history(inventory["references"], mids)
    profile = HbmProfile(launch["output"], readonly=True)
    try:
        factor, padding = profile.policy()
    finally:
        profile.db.close()
    rows, sources, outcomes, counts, totals = {}, {}, Counter(), Counter(), Counter()
    all_comparisons = Counter()
    for s in inventory["subjects"]:
        mid = s["model_id"]
        arm, source, observed, per_batch, reports, missing = review_observations(
            mid, jobs, verified, matrix["unavailable"])
        policy = source["model_policy"]
        model = subjects[mid]["model"]
        if (arm == "weak_log") != (model["config"]["allocation"] == "shannon"):
            raise ValueError("Shannon source arm mismatch")
        software = {k: source[k] for k in ("source", "foundation", "identity")}
        software["adapter"] = reference(Path(source["source"]) / "pilotlab" / "resume_adapter3.py")
        if sources.setdefault(arm, software) != software:
            raise ValueError("heterogeneous software for the same source arm")
        for rep in reports:
            for case in read_bound(rep["plan"])["cases"]:
                if read_bound(case["task"])["model"] != model:
                    raise ValueError("probe model is not the catalog model")
            for values in read_bound(rep["comparison"])["comparisons"].values():
                all_comparisons.update(v["outcome"] for v in values.values())
        ranks = verify_rankings(s, history)
        confirmations = verify_confirmations(s, history)
        if confirmations and ranks["a2"]["selected"] != [s["old_recipe"]]:
            raise ValueError("old confirmation differs from the replayed winner")
        init = []
        for seed in DIAGNOSTIC_SEEDS:
            record = history.get(f"{mid}-init-{seed}")
            if record and not eligible(record[0], record[1]):
                init.append(dict(seed=seed, **record[2]))
        mode, recipes, reason = disposition(s, ranks, observed, missing)
        gap = s["next_review"]
        work = phase_counts(mode, gap, recipes)
        future = evaluation_forecast(work)
        used = s["prior_evaluation_reserved_tokens"]["historical_total_without_double_counting"]
        if prior[mid] != used or used + future > MI:
            raise ValueError("inconsistent or exhausted evaluation budget: " + mid)
        if (gap == "incomplete_capacity_profile") != (mid not in resources):
            raise ValueError("historical capacity status and sizing data disagree")
        for batch, historical in peaks.get(mid, {}).items():
            sizes = per_batch.setdefault(batch, {})
            for field, amount in historical.items():
                sizes[field] = max(amount, sizes.get(field, 0))
        requests = reviewed_requests(per_batch, resources.get(mid, {}), factor, padding)
        if s.get("lineage_reference"):
            read_bound(s["lineage_reference"])
        rows[mid] = dict(mode=mode, reason=reason, old_gap=gap, old_recipe=s.get("old_recipe"),
            old_lineage=s.get("lineage_reference"), old_a3=confirmations, recipe_rankings=ranks,
            recipes=recipes, reports=reports, observed=summarize_observations(observed),
            missing_a2_tasks=missing, target_arm=arm, model_policy=policy, historical_model_key=s["effective_key"],
            init_evidence=init, historical_init_qualified=len(init) == len(DIAGNOSTIC_SEEDS),
            batch_requests=requests, unmeasured_npu_request=envelopes.get(mid, 8 * 2**30), phase_counts=work,
            evaluation=dict(previous=used, additional=future, limit=MI, remaining_after_plan=MI-used-future))
        outcomes.update(r["comparison"]["outcome"] for r in observed.values())
        totals.update(work)
        counts[mode] += 1
    return dict(schema="pilot-r3-early-subject-dispositions-v1", stage_b_gate="closed",
                scope="Recipe retention and bounded Stage A follow-up. One-step probes do not establish full trajectory or six-recipe equivalence.",
                primary_plan=active_ref, inventory=matrix["inventory"], original_inputs=inventory["references"],
                matrix=binding["matrix"], submissions=binding["submissions"], sources=sources,
                commits={k: matrix[k] for k in ("reference_commit", "weak_log_commit")},
                seeds=inventory["seeds"], target_exposures=dict(init=32768, a1=MI//4, a2=MI, a3=2*MI),
                source_probe_jobs=len(verified), comparison_points=sum(outcomes.values()),
                all_arm_comparisons=dict(all_comparisons), target_arm_comparisons=dict(outcomes),
                modes=dict(counts), phase_counts=dict(totals), subjects=rows)


def readable_table(review):
    modes = {"carry_recipe": "沿用旧配方", "compare_finalists": "重比两配方",
             "confirm_historical_winner": "复验旧赢家", "six_recipe_selection": "重做六配方"}
    gaps = {"incomplete_capacity_profile": "容量", "initialization": "初始化",
            "complete_a1_comparison": "A1", "complete_a2_comparison": "A2", "complete_a3_confirmation": "A3",
            "source_equivalence_on_winning_chain_and_competing_recipes": "旧选方已完成"}
    lines = ["# 早期 74 个 Stage A 主体的逐项复核", "",
             "沿用项保留旧选方和两份旧 A3 检查点的来源。补跑从目标源码的种子起点开始，A2、A3 接同版本的父检查点。"
             "每个进入补跑的主体重新执行三个初始化种子，并用两个固定种子比较或确认。全批 B 门禁关闭。"
             "[逐项原始证据、源码和预算](evidence/pilot-r3-deferred-dispositions-20261127.json)", "",
             "表中 A1、A2 边距分别为两个种子在入围线、胜出线的较小 NLL 差。单步一致数覆盖目标版本的损失、活动量、"
             "裁剪后梯度及后继状态，分母只含实际取得的比较点。缺失 A2 单列。单步匹配不能证明完整训练轨迹相同。"
             "[比较矩阵与缺失检查点](evidence/pilot-r3-signal-and-deferred-20261126.json)", "",
             "| 主体 | 旧进度或缺口 | A1 / A2 边距 | 单步一致 / 已测 | 缺失 A2 | 处置与依据 |",
             "|---|---|---:|---:|---:|---|"]
    for mid, row in sorted(review["subjects"].items()):
        ranks = row["recipe_rankings"]
        margins = [format(min(ranks[p]["seed_boundary_margins"]), ".4f") if p in ranks else "—" for p in ("a1", "a2")]
        values = Counter()
        for observation in row["observed"].values():
            values.update(observation["counts"])
        gap, mode = row["old_gap"], row["mode"]
        missing = len(row["missing_a2_tasks"])
        if mode == "carry_recipe":
            changed = row["observed"]["a1"]["counts"].get("sample_changed", 0)
            reason = f"两个种子的 A1 边界、A2 两强及 A3 观测均一致，另有 {changed} 个 A1 更新有差异。"
        elif mode == "confirm_historical_winner":
            reason = "两种子旧排名稳定，两个 A3 后继更新均一致，重新完成整段确认。"
        elif mode == "six_recipe_selection":
            reason = f"原{gaps[gap]}证据不完整，缺少六配方完整排名，按等量曝光重新选方。"
        elif missing:
            reason = f"两种子入围相同，{missing} 条 A2 缺少父状态，按配方和种子重建后比较。"
        elif values["update_error_to_updated"]:
            reason = "目标版完成了旧 CPU 失败批次的更新，须比较两个配方并确认完整 A3。"
        elif gap == "complete_a2_comparison":
            reason = "原 A1 两种子入围相同，原 A2 比较未完成。"
        elif gap == "complete_a3_confirmation":
            reason = "原 A3 尚未完成，已测后继更新有差异，先重比两强。"
        else:
            reason = "原选方完整，但决定入围或胜出的比较点有差异，须在目标版本重比。"
        recipes = "r0–r5" if len(row["recipes"]) == 6 else "、".join(row["recipes"])
        lines.append(f"| `{mid}` | {gaps[gap]} | {' / '.join(margins)} | "
                     f"{values['sample_exact']} / {sum(values.values())} | {missing} | **{modes[mode]} {recipes}**。{reason} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    result = make_review(read(args.launch))
    freeze_json(args.freeze, result)
    write_review_json(args.evidence, result)
    if args.table:
        args.table.write_text(readable_table(result))
    print(json.dumps({k: result[k] for k in ("modes", "phase_counts", "all_arm_comparisons", "target_arm_comparisons")}))


if __name__ == "__main__":
    main()
