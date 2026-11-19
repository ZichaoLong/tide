"""Publish the final decode-capacity evidence only after all release gates pass.

Standard library only: no model, device, audit refresh, or experiment execution.
Both output files are staged, fsynced and atomically replaced. --check-only runs
the same qualification and fingerprint gates without creating output files.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
STEM = "decode-capacity-20261118"
MATH = "4f5b9f20a4030d68ee66ae2d26e34659a5e37234"
GIB = 2 ** 30
EXPECTED = {f"{scale}/{kind}/{cards}" for scale, cards in (("9b", 1), ("17b", 2), ("17b", 4))
            for kind in ("tide", "tf5", "tf32", "tf48")}
PHASE = {"confirm": "本轮确认", "screen": "本轮筛选", "historical": "历史参考"}


class GateError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GateError(message)


def relative(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def file_identity(path):
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return dict(path=relative(path), bytes=size, sha256=digest.hexdigest())


def same_file(left, right):
    return bool(left and right and left["bytes"] == right["bytes"] and left["sha256"] == right["sha256"])


class Snapshot:
    """Remember every inspected file; check it again immediately before writing."""
    def __init__(self):
        self.files = {}

    def read(self, path):
        path = path.resolve()
        content = path.read_bytes()
        record = dict(path=relative(path), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        self.remember(path, record)
        return json.loads(content), record

    def remember(self, path, record):
        path = path.resolve()
        previous = self.files.get(path)
        require(previous is None or same_file(previous, record), f"file changed while reading: {path}")
        self.files[path] = record

    def verify(self, expected):
        path = (ROOT / expected["path"]).resolve()
        actual = file_identity(path)
        require(same_file(expected, actual), f"stale evidence fingerprint: {expected['path']}")
        self.remember(path, actual)

    def recheck(self):
        for path, expected in self.files.items():
            require(same_file(expected, file_identity(path)), f"evidence changed before publication: {path}")


def load_records(args, snapshot):
    records, identities = {}, {}
    for key in ("summary", "release", "preflight", "decisions", "controller", "constructors"):
        records[key], identities[key] = snapshot.read(getattr(args, key))
    for key in ("audit", "submissions"):
        expected = records["summary"]["inputs"][key]
        records[key], identities[key] = snapshot.read(ROOT / expected["path"])
        require(same_file(expected, identities[key]), f"summary has stale {key} identity")
    return records, identities


def validate_records(records, identities, snapshot):
    s, release, preflight = records["summary"], records["release"], records["preflight"]
    audit, decisions = records["audit"], records["decisions"]
    require(s.get("schema") == "tide-decode-capacity-summary-1", "unsupported summary schema")
    require(audit.get("schema") == "tide-decode-capacity-audit-1", "unsupported audit schema")
    require(release.get("schema") == "tide-decode-capacity-release-1", "missing/unsupported terminal release audit")
    plan_keys = {f"{p['scale']}/{p['kind']}/{p['cards']}" for p in preflight["plan"]["series"]}
    require(plan_keys == EXPECTED and len(preflight["plan"]["series"]) == len(EXPECTED), "preflight must cover the exact 12 planned series")
    require(s["plan"] == preflight["plan"], "summary/preflight plan mismatch")
    require(s["plan"]["schedule"] == dict(prefill=128, prefill_chunk=32, decode=16)
            and s["plan"]["allocated_budget_gib"] == 48 and s["plan"]["reserved_budget_gib"] == 56,
            "report is scoped to the fixed short request and 48/56 GiB budgets")
    for flag in ("all_terminal", "all_current_points_audited", "all_series_have_confirmed_winner",
                 "all_series_manually_finished_with_eligible_selection", "independent_audit_manifest_matches_current"):
        require(s.get(flag) is True, f"summary gate not passed: {flag}")
    require(s["validation_error_count"] == 0 and not s["validation_errors"] and not s["manual_decision_issues"], "summary validation errors remain")
    require(audit["summary"]["audit_state"] == "passed" and audit["summary"]["validation_errors"] == 0
            and audit["summary"]["all_terminal"] is True and not audit["summary"]["execution_failures"], "independent raw audit is not fully passed")
    require(s["independent_audit_summary"] == audit["summary"] and s["independent_input_checks"] == audit["inputs"], "summary has stale audit/input results")
    inputs = audit["inputs"]
    require(inputs.get("requested") is True and inputs.get("all_checked") is True and bool(inputs["groups"]), "CPU reconstruction must be requested and all checked")
    for group in inputs["groups"]:
        require(group["state"] == "passed" and not group.get("errors") and group["common_prefix_width"] == 768
                and group["common_prefix_sha256"] == group["expected_prefix_sha256"], "CPU input/prefix reconstruction failed")
    require(set(decisions) == EXPECTED, "all 12 series must have final decisions")
    series = {p["series"]: p for p in s["series"]}
    require(set(series) == EXPECTED and len(s["series"]) == len(EXPECTED), "summary series coverage mismatch")
    byname = {p["name"]: p for p in s["runs"]}
    audited = {p["name"]: p for p in audit["runs"]}
    submitted = {p["name"]: p for p in records["submissions"]}
    require(len(byname) == len(s["runs"]) == len(audited) == len(audit["runs"]) == len(submitted)
            == len(records["submissions"]) and set(byname) == set(audited) == set(submitted), "run identities/coverage mismatch")
    rebuilt_names = [name for group in inputs["groups"] for name in group["runs"]]
    require(len(rebuilt_names) == len(set(rebuilt_names)) and set(rebuilt_names) == set(byname), "CPU reconstruction does not uniquely cover all submitted points")
    require(same_file(audit["submissions"], identities["submissions"]), "audit/submission manifest mismatch")
    for key, summary_key in (("preflight", "preflight"), ("decisions", "manual_decisions")):
        require(same_file(s["inputs"][summary_key], identities[key]), f"summary has stale {key}")
    for record in s["inputs"].values():
        if record:
            snapshot.verify(record)
    classes = Counter()
    for name, point in byname.items():
        independent = audited[name]
        require(point["submission"] == independent["submission"] == submitted[name], f"submission mismatch: {name}")
        require(point["terminal"] is True and point["independent_audit"]["current"] is True
                and point["independent_audit"]["input_verification"] == independent["input_verification"] == "passed", f"point not fully audited: {name}")
        require(not point["validation_errors"] and not independent["audit_errors"], f"point audit errors: {name}")
        require(point["classification"] == independent["classification"]
                and point["classification"] in ("completed_within_budget", "sampled_budget_refusal"), f"unresolved execution outcome: {name}")
        require(point["files"] == independent["files"], f"raw inventory mismatch: {name}")
        for identity in point["files"].values():
            snapshot.verify(identity)
        rebuilt = point["input_reconstruction"]
        require(rebuilt and rebuilt["state"] == "passed" and rebuilt["full_sha256"] == point["result"]["input_sha256"], f"missing exact reconstructed input: {name}")
        require(point["result"]["arguments"]["precision"] == "fp32", f"unexpected precision: {name}")
        classes[point["classification"]] += 1
    require(dict(classes) == s["counts"]["classifications"] == audit["summary"]["classes"], "classification counts disagree")
    for key, row in series.items():
        decision, winner = decisions[key], row["confirmed_winner"]
        require(decision["state"] == "finished" and row["manual_decision"] == decision, f"series not finished: {key}")
        require(row["manual_decision_validation"]["state"] == "finished_with_eligible_selection"
                and row["manual_decision_validation"]["selection_is_highest_confirmed"] is True, f"manual selection is not the qualified winner: {key}")
        require(winner and winner["name"] == decision["winner"], f"manual/confirmed winner mismatch: {key}")
        point = byname[winner["name"]]
        require(point["series"] == key and point["eligible_confirmation"] is True and point["phase"] == "confirm"
                and winner["metrics"] == point["result"]["metrics"], f"invalid current confirmation: {key}")
        require(winner["sampling"]["warmup"] == 2 and winner["sampling"]["normal_samples"] == 5, f"wrong formal sample protocol: {key}")
        eligible = [p for p in s["runs"] if p["series"] == key and p["eligible_confirmation"]]
        actual = min(eligible, key=lambda p: (-p["result"]["metrics"]["median_stable_decode_positions_per_second"], p["batch"], p["name"]))
        require(actual["name"] == winner["name"], f"winner is not highest confirmed throughput: {key}")
        require(all(p["allocated"] <= preflight["plan"]["allocated_budget_gib"] * GIB
                    and p["reserved"] <= preflight["plan"]["reserved_budget_gib"] * GIB
                    for p in winner["metrics"]["peak_bytes_by_device"].values()), f"winner exceeded a per-card budget: {key}")
        require("final_state_memory" in winner["metrics"], f"summary requires final-state memory metrics: {key}")
    constructors = {(c["scale"], c["kind"], c["cards"]): c for c in records["constructors"]}
    require(len(constructors) == len(records["constructors"]) == 9, "missing/duplicate Transformer constructors")
    for row in s["series"]:
        if row["kind"] == "tide":
            continue
        point = byname[row["confirmed_winner"]["name"]]
        constructor = constructors[(row["scale"], row["kind"], row["cards"])]
        spec = point["result"]["spec"]
        require(all(spec[k] == v for k, v in constructor["match"]["constructor"].items())
                and constructor["match"]["actual"]["parameters"] == point["result"]["parameters"], "constructor differs from executed model: " + row["series"])
    expected_pairs = {(scale, cards, kind) for scale, cards in (("9b", 1), ("17b", 2), ("17b", 4)) for kind in ("tf5", "tf32", "tf48")}
    for table, predicate in (("same_batch", lambda p: p["tide"]["batch"] == 8),
                             ("independently_selected_budget_best", lambda p: True)):
        pairs = [p for p in s["comparisons"][table] if predicate(p)]
        require(len(pairs) == len(expected_pairs) and {(p["scale"], p["cards"], p["transformer_kind"]) for p in pairs} == expected_pairs, "incomplete comparison table: " + table)
        for pair in pairs:
            require(isinstance(pair["tide_over_transformer_throughput"], (int, float))
                    and math.isfinite(pair["tide_over_transformer_throughput"]) and pair["tide_over_transformer_throughput"] > 0,
                    "unaligned comparison: " + table)
            require(all(pair["checks"][key] is True for key in ("same_card_count", "same_schedule", "same_requested_precision", "same_seed")), "comparison workload mismatch: " + table)
            if table == "same_batch":
                require(pair["checks"]["same_batch"] is True and pair["checks"]["shared_768_prefix_equal"] is True,
                        "same-B8 comparison lacks equal batch/common-prefix evidence")
            if table == "independently_selected_budget_best":
                require(pair["both_manually_finished"] is True and pair["both_manual_selections_match_highest_confirmed"] is True, "budget-best pair not final")
    for point in s["historical_references"]:
        require(point["eligible_historical_reference"] is True and not point["validation_errors"], "historical reference not independently retained: " + point["name"])
        for record in point["files"].values():
            snapshot.verify(record)
    lifecycle = {r["name"]: r for r in release["lifecycle"]}
    require(set(lifecycle) == set(byname) and len(lifecycle) == len(release["lifecycle"]), "terminal release coverage mismatch")
    require(release["upstream"]["state"] == "passed" and release["upstream"]["revision"] == MATH, "fixed mathematics verification failed")
    require(release["production_tree"] == preflight["production_tree"], "production source tree changed")
    require(0 <= release["finite_execution_elapsed_seconds"] <= preflight["plan"]["maximum_project_wall_seconds"], "finite search exceeded wall-time limit")
    for name, row in lifecycle.items():
        require(row["properties"]["MainPID"] == "0" and row["properties"]["ActiveState"] in ("inactive", "failed")
                and not row["live_recorded_descendants"] and not row["source_mismatches"]
                and row["frozen_regular_files"] > 0, f"release is not terminal and source-verified: {name}")
        require(row["state"] == byname[name]["outer_state"] and row["source"] == submitted[name]["source"]
                and row["queue"]["status"] == ("completed" if byname[name]["classification"] == "completed_within_budget" else "failed")
                and row["queue"]["return_code"] == row["status"]["exit_code"]
                and row["queue"]["npu_count"] == submitted[name]["cards"], f"terminal release/measurement mismatch: {name}")
    require(release["counts"] == dict(Counter(r["state"] for r in lifecycle.values())), "release outcome counts disagree")
    require(release["raw_file_count"] == len(release["raw_files"]), "release raw-file count mismatch")
    require(set(release["source"]) == {p["source"] for p in submitted.values()}, "release source coverage mismatch")
    released_raw = {r["path"]: r for r in release["raw_files"]}
    require(len(released_raw) == len(release["raw_files"]), "duplicate release raw identities")
    for point in byname.values():
        for identity in point["files"].values():
            require(same_file(released_raw.get(identity["path"]), identity), "release omitted/changed raw file: " + identity["path"])
    released_inputs = {r["path"]: r for r in release["input_records"]}
    for key in ("summary", "preflight", "decisions", "audit", "submissions"):
        require(same_file(released_inputs.get(identities[key]["path"]), identities[key]), "release predates final " + key)
    for group in ("raw_files", "input_records", "tools", "helpers"):
        for record in release[group]:
            snapshot.verify(record)
    protected = release["protected_user_files"]
    require(all(r["unchanged"] is True for r in protected) and len(protected) == len(preflight["user_files"]), "protected user-file audit failed")
    original_user_files = {r["path"]: r for r in preflight["user_files"]}
    for record in protected:
        require(same_file(original_user_files.get(record["path"]), record), "protected user-file identity differs from preflight")
        snapshot.verify(record)
    return dict(state="passed", planned_series=len(series), current_runs=len(byname), classifications=dict(classes),
                cpu_input_groups=len(inputs["groups"]), historical_references=len(s["historical_references"]),
                release_raw_files=release["raw_file_count"], source=release["source"], production_tree=release["production_tree"],
                upstream=release["upstream"], lifecycle_scope="verified release snapshot of MainPID and recorded PID/start descendants; no new process query by report generator")


def ordered(rows):
    return sorted(rows, key=lambda r: (int(r["scale"][:-1]), r["cards"], {"tide": 0, "tf5": 1, "tf32": 2, "tf48": 3}.get(r.get("transformer_kind", r.get("kind")), 9), r.get("batch", 0), r.get("name", "")))


def model_name(kind):
    names = {"tide": "Tide", "tf5": "Transformer C5", "tf32": "Transformer 32层", "tf48": "Transformer 48层"}
    require(kind in names, "unknown model kind for rendering: " + str(kind))
    return names[kind]


def series_label(row):
    return f"{row['scale'].upper()} / {row['cards']}卡 / {model_name(row.get('transformer_kind', row.get('kind')))}"


def number(value, places=3):
    return "—" if value is None else f"{value:.{places}f}"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def point_identity(point):
    sample = point["sampling"]
    return (f"{PHASE[point['phase']]} w{sample['warmup']}/s{sample['normal_samples']}; "
            f"`{point['source']['source_commit'][:8]}`; 卡{point['physical_devices']}")


def peak_text(metric):
    return "; ".join(f"{device}={pair['allocated']/GIB:.3f}/{pair['reserved']/GIB:.3f}"
                     for device, pair in metric["peak_bytes_by_device"].items())


def stop_description(reason):
    if "adjacent" in reason:
        return "相邻整数 B 的采样预算边界"
    if "two consecutive" in reason:
        return "连续两次吞吐增益≤2%"
    if "eight-screen" in reason:
        return "有限筛选点数上限"
    if "confirmation stage" in reason:
        return "已进入人工选择的确认阶段"
    return reason


def refusal_detail(point):
    breaches = {}
    for breach in point["independent_audit"]["budget_breaches"]:
        key = (breach["device"], breach["kind"])
        old = breaches.get(key)
        if old is None or breach["observed_bytes"] > old["observed_bytes"]:
            breaches[key] = breach
    return dict(name=point["name"], batch=point["batch"], phase=point["phase"],
                maximum_breaches=list(breaches.values()), all_breaches=point["independent_audit"]["budget_breaches"],
                error=point["result"]["error"], normal_samples_retained=point["result"]["sampling"]["normal_samples"])


def build_payload(records, identities, qualification):
    s = records["summary"]
    byname = {p["name"]: p for p in s["runs"]}
    shapes, winners, searches, activity = [], [], [], []
    constructors = {(c["scale"], c["kind"], c["cards"]): c for c in records["constructors"]}
    for row in ordered(s["series"]):
        point = byname[row["confirmed_winner"]["name"]]
        result, winner = point["result"], row["confirmed_winner"]
        shape = dict(series=row["series"], scale=row["scale"], kind=row["kind"], cards=row["cards"],
                     parameters=result["parameters"], spec=result["spec"], placement=result["placement"])
        if row["kind"] != "tide":
            shape["constructor_record"] = constructors[(row["scale"], row["kind"], row["cards"])]
        else:
            shape["matching_depth"] = result["matching_reference"]["matches"]["C"]["targets"]["blocks"]
        shapes.append(shape)
        winners.append(dict(series=row["series"], scale=row["scale"], kind=row["kind"], **winner))
        current = [p for p in s["runs"] if p["series"] == row["series"]]
        old = [p for p in s["historical_references"] if p["series"] == row["series"]]
        passing = sorted({p["batch"] for p in current if p["classification"] == "completed_within_budget"})
        rejected = [refusal_detail(p) for p in current if p["classification"] == "sampled_budget_refusal"]
        upper = min((r["batch"] for r in rejected if r["batch"] > max(passing)), default=None)
        searches.append(dict(series=row["series"], scale=row["scale"], kind=row["kind"], cards=row["cards"],
            historical_batches=sorted({p["batch"] for p in old}),
            screen_batches=sorted({p["batch"] for p in current if p["phase"] == "screen"}),
            confirmed_batches=sorted({p["batch"] for p in current if p["eligible_confirmation"]}),
            passing_batches=passing, refusals=rejected,
            adjacent_passing_batch=max(passing) if upper == max(passing) + 1 else None,
            adjacent_refused_batch=upper if upper == max(passing) + 1 else None,
            selected_batch=winner["batch"], decision=records["decisions"][row["series"]]))
    for point in ordered([p for p in s["runs"] if p["kind"] == "tide" and p["eligible_confirmation"]]):
        result = point["result"]
        activity.append(dict(name=point["name"], series=point["series"], scale=point["scale"], kind=point["kind"],
            cards=point["cards"], batch=point["batch"], selected=any(w["name"] == point["name"] for w in winners),
            audit=result["activity_audit"], diagnostics=result["diagnostics"], final_state_memory=result["metrics"]["final_state_memory"],
            activity_diagnostics=result["matching_reference"].get("diagnostics"),
            activity_matching_boundary=result["matching_reference"].get("matching_boundary"),
            source=result["source"], physical_devices=result["physical_devices"], timing_scope="activity replay covers entire request; not a stable-decode-only cost profile"))
    commands = [dict(name=p["name"], series=p["series"], phase=p["phase"], batch=p["batch"],
                source=p["submission"]["source"], submission_argv=p["submission"]["argv"],
                frozen_payload_argv=p["launch"]["arguments"]["command"], runtime_argv=p["launch"]["command"],
                source_path=p["launch"]["source_path"], files=p["files"])
                for p in s["runs"]]
    return dict(schema="tide-decode-capacity-evidence-1", recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        generator=file_identity(Path(__file__)), qualification=qualification, inputs=identities,
        scope="finite FP32 fixed-model/request per-card-budget batch search; headless backbone, not language-model TTFT; no new A/B/D maximum or training/recovery qualification",
        plan=records["preflight"]["plan"], model_shapes=shapes, confirmed_winners=winners,
        same_batch_8=[p for p in s["comparisons"]["same_batch"] if p["tide"]["batch"] == 8],
        independently_selected_budget_best=s["comparisons"]["independently_selected_budget_best"],
        searches=searches, tide_formal_activity=activity,
        historical_batch_1=[p for p in s["historical_references"] if p["batch"] == 1],
        reproduction=dict(current_runs=commands,
            historical_policy="original source/arguments/raw fingerprints retained under records.summary.historical_references; no relabelling as current confirmation",
            rerun_policy="use fresh launch name and output paths under the recorded frozen revision; preserve device admission, time/RSS and per-card memory budgets"),
        records=records)


def render_markdown(report):
    s, release = report["records"]["summary"], report["records"]["release"]
    plan, q = report["plan"], report["qualification"]
    schedule = plan["schedule"]
    stable = schedule["decode"] - 1
    normal = s["counts"]["normal_samples_by_classification"]
    sources = ", ".join("`" + source + "`" for source in release["source"])
    tide_settings = sorted({", ".join(f"{key}={row['spec'][key]}" for key in
        ("aggregation", "memory", "allocation", "sharing", "region_batch_size", "ssd_scan_backend", "ssd_summary_backend", "parameter_indexing"))
        for row in report["model_shapes"] if row["kind"] == "tide"})
    lines = [f"# 2026-11-18：固定显存预算下的 decode batch 搜索\n",
        f"{q['planned_series']} 个预定系列均已结束并选出正式确认点。当前 {q['current_runs']} 个作业中，"
        f"{q['classifications'].get('completed_within_budget', 0)} 个通过预算、{q['classifications'].get('sampled_budget_refusal', 0)} 个被采样预算拒绝；"
        f"预算内保留 {normal.get('completed_within_budget', 0)} 个正常计时样本，拒绝点另保留 {normal.get('sampled_budget_refusal', 0)} 个样本，后者不参与最佳点选择。"
        f"历史 {s['counts']['historical_references']} 个参考点、{s['counts']['historical_normal_samples']} 个样本保持原身份。\n",
        f"本轮源码 {sources}；生产 `src/tide` 树 `{release['production_tree']}`。"
        f"完整命令、构造器、样本、原始文件指纹、CPU 输入审计、搜索决策及终态核对见[同名 JSON]({STEM}.json)。\n",
        "## 共同口径与模型形状\n",
        f"全部 FP32，prompt {schedule['prefill']}、prefill chunk {schedule['prefill_chunk']}、decode {schedule['decode']}。"
        f"请求从零态开始，请求内真实续接 KV/SSD/pending；稳定 decode 排除第一次增量，分母是余下 {stable} 次的同步耗时，分子是 {stable}×B 个输入位置。"
        "吞吐为各正常样本吞吐的中位数，平均步 ms 为各样本稳定耗时/调用数的中位数。每卡吞吐是总吞吐/参与卡数，不代表各卡负载相等。\n",
        f"筛选 warmup{plan['screen']['warmup']}/samples{plan['screen']['samples']}，正式确认 warmup{plan['confirmation']['warmup']}/samples{plan['confirmation']['samples']}；"
        "不混合两种样本总体。正常计时使用常驻模型及已在设备的输入，排除构造、初始哈希、输入传输、热身和诊断；跨卡执行中的 host-stage 传输包含在请求计时内，所有参与卡同步。"
        "没有任务头、词元采样、网络或服务队列；这里是主干输入位置/s，首主干输出也不是语言模型 TTFT。\n",
        f"每卡 allocated/reserved 采样预算为 {plan['allocated_budget_gib']}/{plan['reserved_budget_gib']} GiB，"
        "是留余量的实验边界，不是硬件容量极限或硬隔离。峰值取各卡各正常请求峰的最大值，不跨卡相加。"
        "协作队列独占准入，主机及外部任务仍不受控制；实际物理卡、主机负载和线程身份逐点保留。\n",
        "C5 是原近深度×近总参数 C 构造器；32/48 层为指定深度后近总参数的补充模型。"
        "它们不替代[四类联合匹配](../benchmark-contract.md)。改变深度同时改变宽度、FFN 和头宽，不能把性能差归结为纯层数效应；"
        "构造器搜索的是近参数形状，并未搜索全局最快 Transformer。\n"]
    rows = []
    for row in report["model_shapes"]:
        spec = row["spec"]
        if row["kind"] == "tide":
            topology = spec["topology"]
            shape = f"HB {topology['timing']}, n{topology['n']}+h{topology['hubs']}, {topology['profile']}; 图深度{row['matching_depth']}"
            dimensions, deviation = str(spec["d"]) + "/区域本地绑定/—", "目标本身"
        else:
            shape = f"{spec['layers']}层, {spec['heads']}头, KV窗口{spec['window']}"
            dimensions = f"{spec['d']}/{spec['ffn_width']}/{spec['head_dim']}"
            deviation = f"{row['constructor_record']['match']['relative_deviation']['parameters'] * 100:+.6f}%"
        rows.append([series_label(row), shape, dimensions, number(row["parameters"] / 1e9, 6), deviation])
    lines += [table(["规模/设备/模型", "结构", "d/FFN/头宽", "总参数 B", "参数偏差"], rows),
        "Tide 的实际执行配置为 " + "；".join("`" + settings + "`" for settings in tide_settings) + "。输入 B 与区域合批是两个轴。"
        "多卡 Tide 按完整区域放置并 per_device 并发；Transformer 按连续完整层放置，跨卡均 host-stage，没有副本并行。"
        "每个参数分区处理同一组 B 个样本的所属区域或层，B 未按设备平分，不是每卡 B/卡数的独立微批。"
        "Transformer 保留融合 QKV、融合 gate/up、真实 KV 和 auto Attention；本轮没有穷举 attention、重算、packing、结构或副本候选。\n",
        "## 各系列预算内最佳正式点\n",
        "最佳指本次有限搜索中、独立审计通过的正式确认点里吞吐最高者。人工停止决定与资格校验分开；不把最大可容纳 B 自动视为吞吐最高 B。\n"]
    rows = [[series_label(w), w["batch"], number(w["metrics"]["median_stable_decode_positions_per_second"]),
             number(w["metrics"]["median_mean_step_ms"], 2), number(w["metrics"]["per_card_positions_per_second"]),
             peak_text(w["metrics"]), w["physical_devices"], number(100*w["metrics"]["range_over_median"], 2)]
            for w in report["confirmed_winners"]]
    lines += [table(["系列", "B", "总位置/s", "平均步 ms", "每卡位置/s", "各逻辑卡峰 allocated/reserved GiB", "物理卡（按逻辑序）", "样本极差/中位数 %"], rows),
        "总吞吐增大可伴随单请求步时延增加。每卡峰值是整条短请求的采样峰，未把常驻权重、最终状态和瞬时工作空间简单等同。\n"]
    alternatives = []
    for w in report["confirmed_winners"]:
        winning_rate = w["metrics"]["median_stable_decode_positions_per_second"]
        for point in s["runs"]:
            if (point["series"] != w["series"] or not point["eligible_confirmation"]
                    or point["name"] == w["name"]):
                continue
            metric = point["result"]["metrics"]
            rate = metric["median_stable_decode_positions_per_second"]
            alternatives.append([series_label(w), point["batch"], number(rate),
                number(100 * (rate / winning_rate - 1), 4), peak_text(metric)])
    if alternatives:
        lines += ["其他正式确认点如下。中位数排名用于有限选择，"
                  "不是显著性检验；差距很小时可结合显存余量选择，不应宣称确定的性能优势。\n",
                  table(["系列", "另一正式 B", "总位置/s", "相对入选点 %", "各卡峰 allocated/reserved GiB"], alternatives)]
    latency_rows = []
    for w in report["confirmed_winners"]:
        m = w["metrics"]
        latency_rows.append([series_label(w), w["batch"],
            number(1000*m["first_any_backbone_output_seconds"], 2),
            number(m["prompt_output_ready_seconds"]),
            number(1000*m["first_incremental_call_seconds"], 2),
            number(m["median_p95_step_ms"], 2)])
    lines += ["## 热态请求中的首输出、prompt 与增量时延\n",
        table(["最佳正式点", "B", "首批主干输出 ms", "完整prompt就绪 s", "首次增量 ms", "稳定增量样本内p95 ms"], latency_rows),
        "首批主干输出在第一个32位置chunk之后即可出现，不等于128位置prompt全部处理完成。"
        "这些是同一批请求的主干时间，模型与输入已常驻；不含模型加载或语言模型任务头。"
        "吞吐最佳点不自动成为首输出或单序列decode延迟最佳点；历史B1身份另列，不能据本表推断服务队列TTFT。\n",
        "## 同 B8 与各自预算内最佳\n",
        "同 B8 回答固定输入批量的表现；各自预算最佳回答同一每卡预算下各模型选择自身 B 后的总吞吐。后者 B 可不同，不能称为同负载加速。"
        "代表点按本轮确认→本轮筛选→历史参考选择，同阶段优先更多样本，再按名称；不会为提高比值挑最快一次。\n"]
    rows = []
    for p in ordered(report["same_batch_8"]):
        rows.append([series_label(p), number(p["tide"]["metrics"]["median_stable_decode_positions_per_second"]),
                     number(p["transformer"]["metrics"]["median_stable_decode_positions_per_second"]),
                     number(p["tide_over_transformer_throughput"]), point_identity(p["tide"]), point_identity(p["transformer"])])
    lines.append(table(["规模/设备/对照", "Tide B8位置/s", "Transformer B8位置/s", "Tide/TF", "Tide 身份", "Transformer 身份"], rows))
    four = {p["tide"]["name"]: p["tide"] for p in report["same_batch_8"] if p["cards"] == 4}
    if four:
        lines.append("四卡 Tide B8 的实际身份为 " + "；".join(f"`{p['name']}`：{point_identity(p)}" for p in four.values()) + "。\n")
    lines.append("表内 w/s 分别为 warmup/正常样本数；Transformer B8 的筛选点若为 s2，就只代表两样本筛选，并未升级为五样本确认。"
                 "历史四卡 Tide 和本轮 Transformer 不冒充同轮交错配对。源码、物理卡、样本数及共同 768 维输入前缀逐对检查保存在 JSON；同前缀不代表模型语义或逐值轨迹相同。\n")
    rows = [[series_label(p), p["tide"]["batch"], p["transformer"]["batch"],
             number(p["tide"]["metrics"]["median_stable_decode_positions_per_second"]),
             number(p["transformer"]["metrics"]["median_stable_decode_positions_per_second"]), number(p["tide_over_transformer_throughput"])]
            for p in ordered(report["independently_selected_budget_best"])]
    lines += [table(["规模/设备/对照", "Tide B", "TF B", "Tide位置/s", "TF位置/s", "Tide/TF（各自B）"], rows),
        "两张表都只能比较记录中的实现与请求，不认证全局最优，也不是隔离调度器后的纯执行器加速。不同 B 会改变独立样本集合和 Tide 平均活动量；不能假定活动轨迹保持不变。\n",
        "## 有限搜索、停止原因与预算拒绝\n"]
    rows = []
    for row in report["searches"]:
        refused = []
        for refusal in row["refusals"]:
            detail = "; ".join(f"{b['kind']} {b['device']} {b['observed_bytes']/GIB:.3f}>{b['limit_bytes']/GIB:g}GiB"
                               for b in refusal["maximum_breaches"])
            refused.append(f"B{refusal['batch']}: {detail}")
        adjacent = (f"B{row['adjacent_passing_batch']}→B{row['adjacent_refused_batch']}"
                    if row["adjacent_passing_batch"] is not None else "未形成相邻拒绝边界")
        rows.append([series_label(row), ",".join(map(str,row["historical_batches"])) or "—",
                     ",".join(map(str,row["screen_batches"])) or "—", ",".join(map(str,row["confirmed_batches"])),
                     "；".join(refused) or "无", adjacent + "; " + stop_description(row["decision"]["stop_reason"])])
    lines += [table(["系列", "保留历史 B", "本轮筛选 B", "正式确认 B", "拒绝点及触发预算", "边界/停止依据"], rows),
        f"每系列最多 {plan['maximum_new_screens_per_series']} 个新筛选点、{plan['maximum_confirmations_per_series']} 个正式确认点，"
        f"单点预算 {plan['maximum_point_seconds']} 秒、整体 {plan['maximum_project_wall_seconds']/3600:g} 小时、最多同时 {plan['maximum_concurrent_cards']} 卡。"
        f"实际提交至最后作业结束的计入墙钟为 {release['finite_execution_elapsed_seconds']/60:.2f} 分钟。"
        "采样预算拒绝以实际 allocated/reserved 越界记录为依据，与硬件 OOM 分开；拒绝前已测样本保留但不入选。"
        "达到相邻整数边界也只约束该实现、短请求和预算，不能外推硬件极限、长上下文容量或所有整数 B 的全局吞吐最优。"
        "完整预估、人工和自动控制决策保留在 JSON。\n",
        "本轮还观察到 reserved 峰值不单调：17B 双卡48层的 B32/B49 筛选点分别为47.314/43.098 GiB，"
        "allocated 则由38.205增至41.988 GiB，因此该系列继续扩大 batch。两个点的物理卡和 allocator 历史各自保留，"
        "本轮没有 trace 将下降归因到某个具体内核或分配机制。其他系列在相邻 reserved 拒绝处停止，"
        "也不排除更大 B 再次落入预算的可能；本轮没有穷举这种非单调区间。\n",
        "## Tide 正式点的活动、执行工作与 packing\n",
        "这些统计来自单独的完整请求审计，覆盖 prompt 与全部 decode，不是稳定 decode 的时间剖析。"
        "实际触及参数按样本/输入时刻、在 settled 层间去重；时间和 batch 并集可覆盖全部参数，但不等于每步全部活跃。"
        "Full 比例和不同 B 下的活动量按实际记录展示，不把高活动本身视为实现错误；参数触及计数也不是 DRAM 读字节。\n"]
    rows, workrows, packrows, executionrows = [], [], [], []
    for point in report["tide_formal_activity"]:
        a = point["audit"]
        label = series_label(point) + f" / B{point['batch']}" + ("（选中）" if point["selected"] else "（另一确认点）")
        for scope, diagnostic in (("普通独立诊断", point["diagnostics"]), ("matching reference 活动审计", point["activity_diagnostics"])):
            execution = (diagnostic or {}).get("execution") or {}
            executionrows.append([label, scope, execution.get("requested_region_concurrency", "未记录"),
                                  execution.get("effective_region_concurrency", "未记录"), execution.get("hooks", "未记录")])
        effective, work = a["effective_touched"], a["work_per_input_position"]
        rows.append([label, number(effective["mean"]/1e9, 6), number(100*effective["mean"]/a["parameters"]["unique_parameters"], 2),
                     number(effective["min"]/1e9, 6) + "/" + number(effective["max"]/1e9, 6),
                     number(effective["unique_across_time_and_batch"]/1e9, 6), a["input_positions"]])
        workrows.append([label, number(work["full_actions"]), number(work["positive_messages"]) + "/" + number(work["message_slots"]),
                         number(work["candidate_updates"]) + "/" + number(work["candidate_slots"]),
                         number(work["regional_attention_positive_pairs"]), number(work["kv_attention_effective_pairs"])])
        for family, packed in a["query_packing"].items():
            packrows.append([label, family, packed["calls"], packed["selected_rows"], packed["executed_rows"],
                             number(packed["executed_rows"]/packed["selected_rows"]) if packed["selected_rows"] else "—"])
    lines += [table(["正式点", "诊断边界", "请求区域并发", "实际区域并发", "hooks 记录"], executionrows),
        "并发/串行诊断身份按原始 execution 字段逐项列出；如果记录为 serial_region_audit，就只认证该审计执行。"
        "即使诊断记录为 per_device，也不把单独 replay 的活动计数当作每个正常计时样本逐活动轨迹已相等的证明。\n",
              table(["正式点", "触及均值 B", "均值/总参数 %", "逐位置最小/最大 B", "时间与batch并集 B", "审计输入位置"], rows),
              table(["正式点", "Full/位置", "正消息/消息槽 每位置", "候选更新/候选槽 每位置", "区域正attention对/位置", "KV有效attention对/位置"], workrows),
              table(["正式点", "packing族", "调用数", "选择行", "执行行", "执行/选择"], packrows),
        "packing 执行行含 padding。这些工作量可解释为什么低活跃参数并不按比例消除接收、准备、调度和填充工作，但本轮未新增分段 trace，不能把吞吐差精确归因到某个耗时占比。"
        "已有公式、伪代码、分配与 packing 轨迹见[性能归因](reception-performance-20261118.md)，其具体形状和采样边界仍适用。\n",
        "## 最终状态存储与请求峰值分开\n",
        "下表是正常请求结束后 private state 的唯一 retained storage（包含 view tail），按所有设备求总量；不是任何时刻的峰值，也不是单卡量。"
        "除以 B 只是包括共享元数据的算术归一化，不是单独测出的逐序列 allocation。权重、临时投影/合批工作区和 allocator reserved 差异不能由这一终态总量单独推得。"
        "独立诊断 replay 的对应值另列，未与正常样本混合。\n"]
    rows = []
    for w in report["confirmed_winners"]:
        memory = w["metrics"]["final_state_memory"]
        rows.append([series_label(w), w["batch"], number(memory["median_total_retained_storage_bytes"]/GIB, 4),
                     number(memory["median_retained_storage_bytes_per_batch_item"]/GIB, 4),
                     number(memory["diagnostic_total_retained_storage_bytes"]/GIB, 4),
                     ",".join(map(str, sorted(set(memory["normal_sample_final_state_positions"]))))])
    lines += [table(["最佳正式点", "B", "全设备终态总量 GiB", "终态总量/B GiB", "独立诊断总量 GiB", "最终输入位置"], rows),
        "## 历史 B1 仅保留原身份\n",
        "这些历史短请求没有在本轮重新测量。原 Tide/C5 与补充 32/48 层的源码、warmup/samples 和物理卡可能不同；不把它们加入当前确认排名，也不改称当前环境下的配对重测。\n"]
    rows = []
    for point in ordered(report["historical_batch_1"]):
        r = point["result"]
        rows.append([series_label(point), number(r["metrics"]["median_stable_decode_positions_per_second"]),
                     number(r["metrics"]["median_mean_step_ms"],2), f"{r['sampling']['warmup']}/{r['sampling']['normal_samples']}",
                     "`"+r["source"]["source_commit"][:8]+"`", r["physical_devices"]])
    lines += [table(["历史系列 B1", "位置/s", "平均步 ms", "warmup/samples", "原源码", "原物理卡"], rows),
        "原始记录及先前 batch/多卡限制见[深度与 batch 补测](transformer-depth-batch-20261118.md)。\n",
        "## 审计、范围与复现\n",
        f"{q['current_runs']} 个当前测点的原始身份、20 次调用、0→144 输入时钟、稳定吞吐分母、有限性、参与卡及每卡预算全部完成独立审计；"
        f"CPU 对 {q['cpu_input_groups']} 组输入独立重建完整哈希与共同 768 维前缀，覆盖全部当前点。"
        f"终态 release 核验 {q['release_raw_files']} 份原始文件指纹、所有服务 MainPID=0、已记录 PID/start 后代无存活、冻结源码文件与提交一致，"
        f"固定数学 `{MATH}` 的 {release['upstream']['files']} 个文件通过，受保护用户文件未改变。"
        "这是记录时刻的终态核验；报告生成器只校验该 release 快照及文件新鲜度，不重新查询系统服务。\n",
        "本次只完成固定 FP32 短请求的 batch 搜索。既有大模型长 decode、真实缓存与首输出边界见[推理证据](reception-inference-20261118.md)；"
        "短 prompt128/chunk32/decode16 不能继承为更长历史的容量/吞吐结论。"
        "既有 FP32/BF16 超差和有界数值诊断仍见[数值证据](reception-numerics-20261118.md)及[验证/后端范围](reception-validation-20261118.md)。"
        "有限值与吞吐完成不替代不同实现逐值/梯度等价证明，不新增四卡训练、恢复、其他后端、HCCL/peer DMA 或全模型 LibTorch 资格。"
        "本轮也没有最大吞吐 A/B/D、新任务效果、全局候选/结构搜索或副本并行测量。\n",
        "复现从 JSON 的 `reproduction.current_runs` 选择测点，使用其 `source` 冻结提交、`submission_argv` 或 `frozen_payload_argv`；"
        "必须换新作业名及输出目录，保留预算、设备准入、线程/RSS/时间限制。模型入口为 `benchmarks.inference`（Tide）与 `benchmarks.transformer_depth`（Transformer），"
        "沿用[执行契约](../execution-flows.md)。来源、实际 argv、初始参数/完整输入哈希、每样本 timing/allocator、逻辑到物理卡映射均在 JSON 的 `records.summary.runs`。\n",
        "从保留的终态记录重新校验并生成证据、不重新运行模型或覆盖原输入审计：\n",
        "```sh\npython artifacts/decode-capacity-20261118/write_report.py --check-only\npython artifacts/decode-capacity-20261118/write_report.py\n```\n",
        "首次审计的冻结命令在 JSON 的 `records.release.auxiliary_lifecycle`；"
        "`release.py` 还核验本轮四小时执行窗口，因此事后复核应使用保存的 release 及文件指纹，"
        "不重写时间或覆盖原审计来制造新的终态资格。需要独立重建输入时使用新输出目录和有界 CPU 作业。\n",
        "生成器要求全部系列完成、有最高正式确认点、人工选择与资格一致、原始审计及 CPU 输入检查全部通过、release 终态覆盖全部测点且指纹仍匹配；任一条件未满足就拒绝写入。\n"]
    return "\n".join(lines)


def atomic_outputs(directory, outputs):
    directory.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for name, content in outputs:
            fd, temporary = tempfile.mkstemp(prefix=".decode-report.", dir=directory)
            staged.append((temporary, directory/name))
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, target in staged:
            os.replace(temporary, target)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        for temporary, _ in staged:
            if os.path.exists(temporary):
                os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, filename in (("summary", "summary.json"), ("release", "release-audit.json"),
                           ("preflight", "preflight.json"), ("decisions", "series-decisions.json"),
                           ("controller", "controller-decisions.json"), ("constructors", "constructors.json")):
        parser.add_argument("--" + name, type=Path, default=BASE/filename)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"docs/evidence")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    writing = False
    try:
        snapshot = Snapshot()
        records, identities = load_records(args, snapshot)
        qualification = validate_records(records, identities, snapshot)
        report = build_payload(records, identities, qualification)
        snapshot.verify(report["generator"])
        markdown = render_markdown(report).encode("utf-8")
        report["markdown"] = dict(path=relative(args.output_dir/(STEM+".md")), bytes=len(markdown), sha256=hashlib.sha256(markdown).hexdigest())
        machine = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        snapshot.recheck()
        if not args.check_only:
            for suffix in (".json", ".md"):
                require((args.output_dir/(STEM+suffix)).resolve() not in snapshot.files, "output would overwrite an input")
            writing = True
            atomic_outputs(args.output_dir, [(STEM+".json", machine), (STEM+".md", markdown)])
        print(json.dumps(dict(state="passed", check_only=args.check_only, **{k:v for k,v in qualification.items() if k != "state"},
                              output_directory=relative(args.output_dir), markdown_bytes=len(markdown), json_bytes=len(machine)), ensure_ascii=False))
        return 0
    except (GateError, OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        print(json.dumps(dict(state="refused", outputs_written="possibly_partial" if writing else False,
                              error=f"{type(exc).__name__}: {exc}"), ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
