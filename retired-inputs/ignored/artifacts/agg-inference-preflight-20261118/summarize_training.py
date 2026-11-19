import hashlib
import json
from pathlib import Path

from scripts.job import atomic

ROOT = Path.cwd()


def read(path):
    path = Path(path)
    value = json.loads(path.read_text())
    return value, dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def identity(directory, value, file):
    status, status_id = read(directory / "status.json")
    launch, launch_id = read(directory / "launch.json")
    return dict(record=file, source=value.get("source"), harness_source=value.get("harness_source"), state=value.get("state"),
                outer_state=status["state"], launch=launch_id, lifecycle=status_id,
                physical_devices=value.get("physical_devices"),
                cpu_affinity=value.get("cpu_affinity"), requested_cpu_affinity=launch["arguments"].get("affinity"),
                resource_identity=value.get("resource_identity"),
                arguments=value.get("arguments"), command=launch["arguments"]["command"])


record = dict(schema="tide-reception-training-summary-1", continuous=[], restore=[],
              modes=[], capacity=[], selected_comparisons=[], incomplete=[],
              qualification="finite completion and numerical equivalence are separate; no thresholds changed")
for directory in sorted(Path("artifacts").glob("agg-train-*-fp32-100-v1")):
    path = directory / "continuous.json"
    if not path.exists():
        record["incomplete"].append(str(directory)); continue
    value, file = read(path)
    if value["state"] != "completed":
        record["incomplete"].append(str(path)); continue
    rows = value["observations"]
    row = identity(directory, value, file)
    row.update(parameters=value["parameters"], spec=value["spec"],
               updates=len(rows), steady_rule="sum(input_positions)/sum(complete_seconds), updates 3 through 100",
               steady_positions_per_second=sum(x["input_positions"] for x in rows[2:]) / sum(x["complete_seconds"] for x in rows[2:]),
               peak_allocated_bytes=max(x["device_memory"]["peak_allocated_bytes"] for x in rows),
               peak_reserved_bytes=max(x["device_memory"]["peak_reserved_bytes"] for x in rows),
               all_finite=all(x["finite_output"] and x["finite_parameters"] and x["finite_gradients"] and
                   x["continuation"]["optimizer_finite"] and x["continuation"]["private_state_finite"] for x in rows),
               all_detached=all(x["continuation"]["private_state_detached"] for x in rows),
               final_continuation=rows[-1]["continuation"], final_state_position=rows[-1]["state_position"],
               checkpoint=value.get("checkpoint"), checkpoint_device_memory=value.get("checkpoint_device_memory"),
               timing=value["timing"], training_boundary=value["training_boundary"],
               initial_parameter_sha256=value["initial_parameters_sha256"], input_sha256=value["input_sha256"])
    record["continuous"].append(row)

for directory in sorted(Path("artifacts").glob("agg-restore-*-fp32-100-v1")):
    path=directory/"restore.json"
    if not path.exists():
        record["incomplete"].append(str(directory)); continue
    value,file=read(path)
    row=identity(directory,value,file)
    for key in ("checkpoint_source", "checkpoint_file", "load_complete_seconds", "next_update_complete_seconds",
                "next_update_input_positions_per_second", "load_peak_device_bytes", "next_update_peak_device_bytes",
                "loaded_state_position", "restored_state_position", "optimizer_step_increment_errors",
                "numerical_equivalence_qualification", "loaded_finite", "updated_finite"):
        row[key]=value.get(key)
    row["optimizer_steps_before"]=sorted(set(value.get("optimizer_before",{}).get("steps",{}).values()),key=str)
    row["optimizer_steps_after"]=sorted(set(value.get("optimizer_after",{}).get("steps",{}).values()),key=str)
    record["restore"].append(row)

for directory in sorted(Path("artifacts").glob("agg-modes-*-fp32-v1")):
    value,file=read(directory/"modes.json")
    if value["state"] != "completed_finite":
        record["incomplete"].append(str(directory)); continue
    row=identity(directory,value,file)
    row.update(config=value["config"], joint_matches=value["joint_matches_by_mode"],
               initial_audits={}, observations=[], timing=value["timing"], input_pairing=value["input_pairing"])
    for mode,audit in value["initial_audits"].items():
        row["initial_audits"][mode]={k:audit[k] for k in ("parameters","definition","deduplication","input_positions",
            "sample_time_slots","logical_axis","work","work_per_input_position","query_packing")}
        row["initial_audits"][mode]["effective_touched"]={k:v for k,v in audit["effective_touched"].items() if k!="by_sample_time"}
    for measured in value["observations"]:
        small={k:measured.get(k) for k in ("subject","mode","state","parameters","initial_parameter_sha256",
            "input_sha256","median_seconds","input_positions_per_second","seconds","device_memory",
            "sample_range_over_median","numerical_equivalence_qualification")}
        after=measured.get("post_timing_audit",{})
        small["post_work"]=after.get("work")
        small["post_effective_touched"]={k:v for k,v in after.get("effective_touched",{}).items() if k!="by_sample_time"}
        row["observations"].append(small)
    record["modes"].append(row)

for name in ("agg-fc-capacity-fp32-v1","agg-light64-capacity-fp32-v1","agg-light64-norecompute-fp32-v1","agg-half-capacity-fp32-v1"):
    directory=Path("artifacts")/name
    for path in sorted((directory/"search").glob("*-tide.json")):
        value,file=read(path)
        for measured in value["observations"]:
            row = identity(directory, value, file)
            row.update(state=measured["state"],
                config=value["config"],arguments=value["arguments"],
                parameters=measured.get("parameters"),seconds=measured.get("seconds"),
                median_seconds=measured.get("median_seconds"),positions_per_second=measured.get("input_positions_per_second"),
                device_memory=measured.get("device_memory"))
            record["capacity"].append(row)

directory = Path("artifacts/agg-half-selected-ad-fp32-v1")
search_path = directory / "search/search.json"
if search_path.exists():
    search, search_file = read(search_path)
    row = identity(directory, search, search_file)
    row.update(search_scope=search["search_scope"], matching_scope=search["matching_scope"],
               fixed_history=search["fixed_history"], budget_scope=search["budget_scope"],
               common_shape_table=search["common_shape_table"], children=search["children"],
               observations=[])
    if search["state"] != "completed":
        record["incomplete"].append(str(search_path))
    audit, audit_file = read(directory / "search/000-b4-l128-audit.json")
    activity = audit["initial_audits"]["train"]
    row.update(audit_record=audit_file, config=audit["config"], topology=audit["topology"],
               joint_matches=audit["joint_matches_by_mode"], input_pairing=audit["input_pairing"],
               matching_policy=audit["matching_policy"], precision=audit["precision"],
               initial_tide_parameter_sha256=audit["initial_tide_parameter_sha256"],
               audit_input_sha256=audit["audit_input_sha256"],
               initial_audit={k:activity[k] for k in ("parameters", "definition", "deduplication",
                   "input_positions", "sample_time_slots", "logical_axis", "work", "work_per_input_position", "query_packing")})
    row["initial_audit"]["effective_touched"] = {
        k:v for k,v in activity["effective_touched"].items() if k != "by_sample_time"}
    for path in sorted((directory / "search").glob("*-*.json")):
        if path.name.endswith("audit.json"):
            continue
        value, file = read(path)
        for measured in value.get("observations", []):
            small = {k:measured.get(k) for k in ("subject", "mode", "state", "parameters",
                "initial_parameter_sha256", "input_sha256", "median_seconds", "input_positions_per_second",
                "seconds", "device_memory", "sample_range_over_median", "numerical_equivalence_qualification")}
            after = measured.get("post_timing_audit", {})
            small.update(record=file, source=value["source"], timing=value["timing"],
                         post_work=after.get("work"),
                         post_effective_touched={k:v for k,v in after.get("effective_touched",{}).items()
                                                if k != "by_sample_time"})
            row["observations"].append(small)
    record["selected_comparisons"].append(row)

atomic("artifacts/agg-inference-preflight-20261118/training-summary.json",record)
print(json.dumps({k:len(v) for k,v in record.items() if isinstance(v,list)}))
