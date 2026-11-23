"""Read-only compact viewer for the current round-three records."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .report3 import build_report


def render(report):
    exposure = report["exposure"]
    actual = report["concurrency"]
    lines = [f'round: {report["round"]}',
             f'status: {report["status"]}; stage: {report["stage"]}',
             f'updated_unix: {report["updated_unix"]}; snapshot_age_seconds: {report["snapshot_age_seconds"]}',
             f'blocking_reason: {report["blocking_reason"]}',
             f'gates: A={report["gates"]["stage_a"]["status"]}; B={report["gates"]["stage_b_recorded"]}; '
             f'formal_binding_verified={report["gates"]["formal"]["binding_verified"]}',
             f'tasks: {json.dumps(report["execution"]["task_counts"], sort_keys=True)}',
             f'formal completed subjects: {exposure["completed_subject_models"]}; '
             f'healthy: {exposure["completed_healthy_subject_models"]}; controls: {exposure["completed_control_models"]}',
             f'unique loss tokens: {json.dumps(exposure["unique_loss_tokens_by_category"], sort_keys=True)}',
             f'NPU conservative accounting intervals: peak={actual["peak"]}; live_open={actual["verified_live_open_allocations"]}; '
             f'card_hours={actual["device_hours"]:.6f}; accounting_complete={actual["accounting_complete"]}',
             f'NPU admitted lease intervals: peak={actual["admitted_leases"]["peak"]}; '
             f'card_seconds={actual["admitted_leases"]["device_seconds"]:.3f}; '
             f'timing_complete={actual["admitted_leases"]["timing_complete"]}; '
             f'group_debit_seconds={actual["recorded_group_device_seconds"]:.3f}; '
             f'conservative_overlap_seconds={actual["conservative_accounting_overlap_seconds"]:.3f}',
             f'worker process concurrency: logical_peak={actual["logical_workers"]["peak"]}; '
             f'NPU_worker_peak={actual["npu_workers"]["peak"]}; CPU_worker_peak={actual["cpu_workers"]["peak"]}; '
             f'shared_groups={actual["shared_group_count"]}',
             f'pending result bindings: {report["execution"]["pending_result_bindings"]}',
             f'selected models: {len(report["selected_recipes"])}; '
             f'state loads: {report["state_loading"]["status_counts"]}; '
             f'trajectory: {report["trajectory_reproduction"]["status_counts"]}']
    dispatch = report["resources"].get("dispatch_queue")
    if dispatch:
        waiting = [r for r in report["tasks"] if r["backend"] == "unbound" and r["scheduler_status"] == "pending"]
        models = {r["model_id"] for r in waiting}
        calibrating = models.intersection(dispatch.get("awaiting_calibration", ()))
        probes = Counter(r["scheduler_status"] for r in report["tasks"] if r["stage"] == "dispatch-calibration")
        lines.append(f'A1 shared queue  {len(models)} unbound groups ({len(waiting)} tasks), '
                     f'{len(calibrating)} awaiting paired calibration. Probe states {dict(probes)}')
    if report["selected_recipes"]:
        choices = {key: dict(Counter(str(row.get(key)) for row in report["selected_recipes"]))
                   for key in ("batch_size", "lr", "cpu_threads", "resolved_initialization")}
        lines.append("selected recipe choices: "+json.dumps(choices, ensure_ascii=False, sort_keys=True))
    lines.append(f'training loss-token/s: {report["performance"]["training_loss_tokens_per_second"]}; '
                 f'numerical health: {report["numerical_health"]["eligibility_counts"]}')
    lines.append(f'CPU loss tokens={report["performance"]["cpu_loss_tokens"]}; '
                 f'CPU training token/s={report["performance"]["cpu_training_loss_tokens_per_second"]}; '
                 f'CPU aggregate token/s={report["performance"]["aggregate_cpu_loss_tokens_per_worker_window_second"]}')
    ledger = report["resources"].get("budget_ledger") or {}
    for group in report["resources"].get("shared_groups", []):
        if group.get("backend") != "npu" or group.get("status") not in {"running", "queued", "preparing", "cleanup_blocked"}:
            continue
        policy = group.get("sharing_policy") or {}
        device = (group.get("latest_resources") or {}).get("device") or {}
        lines.append(f'NPU owner {group["id"]}: card={group.get("physical_id")}; '
            f'running={group.get("running_workers")}; pending_inside_owner={group.get("pending_workers")}; '
            f'dynamic={group.get("dynamic_admission")}; admission_limit={policy.get("admission_limit")}; '
            f'HBM_bytes={device.get("used_bytes")}/{device.get("total_bytes")}; '
            f'external_processes={len(device.get("external_occupants", []))}')
    if ledger:
        lines.append(f'global device_seconds={ledger.get("device_seconds")}; '
                     f'max_device_seconds={ledger.get("max_device_seconds")}; deadline_unix={ledger.get("deadline_unix")}')
    for row in report["tasks"]:
        if row["scheduler_status"] in {"running", "pending", "failed", "stopped_budget", "interrupted"}:
            heartbeat = row["live_heartbeat"]
            lines.append(f'{row["id"]}: {row["scheduler_status"]}; '
                         f'tokens={heartbeat.get("tokens", row["cumulative_loss_tokens"])}/{row["target_loss_tokens"]}; '
                         f'reason={row["reason"]}')
    for problem in report["observation_errors"]:
        lines.append(f'observation_error: {problem["path"]}: {problem["reason"]}')
    lines.append("Read-only observation; no training, gate rerun, or scientific success inferred from exit codes.")
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args.round)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) if args.json else render(report), end="\n" if args.json else "")
    return 2 if report["status"] == "unavailable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
