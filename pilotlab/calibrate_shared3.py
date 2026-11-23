"""Run a finite, frozen set of explicitly sized independent worker groups."""
import argparse
from pathlib import Path
import signal

from .io import atomic_json, sha256
from .scheduler3 import Blocked, freeze_json, read
from .scheduler_shared3 import SharedSupervisor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = read(args.plan)
    for path, expected in plan["controller_source_files"].items():
        if sha256(path) != expected:
            raise ValueError("frozen calibration controller changed")
    supervisor = SharedSupervisor(plan["config"], args.output)
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    tasks = [task for group in plan["groups"] for task in group["tasks"]]
    try:
        if supervisor.state["tasks"]:
            raise Blocked("inspect_existing_calibration_instead_of_restarting")
        for task in tasks:
            freeze_json(args.output / "tasks" / task["id"] / "task.json", task)
            supervisor.state["tasks"][task["id"]] = {"status": "pending", "stage": "shared-calibration",
                "model_id": task["model_id"], "device": task["device"], "preparation": True,
                "attempts": [], "device_seconds": 0.}
        for group in plan["groups"]:
            reason = supervisor.resource_reason()
            if reason:
                raise Blocked(reason)
            supervisor.start_group(group["tasks"], group["backend"])
        results = supervisor.run("shared-calibration", tasks)
        if any(r.get("health", {}).get("eligible") is not True for r in results.values()):
            raise Blocked("calibration_worker_health_not_qualified")
        observations = []
        for group, record in zip(plan["groups"], supervisor.state["shared_groups"].values()):
            raw = read(Path(record["path"]) / "group.json")
            tokens = sum(results[t["id"]]["training"]["segment_tokens"] for t in group["tasks"])
            seconds = raw["physical_device_seconds"]
            observations.append({"label": group["label"], "backend": group["backend"],
                "requested_parallel_workers": len(group["tasks"]),
                "peak_parallel_workers": raw["peak_parallel_workers"],
                "processed_loss_tokens": tokens, "physical_device_seconds": seconds,
                "end_to_end_loss_tokens_per_card_second": tokens / seconds if seconds else None,
                "worker_training_rates": [results[t["id"]]["training"]["tokens_per_second"] for t in group["tasks"]],
                "group_path": str(Path(record["path"]) / "group.json"),
                "group_sha256": sha256(Path(record["path"]) / "group.json"),
                "evidence": [{"task_path": supervisor.state["tasks"][t["id"]]["task_path"],
                              "result_path": supervisor.state["tasks"][t["id"]]["result_path"]} for t in group["tasks"]]})
        atomic_json(args.output / "calibration.json", {"schema": "pilot-r3-shared-calibration-result-v1",
            "status": "passed", "groups": observations,
            "scope": "independent FP32 learning workers; different groups may run on different physical cards; includes startup/evaluation/checkpoint in card occupancy"})
        supervisor.finish("completed")
        return 0
    except BaseException as error:
        supervisor.stop_reason = supervisor.stop_reason or f"{type(error).__name__}: {error}"
        supervisor.stop_all()
        return 1
    finally:
        supervisor.close()


if __name__ == "__main__":
    raise SystemExit(main())
