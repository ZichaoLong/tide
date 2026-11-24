"""Round-scoped admission of one observed resident on specified physical cards."""
import math
from pathlib import Path
import re
import time
from uuid import UUID

from .scheduler import proc_identity


def validate_policy(policy, run_id, deadline):
    if not isinstance(policy, dict) or policy.get("run_id") != run_id or not isinstance(run_id, str) or not run_id:
        raise ValueError("temporary NPU placement requires its bound run ID")
    for key in ("candidate_ids", "shared_ids"):
        ids = policy.get(key)
        if (not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("temporary NPU placement requires unique physical " + key)
    if not set(policy["shared_ids"]) <= set(policy["candidate_ids"]):
        raise ValueError("shared NPU IDs must be included in the candidate pool")
    expiry = policy.get("expires_unix")
    if not isinstance(expiry, (int, float)) or not math.isfinite(expiry) or not 0 < expiry <= deadline:
        raise ValueError("resident authorization must end by the campaign deadline")
    UUID(policy["boot_id"])
    resident = policy.get("resident")
    if (not isinstance(resident, dict) or set(resident) != {"uid", "pid", "start_ticks"}
            or any(type(value) is not int for value in resident.values())
            or resident["uid"] < 0 or min(resident["pid"], resident["start_ticks"]) <= 0):
        raise ValueError("resident authorization requires UID, PID and start ticks")


def apply_reviewed_policy(config, review, run_id):
    # A later continuation needs a fresh grant. Restore the predecessor's
    # ordinary limit before applying any policy expressly bound to this run.
    previous = config.pop("temporary_npu", None)
    if previous:
        config.update(previous["ordinary_limits"])
    policy = review.get("temporary_npu")
    if policy is None:
        return
    from .cutover_review3 import PLANNED_CONTROLLER_CONTINUATION
    if review.get("schema") != PLANNED_CONTROLLER_CONTINUATION:
        raise ValueError("resident authorization requires a reviewed controller continuation")
    validate_policy(policy, run_id, config["deadline_unix"])
    ordinary = {key: config[key] for key in ("max_npu_devices", "target_npu_devices")}
    if policy.get("ordinary_limits") != ordinary:
        raise ValueError("temporary authorization differs from the ordinary NPU limits")
    count = len(policy["candidate_ids"])
    config.update(temporary_npu=policy, max_npu_devices=count, target_npu_devices=count)


def validate_lease(manifest):
    policy = manifest["temporary_npu"]
    match = re.fullmatch(r"(.+)-group-[0-9]+", manifest["job_id"])
    validate_policy(policy, match.group(1) if match else None, manifest["deadline_unix"])
    if (manifest["backend"] != "npu" or manifest.get("exclusive_hbm") or not manifest.get("dynamic_admission")
            or manifest.get("external_occupant_policy") != "continue_running"
            or manifest.get("candidate_ids") != policy["candidate_ids"]):
        raise ValueError("resident admission requires a dynamic shared lease within the frozen pool")


def resident_identity(pid):
    identity = proc_identity(pid)
    if identity is None:
        return None
    try:
        uid = Path(f"/proc/{pid}").stat().st_uid
    except OSError:
        return None
    return {"uid": uid, "pid": pid, "start_ticks": identity["start_ticks"]}


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def bind_queue_admission(helper, manifest):
    """Specialize only is_free on this lease's private in-memory helper module.

    Its original attempt_schedule still owns the FIFO, both physical samples,
    the locks and the running manifest. The driver process list stays intact.
    """
    policy = manifest["temporary_npu"]
    samples = {}
    same_boot = boot_id() == policy["boot_id"]
    parse = helper.parse_status

    def parse_status(raw, mapping):
        if "Process id" not in raw:
            raise ValueError("NPU process table is unavailable")
        return parse(raw, mapping)

    class ObservedCard(helper.DeviceStatus):
        def is_free(self, *, max_hbm_mib, max_aicore_percent):
            if self.physical_id not in policy["shared_ids"]:
                return super().is_free(max_hbm_mib=max_hbm_mib, max_aicore_percent=max_aicore_percent)
            observed = [resident_identity(pid) for pid in self.process_ids]
            history = samples.setdefault(self.physical_id, [])
            history.append({"observed_unix": time.time(), "processes": observed, "health": self.health,
                            "hbm_used_mib": self.hbm_used_mib, "aicore_percent": self.aicore_percent})
            if observed != history[0]["processes"]:
                return False
            if not observed:
                return super().is_free(max_hbm_mib=max_hbm_mib, max_aicore_percent=max_aicore_percent)
            return (same_boot and history[-1]["observed_unix"] < policy["expires_unix"]
                    and observed == [policy["resident"]] and self.health == "OK"
                    and self.hbm_used_mib <= max_hbm_mib and self.aicore_percent <= max_aicore_percent)

    helper.parse_status = parse_status
    helper.DeviceStatus = ObservedCard
    return samples


def worker_wait_reason(manifest, device, active):
    policy = manifest.get("temporary_npu")
    if not policy or device["physical_id"] not in policy["shared_ids"]:
        return None
    external = device.get("external_occupants", [])
    if not external:
        return None
    identity = policy["resident"]
    if (len(external) != 1 or any(external[0].get(k) != v for k, v in identity.items())
            or time.time() >= policy["expires_unix"] or boot_id() != policy["boot_id"]):
        return "resident_permission_changed"
    # With no group workers the utilization and HBM still belong to the
    # board/resident. While workers run, the combined HBM check applies.
    if not active and (device["used_bytes"] > manifest.get("admission_max_hbm_mib", 4096) * 2**20
            or device["aicore_percent"] > manifest.get("admission_max_aicore_percent", 5)):
        return "resident_pressure"
    return None
