"""Live resource preflight, distinct from static allocation-free planning."""
import os
import torch
from ..planning.capacity import capacity_budget
from .storage import filesystem, host_memory
from .checkpoint_policy import publication_plan


def host_available():
    return host_memory()["available_bytes"]


def preflight(resolved, output_parent, *, start_window=0):
    resources = dict(resolved["resources"])
    m = resolved["measurement"]
    resources["logical_windows"] = max(resources["logical_windows"], start_window+m["warmup"]+m["steps"]+int(m["checkpoint_check"]))
    host, storage = host_memory(), filesystem(output_parent)
    resources["host_free_bytes"] = host["available_bytes"]
    resources["disk_free_bytes"] = storage["free_bytes"]
    requested_medium = resources.get("storage_medium")
    effective_medium = "tmpfs" if storage["memory_backed"] else "disk"
    resources["storage_medium"] = effective_medium
    resources["devices"] = {key: dict(row) for key, row in resources.get("devices", {}).items()}
    e = resolved["execution"]["effective"]
    for name in e["peer_devices"] or (e["device"],):
        device = torch.device(name)
        if device.type == "cpu":
            continue
        api = getattr(torch, device.type)
        free, total = api.mem_get_info(device)
        row = resources["devices"].setdefault(name, {})
        row.update(free_bytes=int(free), capacity_bytes=int(total))
    budget = capacity_budget(resolved["plan"], resolved["execution"], placement=resolved["payload_placement"],
                mode=m["mode"], optimizer=resolved["task"]["optimizer"]["kind"], checkpoint=True, resources=resources,
                checkpoint_generations=publication_plan(m, start_window)["simultaneous_generations"])
    problems = list(budget["static_rejections"])
    if storage["memory_backed"] is None:
        problems.append("output filesystem: could not establish disk/tmpfs type; inspect the output mount before running")
    if requested_medium is not None and requested_medium != effective_medium:
        problems.append(f"resources.storage_medium: requested {requested_medium}, output mount is {effective_medium}; correct the destination or declared preview")
    for name, row in budget["per_device"].items():
        if row["reservation_exceeds_available"]:
            problems.append(f"{name}: estimated peak {row['estimated_peak_bytes']} exceeds current usable {row['usable_bytes']}")
    for name in ("host", "disk"):
        if budget[name].get("reservation_exceeds_available"):
            problems.append(name+": planned reservation exceeds live free memory/storage")
    budget.update(actual_preflight=True, load_average=list(os.getloadavg()),
                  storage=dict(storage, requested_medium=requested_medium, effective_medium=effective_medium,
                      durability="volatile; lost at container/host restart" if storage["memory_backed"] else "filesystem fsync publication; durability depends on storage"),
                  host_memory_snapshot=host,
                  checkpoint_publication=publication_plan(m, start_window),
                  cpu_affinity=len(os.sched_getaffinity(0)), refused=problems,
                  caveat="resource snapshots and conservative estimates; no runtime or correctness guarantee")
    return budget


def precision(device):
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    if device.type == "npu":
        api = torch.npu
        # Match the explicit IEEE-FP32 site policy; record both matmul/conv.
        if hasattr(api, "matmul"):
            api.matmul.allow_hf32 = False
        if hasattr(api, "conv"):
            api.conv.allow_hf32 = False
        return dict(matmul_allow_hf32=getattr(getattr(api, "matmul", None), "allow_hf32", None),
                    conv_allow_hf32=getattr(getattr(api, "conv", None), "allow_hf32", None))
    return dict(float32_matmul_precision=torch.get_float32_matmul_precision())
