"""Static node-owner validation; docs/resident-weighted-placement.md.

Availability and actual peer access are checked by the native owner. This
metadata helper neither probes hardware nor substitutes an execution path.
"""
import torch


def node_placement(requested, coordinator, count):
    coordinator = torch.device(coordinator)
    if coordinator.type not in {"cuda", "npu"} or coordinator.index is None:
        raise ValueError("weighted coordinator requires an indexed accelerator")
    if requested is None:
        return [coordinator] * count
    if not isinstance(requested, (list, tuple)) or len(requested) != count:
        raise ValueError("weighted placement needs one device per node")
    result = []
    for value in requested:
        if not isinstance(value, (str, torch.device)):
            raise ValueError("weighted node placement requires explicit devices")
        device = torch.device(value)
        if device.type != coordinator.type or device.index is None:
            raise ValueError("weighted node placement requires indexed same-backend devices")
        result.append(device)
    return result
