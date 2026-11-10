"""Bounded finite guards for already shape/device-validated runtime tensors."""
from collections import defaultdict
import torch


def all_finite(values, *, budget=8*1024**2):
    if type(budget) is not int or budget < 1:
        raise ValueError("finite guard requires a positive byte budget")
    groups = defaultdict(list)
    for value in values:
        groups[value.device, value.dtype].append(value)
    with torch.no_grad():
        for tensors in groups.values():
            rows, size = [], 0
            for tensor in tensors:
                count = tensor.numel()*tensor.element_size()
                if rows and size+count > budget:
                    if not bool(torch.isfinite(torch.cat(rows)).all()):
                        return False
                    rows, size = [], 0
                if count > budget:
                    if not bool(torch.isfinite(tensor).all()):
                        return False
                else:
                    rows.append(tensor.reshape(-1)); size += count
            if rows and not bool(torch.isfinite(torch.cat(rows)).all()):
                return False
    return True
