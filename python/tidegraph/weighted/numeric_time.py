"""Int64 differences first; half payload clocks/decay/trigonometry use FP32."""
import torch


def time_values(difference, like):
    return difference.to(torch.float32 if like.dtype == torch.float16 else like.dtype)


def passive_decay(difference, rates, like):
    clock = time_values(difference, like)
    rate = torch.as_tensor(rates, dtype=clock.dtype, device=like.device)
    return torch.exp(-clock*rate).to(like.dtype)
