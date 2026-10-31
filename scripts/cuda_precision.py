"""Explicit IEEE policy for qualification; production library policy is unchanged."""
import os
import torch


def ieee_policy(device):
    if torch.device(device).type != "cuda":
        return None
    # Use one API family consistently, including runtime.manifest's getters.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                nvidia_tf32_override=os.environ.get("NVIDIA_TF32_OVERRIDE"),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
