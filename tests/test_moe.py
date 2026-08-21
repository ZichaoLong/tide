from __future__ import annotations

import copy
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch import nn
from torch.nn import functional as F

from tide.moe import Top1UpcycledMoE


class TinySwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(16, 32, bias=False)
        self.up_proj = nn.Linear(16, 32, bias=False)
        self.down_proj = nn.Linear(32, 16, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


def test_upcycled_top1_preserves_function_dispatches_every_token_and_trains_router() -> None:
    torch.manual_seed(37)
    source = TinySwiGLU()
    expected = copy.deepcopy(source)
    moe = Top1UpcycledMoE(source, hidden_size=16, expert_count=4)
    hidden = torch.randn(4, 32, 16, requires_grad=True)

    actual = moe(hidden)
    torch.testing.assert_close(actual, expected(hidden), atol=2e-6, rtol=2e-6)
    assert moe.last_output is not None
    assert moe.last_output.metrics["active_counts"].sum().item() == 128

    loss = actual.square().mean() + 0.01 * moe.last_output.balance_loss
    loss = loss + 0.001 * moe.last_output.router_z_loss
    loss.backward()
    assert moe.router.weight.grad is not None
    assert moe.router.weight.grad.norm() > 0
    assert sum(
        expert.down_proj.weight.grad is not None for expert in moe.experts
    ) >= 2
