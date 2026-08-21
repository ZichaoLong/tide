"""Function-preserving Top-1 MoE used as the flat sparse-routing anchor."""

from __future__ import annotations

import copy
import dataclasses

import torch
from torch import nn
from torch.nn import functional as F


@dataclasses.dataclass
class MoEOutput:
    hidden: torch.Tensor
    balance_loss: torch.Tensor
    router_z_loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    routes: torch.Tensor
    probabilities: torch.Tensor


class Top1UpcycledMoE(nn.Module):
    """Replace one dense MLP with copied experts and no token dropping.

    Each token is dispatched to exactly one expert.  The selected weight is
    normalized to one, so identical copied experts preserve the source MLP's
    function at initialization.  The router is trained by Switch-style load
    balancing and router z-loss rather than by scaling the expert output.
    """

    def __init__(
        self,
        source_mlp: nn.Module,
        *,
        hidden_size: int,
        expert_count: int = 8,
        router_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if expert_count < 1:
            raise ValueError("expert_count must be positive")
        self.hidden_size = hidden_size
        self.expert_count = expert_count
        self.experts = nn.ModuleList(
            [source_mlp, *(copy.deepcopy(source_mlp) for _ in range(expert_count - 1))]
        )
        self.router = nn.Linear(hidden_size, expert_count, bias=False, dtype=torch.float32)
        nn.init.normal_(self.router.weight, mean=0.0, std=router_init_std)
        self._valid_tokens: torch.Tensor | None = None
        self.last_output: MoEOutput | None = None

    def configure_call(self, valid_tokens: torch.Tensor | None) -> None:
        self._valid_tokens = valid_tokens
        self.last_output = None

    def added_parameter_count(self) -> int:
        source_count = sum(parameter.numel() for parameter in self.experts[0].parameters())
        return sum(parameter.numel() for parameter in self.router.parameters()) + (
            self.expert_count - 1
        ) * source_count

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden [B,T,{self.hidden_size}], got {tuple(hidden.shape)}")
        batch_size, sequence_length, hidden_size = hidden.shape
        logits = self.router(hidden.float())
        probabilities = torch.softmax(logits, dim=-1)
        routes = probabilities.argmax(dim=-1)

        flat_hidden = hidden.reshape(-1, hidden_size)
        flat_routes = routes.reshape(-1)
        result = torch.zeros_like(flat_hidden)
        for expert_index, expert in enumerate(self.experts):
            positions = torch.nonzero(flat_routes == expert_index, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            selected = torch.index_select(flat_hidden, 0, positions)
            expert_output = expert(selected)
            result = torch.index_add(result, 0, positions, expert_output)

        valid_tokens = self._valid_tokens
        if valid_tokens is None:
            valid_tokens = torch.ones(
                batch_size, sequence_length, dtype=torch.bool, device=hidden.device
            )
        else:
            valid_tokens = valid_tokens[:, -sequence_length:].to(device=hidden.device, dtype=torch.bool)
        valid_count = valid_tokens.sum().clamp_min(1)
        one_hot_routes = F.one_hot(routes, num_classes=self.expert_count).to(torch.bool)
        active_counts = (one_hot_routes & valid_tokens.unsqueeze(-1)).sum(dim=(0, 1))
        mean_probability = (
            probabilities * valid_tokens.unsqueeze(-1)
        ).sum(dim=(0, 1)) / valid_count
        route_fraction = active_counts.float() / valid_count
        balance_loss = self.expert_count * (route_fraction.detach() * mean_probability).sum()
        log_partition = torch.logsumexp(logits, dim=-1)
        router_z_loss = (
            log_partition.square() * valid_tokens.to(log_partition.dtype)
        ).sum() / valid_count

        self.last_output = MoEOutput(
            hidden=result.reshape(batch_size, sequence_length, hidden_size),
            balance_loss=balance_loss,
            router_z_loss=router_z_loss,
            metrics={
                "active_counts": active_counts.detach(),
                "mean_probability": mean_probability.detach(),
            },
            routes=routes.detach(),
            probabilities=probabilities.detach(),
        )
        return self.last_output.hidden
