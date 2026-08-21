"""Reference receiver group used by the first TIDE experiments."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


PROFILES = {"selected-dispatch", "bo"}
IMPLEMENTATIONS = {"dense-masked-reference", "packed"}
SCAN_IMPLEMENTATIONS = {"reference", "vectorized"}


def ema_scan_reference(
    observations: torch.Tensor,
    receive_mask: torch.Tensor,
    decay: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    clear_positions: Sequence[int] = (),
) -> torch.Tensor:
    """Literal token-by-token EMA semantics.

    Shapes are ``observations=[B,T,R,S]``, ``receive_mask=[B,T,R]``,
    ``decay=[R,S]``, and ``initial_state=[B,R,S]``.
    """

    state = initial_state
    states: list[torch.Tensor] = []
    clear = set(clear_positions)
    for token_index in range(observations.shape[1]):
        if token_index in clear:
            state = torch.zeros_like(state)
        candidate = decay.unsqueeze(0) * state + (1.0 - decay).unsqueeze(0) * observations[:, token_index]
        state = torch.where(receive_mask[:, token_index].unsqueeze(-1), candidate, state)
        states.append(state)
    if not states:
        return observations.new_empty(
            observations.shape[0], 0, observations.shape[2], observations.shape[3]
        )
    return torch.stack(states, dim=1)


def ema_scan_vectorized(
    observations: torch.Tensor,
    receive_mask: torch.Tensor,
    decay: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Equivalent affine prefix scan for the fixed-length training path."""

    expanded_decay = decay.unsqueeze(0).unsqueeze(0)
    receive = receive_mask.unsqueeze(-1)
    coefficient = torch.where(receive, expanded_decay, torch.ones_like(expanded_decay))
    addition = torch.where(
        receive,
        (1.0 - expanded_decay) * observations,
        torch.zeros_like(observations),
    )
    prefix = torch.cumprod(coefficient, dim=1)
    # With the documented initialization (lambda=0.99, T=512), prefix is far
    # from underflow.  We intentionally do not clamp it: a later unstable
    # setting must fail the finite-value gate instead of changing semantics.
    accumulated = torch.cumsum(addition / prefix, dim=1)
    return prefix * (initial_state.unsqueeze(1) + accumulated)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


@dataclasses.dataclass
class ReceiverOutput:
    hidden: torch.Tensor
    final_state: torch.Tensor
    balance_loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    routes: torch.Tensor | None = None
    probabilities: torch.Tensor | None = None
    states: torch.Tensor | None = None


class TideReceiverGroup(nn.Module):
    """Four fixed receivers with content routing and per-sequence EMA state."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        state_size: int = 128,
        receiver_count: int = 4,
        profile: str = "bo",
        implementation: str = "packed",
        scan_implementation: str = "vectorized",
        rms_norm_eps: float = 1e-6,
        ffn_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if profile not in PROFILES:
            raise ValueError(f"unknown profile: {profile}")
        if implementation not in IMPLEMENTATIONS:
            raise ValueError(f"unknown implementation: {implementation}")
        if scan_implementation not in SCAN_IMPLEMENTATIONS:
            raise ValueError(f"unknown scan implementation: {scan_implementation}")
        if receiver_count < 1 or state_size < 1:
            raise ValueError("receiver_count and state_size must be positive")

        self.hidden_size = hidden_size
        self.state_size = state_size
        self.receiver_count = receiver_count
        self.profile = profile
        self.implementation = implementation
        self.scan_implementation = scan_implementation
        self.rms_norm_eps = rms_norm_eps

        self.norm_weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.router = nn.Linear(hidden_size, receiver_count, bias=False, dtype=torch.float32)
        self.observers = nn.ModuleList(
            nn.Linear(hidden_size, state_size, bias=True, dtype=torch.float32)
            for _ in range(receiver_count)
        )
        initial_logit = math.log(0.99 / 0.01)
        self.decay_logits = nn.Parameter(
            torch.full((receiver_count, state_size), initial_logit, dtype=torch.float32)
        )
        self.state_projections = nn.ModuleList(
            nn.Linear(state_size, hidden_size, bias=False, dtype=torch.float32)
            for _ in range(receiver_count)
        )
        self.ffns = nn.ModuleList(
            SwiGLUFFN(hidden_size, intermediate_size).to(dtype=ffn_dtype)
            for _ in range(receiver_count)
        )
        for ffn in self.ffns:
            nn.init.zeros_(ffn.down_proj.weight)

    def zero_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.receiver_count,
            self.state_size,
            dtype=torch.float32,
            device=device,
        )

    def _normalize(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden_fp32 = hidden.float()
        variance = hidden_fp32.square().mean(dim=-1, keepdim=True)
        return hidden_fp32 * torch.rsqrt(variance + self.rms_norm_eps) * self.norm_weight

    def _scan(
        self,
        observations: torch.Tensor,
        receive_mask: torch.Tensor,
        initial_state: torch.Tensor,
        clear_positions: Sequence[int],
    ) -> torch.Tensor:
        decay = torch.sigmoid(self.decay_logits)
        if self.scan_implementation == "reference" or clear_positions:
            return ema_scan_reference(
                observations,
                receive_mask,
                decay,
                initial_state,
                clear_positions=clear_positions,
            )
        return ema_scan_vectorized(observations, receive_mask, decay, initial_state)

    def _dense_masked(
        self,
        message: torch.Tensor,
        states: torch.Tensor,
        routes: torch.Tensor,
        probabilities: torch.Tensor,
        read_state: bool,
    ) -> torch.Tensor:
        result = torch.zeros_like(message, dtype=self.ffns[0].down_proj.weight.dtype)
        for receiver_index in range(self.receiver_count):
            state_delta = self.state_projections[receiver_index](states[:, :, receiver_index])
            if not read_state:
                state_delta = torch.zeros_like(state_delta)
            ffn_input = (message + state_delta).to(self.ffns[receiver_index].down_proj.weight.dtype)
            delta = self.ffns[receiver_index](ffn_input)
            weight = probabilities[:, :, receiver_index] * (routes == receiver_index)
            result = result + delta * weight.unsqueeze(-1).to(delta.dtype)
        return result

    def _packed(
        self,
        message: torch.Tensor,
        states: torch.Tensor,
        routes: torch.Tensor,
        probabilities: torch.Tensor,
        read_state: bool,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = message.shape
        flat_message = message.reshape(-1, hidden_size)
        flat_routes = routes.reshape(-1)
        flat_probabilities = probabilities.reshape(-1, self.receiver_count)
        result = message.new_zeros(
            batch_size * sequence_length,
            hidden_size,
            dtype=self.ffns[0].down_proj.weight.dtype,
        )
        for receiver_index in range(self.receiver_count):
            positions = torch.nonzero(flat_routes == receiver_index, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            selected_message = torch.index_select(flat_message, 0, positions)
            flat_state = states[:, :, receiver_index].reshape(-1, self.state_size)
            selected_state = torch.index_select(flat_state, 0, positions)
            state_delta = self.state_projections[receiver_index](selected_state)
            if not read_state:
                state_delta = torch.zeros_like(state_delta)
            ffn_input = (selected_message + state_delta).to(
                self.ffns[receiver_index].down_proj.weight.dtype
            )
            delta = self.ffns[receiver_index](ffn_input)
            selected_probability = torch.index_select(flat_probabilities, 0, positions)[
                :, receiver_index
            ]
            weighted = delta * selected_probability.unsqueeze(-1).to(delta.dtype)
            result = torch.index_add(result, 0, positions, weighted)
        return result.reshape(batch_size, sequence_length, hidden_size)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        valid_tokens: torch.Tensor | None = None,
        read_state: bool = True,
        clear_positions: Sequence[int] = (),
        return_artifacts: bool = False,
    ) -> ReceiverOutput:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden [B,T,{self.hidden_size}], got {tuple(hidden.shape)}")
        batch_size, sequence_length, _ = hidden.shape
        if initial_state is None:
            initial_state = self.zero_state(batch_size, hidden.device)
        expected_state = (batch_size, self.receiver_count, self.state_size)
        if tuple(initial_state.shape) != expected_state:
            raise ValueError(f"expected initial state {expected_state}, got {tuple(initial_state.shape)}")
        if initial_state.dtype != torch.float32 or initial_state.device != hidden.device:
            initial_state = initial_state.to(device=hidden.device, dtype=torch.float32)

        message = self._normalize(hidden)
        logits = self.router(message)
        probabilities = torch.softmax(logits, dim=-1)
        routes = probabilities.argmax(dim=-1)
        one_hot_routes = F.one_hot(routes, num_classes=self.receiver_count).to(torch.bool)

        if valid_tokens is None:
            valid_tokens = torch.ones(
                batch_size, sequence_length, dtype=torch.bool, device=hidden.device
            )
        else:
            valid_tokens = valid_tokens.to(device=hidden.device, dtype=torch.bool)
        if self.profile == "bo":
            receive_mask = valid_tokens.unsqueeze(-1).expand(-1, -1, self.receiver_count)
        else:
            receive_mask = one_hot_routes & valid_tokens.unsqueeze(-1)

        observations = torch.stack(
            [torch.tanh(observer(message)) for observer in self.observers],
            dim=2,
        )
        states = self._scan(observations, receive_mask, initial_state, clear_positions)

        if self.implementation == "dense-masked-reference":
            delta = self._dense_masked(message, states, routes, probabilities, read_state)
        else:
            delta = self._packed(message, states, routes, probabilities, read_state)
        delta = delta.to(hidden.dtype) * valid_tokens.unsqueeze(-1).to(hidden.dtype)

        valid_count = valid_tokens.sum().clamp_min(1)
        mean_probability = (
            probabilities * valid_tokens.unsqueeze(-1)
        ).sum(dim=(0, 1)) / valid_count
        balance_loss = ((mean_probability - 1.0 / self.receiver_count) ** 2).sum()
        active_counts = (
            one_hot_routes & valid_tokens.unsqueeze(-1)
        ).sum(dim=(0, 1))
        receive_counts = receive_mask.sum(dim=(0, 1))
        metrics = {
            "active_counts": active_counts.detach(),
            "receive_counts": receive_counts.detach(),
            "mean_probability": mean_probability.detach(),
            "state_norm": states.detach().float().norm(dim=-1).mean(),
        }
        return ReceiverOutput(
            hidden=hidden + delta,
            final_state=states[:, -1].detach() if sequence_length else initial_state.detach(),
            balance_loss=balance_loss,
            metrics=metrics,
            routes=routes.detach() if return_artifacts else None,
            probabilities=probabilities.detach() if return_artifacts else None,
            states=states.detach() if return_artifacts else None,
        )
