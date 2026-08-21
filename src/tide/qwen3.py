"""Qwen3 integration for TIDE receiver groups."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .moe import MoEOutput, Top1UpcycledMoE
from .receiver import ReceiverOutput, TideReceiverGroup


@dataclasses.dataclass
class TideCausalLMOutput:
    loss: torch.Tensor | None
    lm_loss: torch.Tensor | None
    balance_loss: torch.Tensor
    router_z_loss: torch.Tensor
    logits: torch.Tensor
    past_key_values: Any
    states: dict[int, torch.Tensor]
    metrics: dict[int, dict[str, torch.Tensor]]
    artifacts: dict[int, ReceiverOutput | MoEOutput]


class TideWrappedDecoderLayer(nn.Module):
    def __init__(self, original_layer: nn.Module, receiver_group: TideReceiverGroup) -> None:
        super().__init__()
        self.original_layer = original_layer
        self.receiver_group = receiver_group
        self._initial_state: torch.Tensor | None = None
        self._valid_tokens: torch.Tensor | None = None
        self._read_state = True
        self._clear_positions: tuple[int, ...] = ()
        self._shuffle_positions: tuple[int, ...] = ()
        self._return_artifacts = False
        self.last_output: ReceiverOutput | None = None

    def configure_call(
        self,
        *,
        initial_state: torch.Tensor | None,
        valid_tokens: torch.Tensor | None,
        read_state: bool,
        clear_positions: Sequence[int],
        shuffle_positions: Sequence[int],
        return_artifacts: bool,
    ) -> None:
        self._initial_state = initial_state
        self._valid_tokens = valid_tokens
        self._read_state = read_state
        self._clear_positions = tuple(clear_positions)
        self._shuffle_positions = tuple(shuffle_positions)
        self._return_artifacts = return_artifacts
        self.last_output = None

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        hidden_states = self.original_layer(hidden_states, *args, **kwargs)
        valid_tokens = self._valid_tokens
        if valid_tokens is not None and valid_tokens.shape[1] != hidden_states.shape[1]:
            valid_tokens = valid_tokens[:, -hidden_states.shape[1] :]
        self.last_output = self.receiver_group(
            hidden_states,
            initial_state=self._initial_state,
            valid_tokens=valid_tokens,
            read_state=self._read_state,
            clear_positions=self._clear_positions,
            shuffle_positions=self._shuffle_positions,
            return_artifacts=self._return_artifacts,
        )
        return self.last_output.hidden


class TideQwen3ForCausalLM(nn.Module):
    """Thin owner of a Hugging Face Qwen3 model and its TIDE call state."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        layer_indices: Sequence[int],
        profile: str,
        receiver_count: int = 4,
        expert_count: int = 8,
        state_size: int = 128,
        implementation: str = "packed",
        scan_implementation: str = "vectorized",
        balance_coefficient: float = 0.01,
        router_z_coefficient: float = 0.001,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.profile = profile
        self.balance_coefficient = balance_coefficient
        self.router_z_coefficient = router_z_coefficient
        self.layer_indices = tuple(sorted(set(layer_indices)))
        self.wrapped_layers: dict[int, TideWrappedDecoderLayer] = {}
        self.moe_layers: dict[int, Top1UpcycledMoE] = {}

        layers = self.base_model.model.layers
        for layer_index in self.layer_indices:
            if layer_index < 0 or layer_index >= len(layers):
                raise ValueError(f"layer index {layer_index} is outside 0..{len(layers) - 1}")
            if profile == "upcycled-moe":
                moe = Top1UpcycledMoE(
                    layers[layer_index].mlp,
                    hidden_size=base_model.config.hidden_size,
                    expert_count=expert_count,
                    router_init_std=base_model.config.initializer_range,
                )
                layers[layer_index].mlp = moe
                self.moe_layers[layer_index] = moe
                continue
            if profile not in {"selected-dispatch", "bo"}:
                raise ValueError(f"unknown non-dense profile: {profile}")
            group = TideReceiverGroup(
                hidden_size=base_model.config.hidden_size,
                intermediate_size=base_model.config.intermediate_size,
                state_size=state_size,
                receiver_count=receiver_count,
                profile=profile,
                implementation=implementation,
                scan_implementation=scan_implementation,
                rms_norm_eps=base_model.config.rms_norm_eps,
                ffn_dtype=next(layers[layer_index].parameters()).dtype,
            )
            wrapped = TideWrappedDecoderLayer(layers[layer_index], group)
            layers[layer_index] = wrapped
            self.wrapped_layers[layer_index] = wrapped

    @property
    def config(self) -> Any:
        return self.base_model.config

    def extension_parameter_ids(self) -> set[int]:
        receiver_ids = {
            id(parameter)
            for wrapped in self.wrapped_layers.values()
            for parameter in wrapped.receiver_group.parameters()
        }
        router_ids = {
            id(parameter)
            for moe in self.moe_layers.values()
            for parameter in moe.router.parameters()
        }
        return receiver_ids | router_ids

    def added_parameter_count(self) -> int:
        receiver_count = sum(
            parameter.numel()
            for wrapped in self.wrapped_layers.values()
            for parameter in wrapped.receiver_group.parameters()
        )
        return receiver_count + sum(moe.added_parameter_count() for moe in self.moe_layers.values())

    def forward(
        self,
        *args: Any,
        tide_states: Mapping[int, torch.Tensor] | None = None,
        tide_read_state: bool = True,
        tide_clear_positions: Mapping[int, Sequence[int]] | None = None,
        tide_shuffle_positions: Mapping[int, Sequence[int]] | None = None,
        tide_return_artifacts: bool = False,
        **kwargs: Any,
    ) -> TideCausalLMOutput:
        attention_mask = kwargs.get("attention_mask")
        initial_states = {} if tide_states is None else tide_states
        clear_positions = {} if tide_clear_positions is None else tide_clear_positions
        shuffle_positions = {} if tide_shuffle_positions is None else tide_shuffle_positions
        for layer_index, wrapped in self.wrapped_layers.items():
            wrapped.configure_call(
                initial_state=initial_states.get(layer_index),
                valid_tokens=attention_mask,
                read_state=tide_read_state,
                clear_positions=clear_positions.get(layer_index, ()),
                shuffle_positions=shuffle_positions.get(layer_index, ()),
                return_artifacts=tide_return_artifacts,
            )
        for moe in self.moe_layers.values():
            moe.configure_call(attention_mask)

        kwargs["return_dict"] = True
        outputs = self.base_model(*args, **kwargs)
        states: dict[int, torch.Tensor] = {}
        metrics: dict[int, dict[str, torch.Tensor]] = {}
        artifacts: dict[int, ReceiverOutput | MoEOutput] = {}
        losses: list[torch.Tensor] = []
        router_z_losses: list[torch.Tensor] = []
        for layer_index, wrapped in self.wrapped_layers.items():
            if wrapped.last_output is None:
                raise RuntimeError(f"TIDE layer {layer_index} did not execute")
            states[layer_index] = wrapped.last_output.final_state
            metrics[layer_index] = wrapped.last_output.metrics
            losses.append(wrapped.last_output.balance_loss)
            if tide_return_artifacts:
                artifacts[layer_index] = wrapped.last_output

        for layer_index, moe in self.moe_layers.items():
            if moe.last_output is None:
                raise RuntimeError(f"upcycled MoE layer {layer_index} did not execute")
            metrics[layer_index] = moe.last_output.metrics
            losses.append(moe.last_output.balance_loss)
            router_z_losses.append(moe.last_output.router_z_loss)
            if tide_return_artifacts:
                artifacts[layer_index] = moe.last_output

        if losses:
            balance_loss = torch.stack(losses).mean()
        else:
            balance_loss = outputs.logits.new_zeros((), dtype=torch.float32)
        if router_z_losses:
            router_z_loss = torch.stack(router_z_losses).mean()
        else:
            router_z_loss = outputs.logits.new_zeros((), dtype=torch.float32)
        total_loss = outputs.loss
        if total_loss is not None and losses:
            total_loss = total_loss + self.balance_coefficient * balance_loss
        if total_loss is not None and router_z_losses:
            total_loss = total_loss + self.router_z_coefficient * router_z_loss
        return TideCausalLMOutput(
            loss=total_loss,
            lm_loss=outputs.loss,
            balance_loss=balance_loss,
            router_z_loss=router_z_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            states=states,
            metrics=metrics,
            artifacts=artifacts,
        )


def load_qwen3_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    layer_indices: Sequence[int],
    profile: str,
    receiver_count: int,
    expert_count: int,
    state_size: int,
    implementation: str,
    scan_implementation: str,
    balance_coefficient: float,
    router_z_coefficient: float,
    attention_implementation: str,
    initialization: str = "pretrained",
    local_files_only: bool = True,
) -> TideQwen3ForCausalLM:
    from transformers import AutoConfig, AutoModelForCausalLM

    if initialization == "pretrained":
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation=attention_implementation,
            local_files_only=local_files_only,
        )
    elif initialization == "random":
        config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
        base_model = AutoModelForCausalLM.from_config(
            config,
            dtype=dtype,
            attn_implementation=attention_implementation,
        )
    else:
        raise ValueError(f"unknown initialization: {initialization}")
    if base_model.config.model_type != "qwen3":
        raise ValueError(f"expected a Qwen3 checkpoint, found {base_model.config.model_type}")
    base_model.config.use_cache = False
    return TideQwen3ForCausalLM(
        base_model,
        layer_indices=layer_indices,
        profile=profile,
        receiver_count=receiver_count,
        expert_count=expert_count,
        state_size=state_size,
        implementation=implementation,
        scan_implementation=scan_implementation,
        balance_coefficient=balance_coefficient,
        router_z_coefficient=router_z_coefficient,
    )
