"""Base-block integration for the four SettleGraph placements.

The attention and MLP callables return their residual branches.  The graph
callable returns a complete hidden with the same shape, dtype, and device as
its input.  This module only composes those values; normalization and any
state transaction remain responsibilities of the supplied callables.
"""

from __future__ import annotations

from typing import Callable, Tuple

from torch import Tensor


HiddenTransform = Callable[[Tensor], Tensor]

POST = "POST"
PARBLK = "PARBLK"
PARATTN = "PARATTN"
PARMLP = "PARMLP"

SUPPORTED_PLACEMENTS: Tuple[str, ...] = (POST, PARBLK, PARATTN, PARMLP)


class PlacementContractError(ValueError):
    """A placement name or supplied hidden transform violates the contract."""


def _apply_hidden_transform(
    role: str,
    transform: HiddenTransform,
    hidden: Tensor,
) -> Tensor:
    result = transform(hidden)
    if not isinstance(result, Tensor):
        raise PlacementContractError(
            f"{role} must return a torch.Tensor, got {type(result).__name__}"
        )
    if result.shape != hidden.shape:
        raise PlacementContractError(
            f"{role} changed hidden shape from {tuple(hidden.shape)} "
            f"to {tuple(result.shape)}"
        )
    if result.dtype != hidden.dtype:
        raise PlacementContractError(
            f"{role} changed hidden dtype from {hidden.dtype} to {result.dtype}"
        )
    if result.device != hidden.device:
        raise PlacementContractError(
            f"{role} changed hidden device from {hidden.device} to {result.device}"
        )
    return result


def apply_placement(
    hidden: Tensor,
    *,
    placement: str,
    attention: HiddenTransform,
    mlp: HiddenTransform,
    graph: HiddenTransform,
) -> Tensor:
    """Apply one documented SettleGraph placement to a base block.

    ``attention(x)`` is the attention residual branch and ``mlp(u)`` is the
    dense-MLP residual branch.  ``graph(h)`` is the complete SettleGraph
    hidden, so its residual at an input ``h`` is ``graph(h) - h``.

    Args:
        hidden: The base-block input ``x``.  Any leading dimensions are
            allowed; each transform must preserve the entire Tensor contract.
        placement: One of ``POST``, ``PARBLK``, ``PARATTN``, or ``PARMLP``.
        attention: Callable producing the attention residual.
        mlp: Callable producing the dense-MLP residual.
        graph: Callable producing a complete graph hidden.

    Returns:
        The block output ``y`` for the selected placement.
    """

    if not isinstance(hidden, Tensor):
        raise TypeError(
            f"hidden must be a torch.Tensor, got {type(hidden).__name__}"
        )
    if placement not in SUPPORTED_PLACEMENTS:
        allowed = ", ".join(SUPPORTED_PLACEMENTS)
        raise PlacementContractError(
            f"unsupported placement {placement!r}; expected one of: {allowed}"
        )

    attention_residual = _apply_hidden_transform(
        "attention", attention, hidden
    )
    attention_output = hidden + attention_residual

    if placement == POST:
        mlp_residual = _apply_hidden_transform(
            "mlp", mlp, attention_output
        )
        base_output = attention_output + mlp_residual
        return _apply_hidden_transform("graph", graph, base_output)

    if placement == PARBLK:
        mlp_residual = _apply_hidden_transform(
            "mlp", mlp, attention_output
        )
        base_output = attention_output + mlp_residual
        graph_output = _apply_hidden_transform("graph", graph, hidden)
        return base_output + (graph_output - hidden)

    if placement == PARATTN:
        graph_output = _apply_hidden_transform("graph", graph, hidden)
        merged_attention_output = attention_output + (graph_output - hidden)
        mlp_residual = _apply_hidden_transform(
            "mlp", mlp, merged_attention_output
        )
        return merged_attention_output + mlp_residual

    # PARMLP: the dense MLP and graph receive the same attention output.
    mlp_residual = _apply_hidden_transform("mlp", mlp, attention_output)
    base_output = attention_output + mlp_residual
    graph_output = _apply_hidden_transform("graph", graph, attention_output)
    return base_output + (graph_output - attention_output)


__all__ = [
    "HiddenTransform",
    "PARATTN",
    "PARBLK",
    "PARMLP",
    "POST",
    "PlacementContractError",
    "SUPPORTED_PLACEMENTS",
    "apply_placement",
]
