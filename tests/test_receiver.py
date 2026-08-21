from __future__ import annotations

import io
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

from tide.receiver import (
    TideReceiverGroup,
    ema_scan_reference,
    ema_scan_vectorized,
)


def _randomize_down(group: TideReceiverGroup) -> None:
    with torch.no_grad():
        for receiver in group.ffns:
            receiver.down_proj.weight.normal_(std=0.03)


def test_vectorized_scan_matches_reference_and_chunks() -> None:
    torch.manual_seed(3)
    observations = torch.randn(2, 9, 4, 5)
    receive = torch.rand(2, 9, 4) > 0.4
    decay = torch.sigmoid(torch.randn(4, 5) + 3.0)
    initial = torch.randn(2, 4, 5)
    expected = ema_scan_reference(observations, receive, decay, initial)
    actual = ema_scan_vectorized(observations, receive, decay, initial)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    first = ema_scan_vectorized(observations[:, :4], receive[:, :4], decay, initial)
    second = ema_scan_vectorized(observations[:, 4:], receive[:, 4:], decay, first[:, -1])
    torch.testing.assert_close(torch.cat([first, second], dim=1), expected, atol=2e-6, rtol=2e-6)


def test_bo_and_selected_dispatch_receive_semantics() -> None:
    hidden = torch.randn(2, 7, 16)
    bo = TideReceiverGroup(16, 32, state_size=6, profile="bo")
    selected = TideReceiverGroup(16, 32, state_size=6, profile="selected-dispatch")
    selected.load_state_dict(bo.state_dict())
    bo_output = bo(hidden, return_artifacts=True)
    selected_output = selected(hidden, return_artifacts=True)
    assert bo_output.metrics["receive_counts"].tolist() == [14, 14, 14, 14]
    assert selected_output.metrics["receive_counts"].sum().item() == 14
    assert torch.equal(
        selected_output.metrics["receive_counts"], selected_output.metrics["active_counts"]
    )
    assert not torch.equal(bo_output.final_state, selected_output.final_state)


def test_packed_matches_dense_forward_backward() -> None:
    torch.manual_seed(11)
    dense = TideReceiverGroup(
        16,
        32,
        state_size=6,
        profile="bo",
        implementation="dense-masked-reference",
    )
    _randomize_down(dense)
    packed = TideReceiverGroup(16, 32, state_size=6, profile="bo", implementation="packed")
    packed.load_state_dict(dense.state_dict())
    dense_input = torch.randn(2, 8, 16, requires_grad=True)
    packed_input = dense_input.detach().clone().requires_grad_(True)
    dense_output = dense(dense_input)
    packed_output = packed(packed_input)
    torch.testing.assert_close(packed_output.hidden, dense_output.hidden, atol=2e-6, rtol=2e-6)
    dense_output.hidden.square().mean().backward()
    packed_output.hidden.square().mean().backward()
    torch.testing.assert_close(packed_input.grad, dense_input.grad, atol=3e-6, rtol=3e-6)
    for dense_parameter, packed_parameter in zip(dense.parameters(), packed.parameters(), strict=True):
        if dense_parameter.grad is None:
            assert packed_parameter.grad is None
        else:
            torch.testing.assert_close(
                packed_parameter.grad,
                dense_parameter.grad,
                atol=3e-6,
                rtol=3e-6,
            )


def test_neutral_initialization_then_inner_gradient_and_round_trip() -> None:
    torch.manual_seed(17)
    group = TideReceiverGroup(16, 32, state_size=6, profile="bo")
    optimizer = torch.optim.AdamW(group.parameters(), lr=0.05)
    hidden = torch.randn(2, 8, 16)
    initial = group(hidden)
    assert torch.equal(initial.hidden, hidden)
    initial.hidden.square().mean().backward()
    assert sum(receiver.down_proj.weight.grad.norm() for receiver in group.ffns) > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = group(hidden)
    second.hidden.square().mean().backward()
    assert group.observers[0].weight.grad is not None
    assert group.observers[0].weight.grad.norm() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    payload = io.BytesIO()
    torch.save(
        {"model": group.state_dict(), "optimizer": optimizer.state_dict()},
        payload,
    )
    payload.seek(0)
    restored = TideReceiverGroup(16, 32, state_size=6, profile="bo")
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.05)
    checkpoint = torch.load(payload, weights_only=True)
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    torch.testing.assert_close(restored(hidden).hidden, group(hidden).hidden)

    for candidate, candidate_optimizer in (
        (group, optimizer),
        (restored, restored_optimizer),
    ):
        candidate_optimizer.zero_grad(set_to_none=True)
        candidate(hidden).hidden.square().mean().backward()
        candidate_optimizer.step()
    for expected, actual in zip(group.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_receiver_full_sequence_matches_token_continuation() -> None:
    torch.manual_seed(23)
    group = TideReceiverGroup(16, 32, state_size=6, profile="bo")
    _randomize_down(group)
    hidden = torch.randn(2, 9, 16)
    full = group(hidden)
    first = group(hidden[:, :4])
    second = group(hidden[:, 4:], initial_state=first.final_state)
    torch.testing.assert_close(
        torch.cat([first.hidden, second.hidden], dim=1), full.hidden, atol=3e-6, rtol=3e-6
    )
    torch.testing.assert_close(second.final_state, full.final_state, atol=2e-6, rtol=2e-6)
