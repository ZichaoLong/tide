"""Native schema/ownership checkpoint tests through the Python adapter."""

from pathlib import Path

import pytest
import torch

import _tide_native as core


def _fixture(dtype, offset=0.0):
    shared = torch.tensor([1.0 + offset, -2.0 + offset], dtype=dtype).requires_grad_()
    other = torch.tensor([0.5 + offset], dtype=dtype).requires_grad_()
    registry = core.ParameterRegistry()
    registry.add("graph.z.shared", shared)
    registry.add("graph.a.shared", shared)
    registry.add("graph.other", other)
    return registry, shared, other


def _group(names, **options):
    group = core.OptimizerGroup()
    group.parameters = list(names)
    for name, value in options.items():
        setattr(group, name, value)
    return group


def _optimizer(registry, kind, variant=False):
    if kind == "sgd":
        groups = [
            _group(["graph.z.shared"], lr=.777 if variant else .1,
                   momentum=.8, weight_decay=.02),
            _group(["graph.other"], lr=.03, momentum=.6, weight_decay=.44 if variant else 0.0),
        ]
        return core.SGD(registry, groups)
    groups = [
        _group(["graph.z.shared"], lr=.777 if variant else .01,
               weight_decay=.1, beta1=.9, beta2=.99,
               eps=1e-5, amsgrad=True),
        _group(["graph.other"], lr=.03, beta1=.8, beta2=.95, eps=1e-5,
               weight_decay=.44 if variant else 0.0),
    ]
    return core.AdamW(registry, groups)


def _train_two_steps(shared, other, optimizer):
    shared.grad = torch.tensor([1.0, -.25], dtype=shared.dtype)
    other.grad = None
    optimizer.step()
    optimizer.zero_grad()
    shared.grad = torch.zeros_like(shared)
    other.grad = torch.zeros_like(other)
    optimizer.step()
    optimizer.zero_grad()


def _state_snapshot(optimizer):
    result = {}
    for name, state in optimizer.state().items():
        result[name] = {
            "step": state.step,
            "momentum": None if state.momentum_buffer is None else state.momentum_buffer.clone(),
            "exp_avg": None if state.exp_avg is None else state.exp_avg.clone(),
            "exp_avg_sq": None if state.exp_avg_sq is None else state.exp_avg_sq.clone(),
            "max_exp_avg_sq": None if state.max_exp_avg_sq is None else state.max_exp_avg_sq.clone(),
        }
    return result


def _assert_state(actual, expected):
    assert set(actual) == set(expected)
    for name in expected:
        assert actual[name]["step"] == expected[name]["step"]
        for slot in ("momentum", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value, reference = actual[name][slot], expected[name][slot]
            if reference is None:
                assert value is None
            else:
                torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["sgd", "adamw"])
def test_native_checkpoint_round_trip_and_next_update(dtype, kind, tmp_path):
    registry, shared, other = _fixture(dtype)
    optimizer = _optimizer(registry, kind)
    _train_two_steps(shared, other, optimizer)
    expected_values = (shared.detach().clone(), other.detach().clone())
    expected_state = _state_snapshot(optimizer)
    expected_groups = optimizer.groups()
    path = tmp_path / f"{kind}.tidenck"
    core.save_checkpoint(str(path), registry, optimizer, "graph-v1")

    restored_registry, restored_shared, restored_other = _fixture(dtype, 10.0)
    restored = _optimizer(restored_registry, kind, variant=True)
    core.load_checkpoint(str(path), restored_registry, restored, "graph-v1")
    torch.testing.assert_close(restored_shared, expected_values[0], rtol=0, atol=0)
    torch.testing.assert_close(restored_other, expected_values[1], rtol=0, atol=0)
    assert restored_registry.alias_partitions() == registry.alias_partitions()
    assert restored.layout().groups == optimizer.layout().groups
    assert [group.lr for group in restored.groups()] == [group.lr for group in expected_groups]
    _assert_state(_state_snapshot(restored), expected_state)

    shared.grad = torch.tensor([.25, -.75], dtype=dtype)
    other.grad = torch.tensor([.5], dtype=dtype)
    restored_shared.grad = shared.grad.clone()
    restored_other.grad = other.grad.clone()
    optimizer.step()
    restored.step()
    torch.testing.assert_close(restored_shared, shared, rtol=1e-7 if dtype is torch.float32 else 0,
                               atol=2e-6 if dtype is torch.float32 else 0)
    torch.testing.assert_close(restored_other, other, rtol=1e-7 if dtype is torch.float32 else 0,
                               atol=2e-6 if dtype is torch.float32 else 0)
    _assert_state(_state_snapshot(restored), _state_snapshot(optimizer))

    before = (restored_shared.detach().clone(), _state_snapshot(restored))
    with pytest.raises(Exception):
        core.load_checkpoint(str(path), restored_registry, restored, "other-graph")
    torch.testing.assert_close(restored_shared, before[0], rtol=0, atol=0)
    _assert_state(_state_snapshot(restored), before[1])

    raw = path.read_bytes()
    for suffix, content in (("corrupt", bytes([raw[12] ^ 0x80]) + raw[13:]),
                            ("short", raw[:-1])):
        damaged = tmp_path / f"{kind}-{suffix}.tidenck"
        damaged.write_bytes(content)
        target_registry, target_shared, target_other = _fixture(dtype, 3.0)
        target = _optimizer(target_registry, kind)
        before_value = target_shared.detach().clone()
        before_state = _state_snapshot(target)
        with pytest.raises(Exception):
            core.load_checkpoint(str(damaged), target_registry, target, "graph-v1")
        torch.testing.assert_close(target_shared, before_value, rtol=0, atol=0)
        _assert_state(_state_snapshot(target), before_state)

    with pytest.raises(Exception):
        core.save_checkpoint(str(path), registry, optimizer, "graph-v1")
    assert path.read_bytes() == raw


def test_native_checkpoint_weights_only_and_alias_mismatch(dtype, tmp_path):
    registry, shared, other = _fixture(dtype)
    path = Path(tmp_path) / "weights.tidenck"
    core.save_checkpoint(str(path), registry, None, "graph-v1")
    restored_registry, restored_shared, restored_other = _fixture(dtype, 4.0)
    core.load_checkpoint(str(path), restored_registry, None)
    torch.testing.assert_close(restored_shared, shared, rtol=0, atol=0)
    torch.testing.assert_close(restored_other, other, rtol=0, atol=0)

    wrong_registry, wrong_shared, wrong_other = _fixture(dtype)
    wrong_registry = core.ParameterRegistry()
    wrong_registry.add("graph.a.shared", wrong_shared)
    wrong_registry.add("graph.other", wrong_other)
    with pytest.raises(Exception):
        core.load_checkpoint(str(path), wrong_registry, None, "graph-v1")
