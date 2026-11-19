"""Exact binary32 device parts, including layouts and subnormal boundaries."""
import math
import os
import random
import struct

import pytest
import torch

from tide.scaled_arithmetic import binary_parts, restore_binary_scale


@pytest.fixture(scope="module")
def device():
    name = os.environ.get("TIDE_STABLE_DEVICE", "cpu")
    if name.startswith("npu"):
        import torch_npu  # noqa: F401
    return torch.device(name)


def values():
    patterns = [0, 1, 2, 0x007fffff, 0x00800000, 0x00800001,
                0x3effffff, 0x3f000000, 0x3f000001, 0x3f7fffff,
                0x3f800000, 0x3f800001, 0x7f7fffff, 0x7f800000]
    rng = random.Random(348)
    patterns += [rng.randrange(0x7f800000) for _ in range(50)]
    patterns += [p | 0x80000000 for p in patterns]
    return [struct.unpack('f', struct.pack('I', p))[0] for p in patterns]


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "expand", "slice", "scalar"])
def test_binary32_parts_match_scalar_frexp_exactly(device, layout):
    cpu = torch.tensor(values()).reshape(8, 16)
    x = cpu.to(device)
    if layout == "transpose":
        cpu, x = cpu.T, x.T
    elif layout == "expand":
        cpu, x = cpu[:1].expand(8, 16), x[:1].expand(8, 16)
    elif layout == "slice":
        cpu, x = cpu[:, ::2], x[:, ::2]
    elif layout == "scalar":
        cpu, x = cpu[0, 1], x[0, 1]
    parts = [math.frexp(v) for v in cpu.flatten().tolist()]
    expected_m = torch.tensor([v[0] for v in parts]).reshape(cpu.shape)
    expected_e = torch.tensor([v[1] for v in parts], dtype=torch.int32).reshape(cpu.shape)
    m, e = binary_parts(x)
    assert m.device == e.device == x.device
    torch.testing.assert_close(m.cpu(), expected_m, rtol=0, atol=0)
    torch.testing.assert_close(e.cpu(), expected_e, rtol=0, atol=0)
    assert torch.equal(torch.signbit(m.cpu()), torch.signbit(expected_m))


def test_restore_extreme_finite_and_signed_zero(device):
    fractions = torch.tensor([-.8816207647, .5, -.5, .5, -.5, 0., -0.], device=device)
    exponents = torch.tensor([128, -148, -148, -150, -150, 999, 999], device=device, dtype=torch.int32)
    actual = restore_binary_scale(fractions, exponents)
    expected = torch.tensor([math.ldexp(float(f), int(e)) for f,e in zip(fractions.cpu(),exponents.cpu())])
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(actual.cpu()), torch.signbit(expected))


def test_restore_overflow_has_correct_sign(device):
    actual = restore_binary_scale(torch.tensor([.75,-.75], device=device),
                                  torch.tensor([130,130], device=device, dtype=torch.int32))
    assert torch.isposinf(actual[0]) and torch.isneginf(actual[1])
