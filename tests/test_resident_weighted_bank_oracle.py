"""Independent bias-sum oracle for the device 193-row capacity witness.

The generated semantic kernels run on CPU; this does not certify CANN's
ReduceSum, capture/replay, or the full resident flow.
"""
import math

import torch

from test_resident_weighted_bank_gemm import backward, forward, library


def accurate_bias(gradient, owners, banks):
    return torch.tensor([
        [math.fsum(gradient[owners == bank, col].tolist()) for col in range(gradient.shape[1])]
        for bank in range(banks)
    ], dtype=torch.float64)


def test_original_device_material_bias_against_independent_accurate_sum(library):
    def values(shape, shift):
        return (torch.arange(math.prod(shape), dtype=torch.float32)*.13+shift).sin().reshape(shape)*.2
    owners = torch.arange(193).remainder(3)
    bar = values((193, 32), .71)
    _, tape = forward(library, values((193, 32), .13), owners,
                      values((3, 32, 32), .37), values((3, 32), .23))
    _, _, bias, _ = backward(library, tape, bar, torch.ones(193, dtype=torch.bool))
    exact = accurate_bias(bar, owners, 3)
    independent = torch.zeros(3, 32, dtype=torch.float64)
    for row in range(193): independent[owners[row]] += bar[row].double()
    torch.testing.assert_close(independent, exact, atol=0, rtol=0)
    torch.testing.assert_close(bias.double(), exact, atol=1e-6, rtol=1e-5)


def test_fp32_sequential_oracle_can_lose_real_connected_contributions():
    owners = torch.arange(193).remainder(3)
    gradient = torch.zeros(193, 1)
    gradient[2, 0], gradient[5, 0], gradient[8, 0] = 2**24, 1, -(2**24)
    legacy = torch.zeros(3, 1)
    for row in range(193): legacy[owners[row]] += gradient[row]
    exact = accurate_bias(gradient, owners, 3)
    assert exact[:, 0].tolist() == [0, 0, 1] and legacy.eq(0).all()
