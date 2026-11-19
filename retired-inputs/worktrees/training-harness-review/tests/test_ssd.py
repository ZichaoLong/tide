"""Independent SSD recurrence and literal logical-time cache checks."""

import pytest
import torch

from tide.ssd import ConvState, logical_conv, ssd_scan


def recurrence(decay, values, B, C, state):
    reads = []
    for i in range(decay.shape[-2]):
        state = (decay[..., i, :, None, None] * state
                 + torch.einsum("...hv,...hk->...hvk", values[..., i, :, :], B[..., i, :, :]))
        reads.append(torch.einsum("...hvk,...hk->...hv", state, C[..., i, :, :]))
    return torch.stack(reads, dim=-3), state


def scan_inputs(length=45):
    generator = torch.Generator().manual_seed(1701)
    shape = (2, 3, length, 2)
    decay = torch.rand(shape, dtype=torch.float64, generator=generator)
    # Include complete forgetting and exact no-write rows in independent checks.
    decay[..., 2, :] = 0
    values = torch.randn(*shape, 3, dtype=torch.float64, generator=generator)
    B = torch.randn(*shape, 4, dtype=torch.float64, generator=generator)
    C = torch.randn(*shape, 4, dtype=torch.float64, generator=generator)
    initial = torch.randn(2, 3, 2, 3, 4, dtype=torch.float64, generator=generator)
    return [x.requires_grad_() for x in (decay, values, B, C, initial)]


@pytest.mark.parametrize("chunk_size", [1, 7, 32])
def test_scan_matches_recurrence_and_all_input_vjps(chunk_size):
    inputs = scan_inputs()
    actual = ssd_scan(*inputs, chunk_size=chunk_size)
    expected = recurrence(*inputs)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)
    roots = [torch.sin(torch.arange(x.numel(), dtype=x.dtype)).reshape_as(x) for x in actual]
    got_grad = torch.autograd.grad(actual, inputs, roots, retain_graph=True)
    want_grad = torch.autograd.grad(expected, inputs, roots)
    for got, want in zip(got_grad, want_grad):
        torch.testing.assert_close(got, want, atol=1e-11, rtol=1e-11)


def test_scan_scalar_example_and_nonempty_continuation():
    decay = torch.full((2, 1), .5, dtype=torch.float64)
    values = torch.ones(2, 1, 1, dtype=torch.float64)
    B, C = torch.ones_like(values), 3 * torch.ones_like(values)
    reads, final = ssd_scan(decay, values, B, C, torch.zeros(1, 1, 1, dtype=torch.float64))
    torch.testing.assert_close(reads.flatten(), torch.tensor([3., 4.5], dtype=torch.float64))
    torch.testing.assert_close(final, torch.full_like(final, 1.5))
    inputs = scan_inputs()
    whole, terminal = ssd_scan(*inputs)
    left, state = ssd_scan(*(x[..., :19, :] if i == 0 else x[..., :19, :, :]
                             for i, x in enumerate(inputs[:4])), inputs[4])
    right, state = ssd_scan(*(x[..., 19:, :] if i == 0 else x[..., 19:, :, :]
                              for i, x in enumerate(inputs[:4])), state)
    torch.testing.assert_close(torch.cat((left, right), dim=-3), whole, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(state, terminal, atol=1e-12, rtol=1e-12)


def test_scan_empty_and_absent_receptions():
    initial = torch.randn(2, 3, 4, dtype=torch.float64)
    empty = torch.empty(0, 2, dtype=torch.float64)
    reads, final = ssd_scan(empty, torch.empty(0, 2, 3, dtype=empty.dtype),
                            torch.empty(0, 2, 4, dtype=empty.dtype),
                            torch.empty(0, 2, 4, dtype=empty.dtype), initial)
    assert reads.shape == (0, 2, 3) and final is initial
    decay = torch.ones(5, 2, dtype=empty.dtype)
    values = torch.zeros(5, 2, 3, dtype=empty.dtype)
    B, C = torch.randn(5, 2, 4, dtype=empty.dtype), torch.randn(5, 2, 4, dtype=empty.dtype)
    reads, final = ssd_scan(decay, values, B, C, initial)
    torch.testing.assert_close(final, initial, atol=0, rtol=0)
    torch.testing.assert_close(reads, torch.einsum("hvk,lhk->lhv", initial, C))


def literal_conv(projected, mass, times, kernel, bias, state, stride):
    """Dictionary recurrence uses exact Python integers, separate from gather code."""
    batch, length, channels = projected.shape
    outputs, caches = [], []
    for row in range(batch):
        cache = {int(t): v for t, v in zip(state.times[row], state.values[row]) if int(t) >= 0}
        results = []
        for i in range(length):
            time = int(times[row, i])
            value = bias[row] + kernel[row, 0] * projected[row, i]
            for lag in range(1, kernel.shape[-2]):
                value = value + kernel[row, lag] * cache.get(time - lag * stride, 0)
            results.append(value)
            if mass[row, i] > 0:
                cutoff = max(0, time - (kernel.shape[-2] - 1) * stride)
                cache = {t: v for t, v in cache.items() if cutoff <= t < time}
                cache[time] = mass[row, i] * projected[row, i]
        outputs.append(torch.stack(results))
        caches.append(cache)
    return torch.stack(outputs), caches


def conv_inputs():
    generator = torch.Generator().manual_seed(771)
    projected = torch.randn(2, 7, 3, dtype=torch.float64, generator=generator).requires_grad_()
    mass = torch.tensor([[.3, 0, .8, .1, 0, .5, 0], [.2, .6, 0, 0, .7, 0, 0]],
                        dtype=torch.float64, requires_grad=True)
    times = torch.tensor([[6, 7, 9, 11, 12, 14, 18], [6, 8, 10, 12, 14, 16, 19]])
    kernel = torch.randn(2, 4, 3, dtype=torch.float64, generator=generator).requires_grad_()
    bias = torch.randn(2, 3, dtype=torch.float64, generator=generator).requires_grad_()
    cache_times = torch.tensor([[0, 1, 3], [0, 2, 5]])
    cache_values = torch.randn(2, 3, 3, dtype=torch.float64, generator=generator).requires_grad_()
    return projected, mass, times, kernel, bias, ConvState(cache_times, cache_values)


def assert_cache(actual, expected):
    for row, cache in enumerate(expected):
        valid = actual.times[row] >= 0
        assert actual.times[row, valid].tolist() == sorted(cache)
        if cache:
            torch.testing.assert_close(actual.values[row, valid], torch.stack([cache[t] for t in sorted(cache)]))


def test_conv_irregular_clock_raw_cache_and_vjps():
    projected, mass, times, kernel, bias, state = conv_inputs()
    actual, terminal = logical_conv(projected, mass, times, kernel, bias, state, stride=2)
    expected, caches = literal_conv(projected, mass, times, kernel, bias, state, 2)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert_cache(terminal, caches)
    roots = (projected, mass, kernel, bias, state.values)
    got_loss = actual.square().sum() + terminal.values.square().sum()
    want_loss = expected.square().sum() + sum(v.square().sum() for cache in caches for v in cache.values())
    got = torch.autograd.grad(got_loss, roots, retain_graph=True)
    want = torch.autograd.grad(want_loss, roots)
    # At exact zero mass both paths declare the no-write branch derivative.
    for x, y in zip(got, want):
        torch.testing.assert_close(x, y, atol=1e-11, rtol=1e-11)


def test_conv_continuation_and_no_positive_write_preserve_expired_cache():
    projected, mass, times, kernel, bias, state = conv_inputs()
    whole, terminal = logical_conv(projected, mass, times, kernel, bias, state, stride=2)
    left, middle = logical_conv(projected[:, :3], mass[:, :3], times[:, :3], kernel, bias, state, stride=2)
    right, final = logical_conv(projected[:, 3:], mass[:, 3:], times[:, 3:], kernel, bias, middle, stride=2)
    torch.testing.assert_close(torch.cat((left, right), 1), whole, atol=0, rtol=0)
    roots = (projected, mass, kernel, bias, state.values)
    whole_grad = torch.autograd.grad(whole.square().sum() + terminal.values.square().sum(),
                                     roots, retain_graph=True)
    split_grad = torch.autograd.grad(left.square().sum() + right.square().sum()
                                     + final.values.square().sum(), roots, retain_graph=True)
    for got, want in zip(split_grad, whole_grad):
        torch.testing.assert_close(got, want, atol=1e-11, rtol=1e-11)
    for row in range(2):
        torch.testing.assert_close(final.times[row, final.times[row] >= 0],
                                   terminal.times[row, terminal.times[row] >= 0], atol=0, rtol=0)
        torch.testing.assert_close(final.values[row, final.times[row] >= 0],
                                   terminal.values[row, terminal.times[row] >= 0], atol=0, rtol=0)
    _, idle = logical_conv(projected, torch.zeros_like(mass), times + 1000, kernel, bias, state, stride=2)
    assert_cache(idle, [{int(t): v for t, v in zip(ts, vs)} for ts, vs in zip(state.times, state.values)])


def test_conv_current_unweighted_and_large_int64_clock():
    # The textbook example is 6 + 2*(epsilon*5) + 3*4 at time four.
    projected = torch.tensor([[4.], [5.], [6.]], dtype=torch.float64)
    mass = torch.tensor([1., .01, 0.], dtype=torch.float64)
    times = torch.tensor([0, 2, 4]) + 2**55
    kernel = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64)
    out, state = logical_conv(projected, mass, times, kernel, torch.zeros(1, dtype=torch.float64), stride=2)
    torch.testing.assert_close(out[-1], torch.tensor([18.1], dtype=torch.float64))
    assert state.times[state.times >= 0].tolist() == times[:2].tolist()
    assert state.times.dtype == torch.int64


def test_conv_node_parameters_broadcast_without_node_or_sample_loop():
    projected = torch.arange(2 * 3 * 5 * 4, dtype=torch.float64).reshape(2, 3, 5, 4) / 50
    mass = torch.ones(2, 3, 5, dtype=torch.float64)
    times = torch.arange(5).expand(2, 3, 5)
    kernel = torch.ones(2, 1, 4, 4, dtype=torch.float64)
    bias = torch.zeros(2, 1, 4, dtype=torch.float64)
    actual, state = logical_conv(projected, mass, times, kernel, bias)
    expected = torch.stack([projected[..., max(0, t - 3):t + 1, :].sum(-2) for t in range(5)], -2)
    torch.testing.assert_close(actual, expected)
    assert state.times.shape == (2, 3, 4)


def test_invalid_metadata_is_rejected():
    inputs = scan_inputs()
    with pytest.raises(ValueError, match="chunk_size"):
        ssd_scan(*inputs, chunk_size=64)
    with pytest.raises(ValueError, match="initial"):
        ssd_scan(*inputs[:4], inputs[4][..., :2])
    projected, mass, times, kernel, bias, state = conv_inputs()
    with pytest.raises(ValueError, match="int64"):
        logical_conv(projected, mass, times.float(), kernel, bias, state)
    with pytest.raises(ValueError, match="stride"):
        logical_conv(projected, mass, times, kernel, bias, state, stride=0)
