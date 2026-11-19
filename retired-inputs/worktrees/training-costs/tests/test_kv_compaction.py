"""Independent timestamp-sort witness for stable chronological compaction."""
import pytest
import torch

from tide.kv import chronological_compaction


@pytest.mark.parametrize("length", [1, 127, 256, 521])
def test_compaction_matches_stable_sort_including_huge_times_and_empty_rows(length):
    generator = torch.Generator().manual_seed(length)
    keep = torch.rand(3, 4, length, generator=generator) > .4
    keep[0, 0] = False
    keep[0, 1] = True
    # Previously compacted invalid positions may have arbitrary old times.
    times = (2**55 + 17 + torch.arange(length)*4).expand_as(keep)
    expected = torch.where(keep, times, -torch.ones_like(times)).argsort(dim=2, stable=True)
    got = chronological_compaction(keep)
    assert torch.equal(got, expected)
    x = torch.randn(3, 4, length, generator=generator, dtype=torch.float64,
                    requires_grad=True)
    coefficient = torch.randn(3, 4, min(128, length), generator=generator, dtype=x.dtype)
    ga = torch.autograd.grad((x.gather(2, got[:, :, -128:])*coefficient).sum(), x)[0]
    gb = torch.autograd.grad((x.gather(2, expected[:, :, -128:])*coefficient).sum(), x)[0]
    assert torch.equal(ga, gb)
