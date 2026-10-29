"""CPU structural check only; actual two-device event handling needs hardware."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import pytest
import torch
from weighted_peer_bridge import _Copy, cpu_bridge


def test_bridge_independent_zero_absent_roots_versions_and_restore():
    import tidegraph.transfer as transport
    original = transport.copy_rows
    counts = dict(forward_calls=0, backward_calls=0, forward_bytes=0, backward_bytes=0)
    a, b = [torch.ones(3, requires_grad=True) for _ in range(2)]
    x, y = [_Copy.apply(t, torch.device("cpu"), torch.float64, counts) for t in (a, b)]
    ga, gb = torch.autograd.grad(x.sum()*0, (a, b), allow_unused=True, retain_graph=True)
    assert torch.equal(ga, torch.zeros_like(a)) and gb is None
    with pytest.raises(ValueError, match="first-order"):
        torch.autograd.grad(y.sum(), b, create_graph=True)
    with torch.no_grad():a.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        x.sum().backward()
    with cpu_bridge():
        assert transport.copy_rows is not original
    assert transport.copy_rows is original
