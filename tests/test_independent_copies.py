"""Explicit owned copies and default read-only query publication boundaries."""
import os
import pytest
import torch
from tidegraph.runtime import resolve_device
from tidegraph.compare import equivalent
from tidegraph.weighted.independent_copies import detached_copies

DEVICE, _ = resolve_device(os.environ.get("TIDE_DENSE_DEVICE", "cpu"))


def test_foreach_publication_keeps_bits_and_independent_storage(dtype):
    x = torch.tensor([0., -0., float("inf"), -3.], dtype=dtype, device=DEVICE, requires_grad=True)
    values = [x[:2], x[::2], x[1:2], x[:0], x[:2], x.to(torch.float32)]
    copies = detached_copies(values)
    for value, copied in zip(values, copies):
        assert not copied.requires_grad
        torch.testing.assert_close(copied, value, atol=0, rtol=0)
        # Compare host sign bits; NPU signbit(-0.) itself is not the copy oracle.
        assert torch.equal(torch.signbit(copied.cpu()), torch.signbit(value.cpu()))
    original, versions = copies[4].clone(), [t._version for t in copies]
    copies[0].resize_(100).fill_(42)
    torch.testing.assert_close(copies[4], original, atol=0, rtol=0)
    assert [t._version for t in copies[1:]] == versions[1:]
    assert torch.signbit(x.cpu()[1]) and x.cpu()[0] == 0
    assert detached_copies([]) == ()


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_query_output_clone_for_mutation_keeps_other_fields_and_vjp(dtype, native):
    from test_weighted_dense_query import independent
    from test_weighted_query_vjp import native_query
    from tidegraph.weighted.query_vjp import evaluate
    from isolated_cases import vjp
    w, requests, roots = independent(dtype)
    rows = [requests[1], requests[1], requests[2]]
    expected = [w.read_program.step(w, r) for r in rows]
    w.read_program.query_execution = "packed-v2"
    actual = (native_query if native else evaluate)(w.read_program, w, rows)
    objective = actual[1]["score"].square() + actual[2]["evidence"].square().sum()
    before = vjp(objective, roots)
    # Default records are immutable views; a consumer requesting mutation owns
    # the explicit copy instead of forcing one allocation for every record.
    actual[0]["score"].detach().clone().resize_(100).fill_(42)
    equivalent(actual[1:], expected[1:])
    equivalent(vjp(objective, roots), before, atol=0, rtol=0)


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_readonly_query_publication_retains_private_version_guard(dtype, native):
    from test_weighted_dense_query import independent
    from test_weighted_query_vjp import native_query
    from tidegraph.weighted.query_vjp import evaluate
    w, requests, _ = independent(dtype)
    w.read_program.query_execution = "packed-v2"
    actual = (native_query if native else evaluate)(w.read_program, w, [requests[1]]*3)
    loss = actual[1]["score"].square()
    # Illegal live-result mutation must not silently use the corrupted private
    # graph. Mutating a clone above is the supported migration.
    actual[0]["score"].detach().add_(1)
    with pytest.raises(RuntimeError, match="modified|version|inplace"):
        loss.backward()
