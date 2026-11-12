"""Read transport shares true roots across nodes, without a device workload.

CPU dtype changes exercise the real copy/VJP primitive. Destination-dispatch
tests use an explicit fake transport and make no accelerator correctness claim.
"""
import pytest
import torch
from torch import nn

from tidegraph.compare import equivalent
from tidegraph.weighted import KVConfig
from tidegraph.weighted import column_modules, column_placement, placement_tensors
from tidegraph.weighted.column_layout import SequenceLayout
from tidegraph.weighted.kv_columns_state import KVStateColumns, KVWindowColumns, KVWriteBank


def material(dtype=torch.float32, mode="+"):
    def leaf(shape, start):
        n = 1
        for size in shape:
            n *= size
        return (torch.arange(n, dtype=dtype).reshape(shape)*.03+start).requires_grad_()
    common = nn.Parameter(leaf((2, 2), .2))
    independent = nn.Parameter(common.detach().clone())
    layout = SequenceLayout(torch.tensor([0, 3]), torch.tensor([3, 6, 10]))
    bank_layout = SequenceLayout(torch.tensor([0, 2]), torch.tensor([1, 3]))
    requests = {}
    for i, key in enumerate((7, 2, 19, 4)):
        node = nn.Module()
        node.kernel, node.read_program, node.full_program = nn.Module(), nn.Module(), nn.Module()
        node.kernel.config = KVConfig(width=2, summary_width=2, memory_read="sum")
        node.kernel.summary = common if i < 3 else independent
        node.read_program.weight = node.kernel.summary  # Cross-role true alias.
        node.kernel.base = nn.Parameter(leaf((4, 2), .5+i))  # Not read or copied.
        bank = KVWriteBank(bank_layout, leaf((2, 2), .4), leaf((2, 2), .6),
                           leaf((2, 2), .8), leaf((2,), .2))
        initial = KVStateColumns(bank_layout, bank.keys, bank.values, bank.summaries, bank.weights,
                                 torch.zeros((1, 3), dtype=dtype), torch.tensor([3]), torch.tensor([2]))
        window = KVWindowColumns(layout, leaf((3,), .3), leaf((3, 2), .1+i), bank, initial)
        requests[key] = None, node, window, initial, mode, torch.device("cpu"), torch.float64
    return requests


def variables(requests):
    result = {}
    def add(name, value):
        if value.requires_grad and all(value is not t for t in result.values()):
            result[name] = value
    for key, (_, node, window, initial, *_) in requests.items():
        for name, value in node.named_parameters(remove_duplicate=False):
            add(f"{key}.{name}", value)
        for name, value in (("mass", window.mass), ("content", window.content),
                            ("weights", initial.weights), ("summaries", initial.summaries)):
            add(f"{key}.{name}", value)
    return result


def install_formula(monkeypatch):
    observed = {}
    def evaluate(prepared, *, max_workspace_bytes):
        observed.update(prepared)
        return {key: window.mass[:, None]*(window.content@node.kernel.summary.t())
                     +window.content@node.read_program.weight.t()
                for key, (_, node, window, _, _) in prepared.items()}
    monkeypatch.setattr(column_modules, "read_many", evaluate)
    return observed


def literal(requests):
    result = {}
    for key, (_, node, window, _, _, _, dtype) in requests.items():
        x, mass = window.content.to(dtype), window.mass.to(dtype)
        left, right = node.kernel.summary.to(dtype), node.read_program.weight.to(dtype)
        result[key] = torch.stack([torch.stack([
            mass[row]*sum(x[row, d]*left[out, d] for d in range(2))
            +sum(x[row, d]*right[out, d] for d in range(2)) for out in range(2)]) for row in range(3)])
    return result


def gradient(loss, roots):
    return dict(zip(roots, torch.autograd.grad(loss, tuple(roots.values()), allow_unused=True, retain_graph=True)))


@pytest.mark.parametrize("mode", ["0", "-", "+"])
@pytest.mark.parametrize("factor", [1., 0.])
def test_grouped_read_copy_preserves_shared_aliases_and_node_roots(monkeypatch, mode, factor):
    requests = material(mode=mode)
    observed = install_formula(monkeypatch)
    calls, original = [], column_placement.mapper
    transfers, copy_tensor = [], placement_tensors.copy_tensor
    def copied(value, device, dtype, **options):
        transfers.append(value.numel())
        return copy_tensor(value, device, dtype, **options)
    monkeypatch.setattr(placement_tensors, "copy_tensor", copied)
    def mapped(values, device, dtype):
        calls.append((tuple(values), device, dtype))
        return original(values, device, dtype)
    monkeypatch.setattr(column_placement, "mapper", mapped)
    actual = column_placement.read_many(requests, max_workspace_bytes=2**20)
    assert list(actual) == list(requests)
    assert len(calls) == 1
    unique = {id(t): t for t in calls[0][0]}
    assert transfers == [sum(t.numel() for t in unique.values())]
    assert sum(transfers) < sum(t.numel() for t in calls[0][0])
    assert observed[7][1].kernel.summary is observed[2][1].kernel.summary is observed[19][1].kernel.summary
    assert observed[19][1].kernel.summary is observed[19][1].read_program.weight
    assert observed[4][1].kernel.summary is not observed[7][1].kernel.summary
    assert torch.equal(observed[4][1].kernel.summary, observed[7][1].kernel.summary)
    for key, request in requests.items():
        _, original_node, original_window, original_initial, *_ = request
        _, placed_node, placed_window, _, placed_initial = observed[key]
        assert placed_node.kernel.base is original_node.kernel.base
        assert original_window.content.dtype == torch.float32
        assert placed_window.content.dtype == torch.float64
        source = placed_initial if mode == "0" else placed_window.bank
        assert source.weights.dtype == source.summaries.dtype == torch.float64
        assert original_initial.weights.dtype == torch.float32
    expected, roots = literal(requests), variables(requests)
    equivalent(actual, expected)
    # Only non-first node 19 is observed; the independent equal-valued owner and
    # every unused node input remain None. Explicit zero remains connected.
    bar = torch.linspace(-.2, .7, 6, dtype=torch.float64).reshape(3, 2)*factor
    got = gradient((actual[19]*bar).sum(), roots)
    equivalent(got, gradient((expected[19]*bar).sum(), roots))
    assert got["7.content"] is got["2.content"] is got["4.content"] is None
    assert got["4.kernel.summary"] is None
    assert got["7.kernel.summary"] is not None
    if factor == 0:
        assert got["7.kernel.summary"].count_nonzero() == 0
        assert got["19.content"].count_nonzero() == 0


@pytest.mark.parametrize("role", ["observed", "unused_borrowed"])
def test_grouped_copy_guards_original_versions(monkeypatch, role):
    requests = material()
    install_formula(monkeypatch)
    actual = column_placement.read_many(requests, max_workspace_bytes=2**20)
    target = requests[19][2].content if role == "observed" else requests[4][2].bank.weights
    with torch.no_grad():
        target.add_(.01)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        actual[19].sum().backward()


def test_same_location_and_dtype_preserves_original_objects(monkeypatch):
    requests = material()
    requests[19] = (*requests[19][:-1], torch.float32)
    observed = install_formula(monkeypatch)
    result = column_placement.read_many(requests, max_workspace_bytes=2**20)
    assert observed[19][1] is requests[19][1]
    assert observed[19][2] is requests[19][2]
    assert observed[19][1].kernel.summary is not observed[7][1].kernel.summary
    equivalent(result, literal(requests))


def test_distinct_destinations_and_dtypes_never_share_a_copy(monkeypatch):
    requests = material()
    destinations = (("cuda:0", torch.float64), ("cuda:0", torch.float64),
                    ("cuda:1", torch.float64), ("cuda:0", torch.float32))
    for key, (device, dtype) in zip(requests, destinations):
        requests[key] = (*requests[key][:-2], torch.device(device), dtype)
    observed, calls = install_formula(monkeypatch), []
    def fake_transport(values, device, dtype):
        # Dispatch/identity witness only; no CUDA/NPU allocation or operation.
        unique = {id(t): t for t in values}
        copies = {key: t.clone().to(dtype) for key, t in unique.items()}
        calls.append((device, dtype))
        return lambda t: copies[id(t)]
    monkeypatch.setattr(column_placement, "mapper", fake_transport)
    column_placement.read_many(requests, max_workspace_bytes=2**20)
    assert calls == [(torch.device("cuda:0"), torch.float64),
                     (torch.device("cuda:1"), torch.float64), (torch.device("cuda:0"), torch.float32)]
    assert observed[7][1].kernel.summary is observed[2][1].kernel.summary
    assert observed[7][1].kernel.summary is not observed[19][1].kernel.summary


def test_no_grad_transport_does_not_build_a_private_graph(monkeypatch):
    requests = material()
    observed = install_formula(monkeypatch)
    with torch.no_grad():
        result = column_placement.read_many(requests, max_workspace_bytes=2**20)
    assert all(not value.requires_grad and value.grad_fn is None for value in result.values())
    assert observed[7][1].kernel.summary is observed[19][1].kernel.summary


def test_transport_first_order_contract_is_retained(monkeypatch):
    requests = material()
    install_formula(monkeypatch)
    actual = column_placement.read_many(requests, max_workspace_bytes=2**20)
    with pytest.raises(ValueError, match="first-order"):
        torch.autograd.grad(actual[19].sum(), requests[19][2].content, create_graph=True)
