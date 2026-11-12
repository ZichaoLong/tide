"""Architecture §2: exact fixed-unit guards, bounded work and complete tails.

No value/gradient formula changes: each fixed scale must contain only literal
ones. The native helper is also used by the standalone C++ model validator.
"""
import os
import pytest
import torch
from tidegraph.runtime import resolve_device
from tidegraph.weighted.transport_validation import validate_unit_scales


def operator(client):
    if client == "python":
        return validate_unit_scales
    from tidegraph.native_loader import load_native
    return load_native(os.environ.get("TIDE_BUILD_DIR")).weighted_validate_unit_scales


def device():
    return resolve_device(os.environ.get("TIDE_COLUMN_DEVICE", "cpu"))[0]


@pytest.mark.parametrize("client", ["python", "native"])
def test_unit_values_empty_strided_aliases_and_no_saved_graph(dtype, client):
    fn = operator(client)
    base = torch.ones((13, 9), dtype=dtype, device=device())
    view = base.t()[:, ::2]
    assert not view.is_contiguous()
    values = [base.new_empty((0, 3)), base[0, 0], view, view, base[:, :1].expand(13, 4)]
    versions = [t._version for t in values]
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        fn(values, budget=3*base.element_size()*7)
        fn([], budget=1)
        fn([base.new_empty(0)], budget=1)
        # Checkpoint inspection may receive a live tensor; validation must not
        # construct a graph even when it does not require the source to be fixed.
        trainable = base.clone().requires_grad_()
        fn([trainable], fixed=False)
    assert not saved and trainable.grad is None
    assert [t._version for t in values] == versions
    assert torch.equal(base, torch.ones_like(base))


@pytest.mark.parametrize("client", ["python", "native"])
@pytest.mark.parametrize("bad", [0., -1., 1.01, float("nan"), float("inf")])
def test_last_element_of_last_strided_tensor_is_always_checked(dtype, client, bad):
    fn = operator(client)
    value = torch.ones((11, 17), dtype=dtype, device=device()).t()
    value[-1, -1] = bad
    prefix = [value.new_ones(()) for _ in range(19)]
    with pytest.raises(ValueError, match="unit scales"):
        fn([*prefix, value], budget=3*value.element_size()*5)


@pytest.mark.parametrize("client", ["python", "native"])
def test_every_trainable_flag_and_later_mutation_are_rechecked(dtype, client):
    fn = operator(client)
    value = torch.ones((), dtype=dtype, device=device())
    values = [value]*17
    fn(values)
    value.fill_(2.)
    with pytest.raises(ValueError, match="unit scales"):
        fn(values)
    value.fill_(1.)
    fn(values)
    for trainable in (value.requires_grad_(), value.new_empty(0).requires_grad_()):
        with pytest.raises(ValueError, match="unit scales"):
            fn([value.detach(), trainable])


@pytest.mark.parametrize("client", ["python", "native"])
def test_one_comparison_per_group_chunk_not_per_scale(dtype, client):
    fn = operator(client)
    for count in (2, 2132):
        scales = list(torch.ones(count, dtype=dtype, device=device()).unbind())
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            fn(scales)
        comparisons = sum(e.count for e in profile.key_averages() if e.key == "aten::equal")
        assert comparisons == 1
    # A tiny byte budget forces more than one complete group; the last chunk
    # must remain checked. Different dtypes never force a conversion.
    values = [torch.ones(9, dtype=dtype, device=device()), torch.ones(1, dtype=torch.int64)]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        fn(values, budget=3*8*2)
    expected = (9+(16//values[0].element_size())-1)//(16//values[0].element_size())+1
    assert sum(e.count for e in profile.key_averages() if e.key == "aten::equal") == expected


@pytest.mark.parametrize("client", ["python", "native"])
def test_explicit_invalid_budget(dtype, client):
    fn = operator(client)
    value = torch.ones(1, dtype=dtype, device=device())
    for budget in (0, -1):
        with pytest.raises(ValueError, match="positive int64"):
            fn([value], budget=budget)
    with pytest.raises(ValueError, match="one element"):
        fn([value], budget=1)


def test_python_binding_and_checkpoint_keep_their_rejection_contract(dtype):
    from tidegraph.weighted import KVConfig, bind, fully_connected
    b = bind(fully_connected(2, 2).graph, memories=KVConfig(width=2, summary_width=2), dtype=dtype)
    value = b.model.output_scale[-1]
    value.requires_grad_(True)
    with pytest.raises(ValueError, match="unit scales"):
        b.model.validate_binding(b.graph)
    value.requires_grad_(False)
    b.model.validate_binding(b.graph)
    value.fill_(2.)
    with pytest.raises(ValueError, match="unit scales"):
        b.model.validate_binding(b.graph)
    with pytest.raises(ValueError, match="checkpoint changes"):
        b.model.validate_checkpoint_weights(b.model.state_dict())
    value.fill_(1.)
    # Existing checkpoint-value validation did not reject a grad-enabled
    # temporary source tensor or inspect unrelated learned parameter values.
    b.model.validate_checkpoint_weights({"output_scale.0": value.clone().requires_grad_(),
                                        "nodes.0.bias": value.new_full((3,), float("nan"))})
