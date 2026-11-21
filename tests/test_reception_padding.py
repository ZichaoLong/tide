"""Absent graph edges are not zero-prior derivative roots."""
import os

import pytest
import torch

from tide import ModelConfig, TideModel, TopologyConfig
from tide.attention import weighted_attention
from tide.model import LayerOutput


@pytest.fixture(scope="module")
def device():
    name = os.environ.get("TIDE_STABLE_DEVICE", "cpu")
    if name.startswith("npu"):
        import torch_npu  # noqa: F401
    return torch.device(name)


def padded_model(legacy=False):
    return TideModel(ModelConfig(
        topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4",
                                n=2, hubs=1, partition="mixed", timing="settle"),
        d=64, aggregation="incoming_light_64", reception_mask_padding=legacy))


@pytest.mark.parametrize("legacy", [False, True])
def test_membership_uses_real_graph_edges_in_exact_owner_order(device, legacy):
    model = padded_model(legacy)
    saw_padding = False
    for region, nodes in enumerate(model.graph.regions):
        degrees = [max(1, len(model.graph.incoming[node])) for node in nodes]
        expected = torch.arange(max(degrees))[None] < torch.tensor(degrees)[:, None]
        actual = model._reception_mask((region,), device)
        if bool(expected.all()):
            assert actual is None
        else:
            saw_padding = True
            assert torch.equal(actual.cpu(), expected)
            assert not actual.is_inference()
    assert saw_padding
    # Preserve explicit legacy data, without allowing it to disable membership.
    assert ModelConfig.from_dict(model.config.to_dict()).reception_mask_padding is legacy


@pytest.mark.parametrize("backend", ["math", "tiled", "auto"])
def test_routed_padding_cannot_poison_shared_sender_gradients(device, backend):
    model = padded_model(False)
    region = next(r for r in range(len(model.graph.regions))
                  if model._reception_mask((r,), "cpu") is not None)
    membership = model._reception_mask((region,), device)
    layers = [LayerOutput(torch.zeros(len(nodes), 1, 1, 64, device=device),
                          torch.ones(len(nodes), 1, 1, device=device, requires_grad=True))
              for nodes in model.layer_nodes]
    embedded = {"D": torch.zeros(1, 1, 64, device=device)}
    _, mass, _ = model._incoming(region, layers, None, embedded, 1, 1, device)
    n, width = membership.shape
    q = torch.zeros(n, 1, 1, 32, device=device)
    q[..., 0] = 1
    k = torch.zeros(n, 1, width, 32, device=device)
    k[..., 0] = torch.where(membership[:, None], -110 * 32**.5, 0)
    v = membership[:, None, :, None].expand(n, 1, width, 32).float()
    output = weighted_attention(q, k, v, mass.reshape(n, 1, 1, width),
                                membership[:, None, None], backend)
    # The literal quotient is identically one for any positive real masses:
    # d(output)/d(real sender strength) = 0. Treating a pad as a real zero
    # instead gives a hypothetical derivative larger than FP32's range,
    # then the actual routing multiply would evaluate Inf * 0.
    torch.testing.assert_close(output, torch.ones_like(output), atol=1e-5, rtol=1e-4)
    root = torch.zeros_like(output)
    root[..., 0] = 3e-6
    used = sorted(model._routes[region][1])
    gradients = torch.autograd.grad(output, [layers[i].strengths for i in used], root)
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-10, rtol=0)


@pytest.mark.parametrize("backend", ["math", "tiled", "auto"])
def test_real_zero_prior_keeps_right_derivative_while_padding_has_none(device, backend):
    q = torch.zeros(1, 1, 1, 32, device=device)
    k = torch.zeros(1, 1, 3, 32, device=device)
    v = torch.tensor([1., 3., 7.], device=device).reshape(1, 1, 3, 1).expand_as(k)
    w = torch.tensor([1., 0., 0.], device=device, requires_grad=True)
    # Second slot is a real zero edge; third is absent. The independent
    # derivative of (1 + 3*t)/(1 + t) at zero is 2.
    mask = torch.tensor([True, True, False], device=device)
    output = weighted_attention(q, k, v, w, mask, backend)
    dw, = torch.autograd.grad(output[..., 0].sum(), (w,))
    torch.testing.assert_close(dw.cpu(), torch.tensor([0., 2., 0.]), atol=0, rtol=0)


def test_legacy_false_is_masked_during_whole_region_recomputation():
    from dataclasses import replace
    torch.manual_seed(524)
    model = padded_model(False)
    model = TideModel(replace(model.config, checkpoint_regions=True, region_batch_size=4))
    calls = []

    def observe(module, args):
        calls.append(args[4])

    handle = model.receive.register_forward_pre_hook(observe)
    try:
        output = model(torch.randn(1, 2, 64))
        output.readout().square().mean().backward()
    finally:
        handle.remove()
    assert any(mask is not None and bool((~mask).any()) for mask in calls)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("backend", ["auto", "math", "tiled"])
def test_ordinary_quad_constant_face_large_logit_keeps_other_roots(device, backend):
    from tide.region import region_control
    scores = torch.tensor([-3., 5.], device=device, requires_grad=True)
    received = torch.ones_like(scores, requires_grad=True)
    control, face = region_control(scores[:, None, None], received[:, None, None],
        torch.zeros(2, dtype=torch.bool, device=device), 1, "quad", "group", return_constant_zero=True)
    assert face.flatten().tolist() == [True, False]
    assert control.flatten().tolist() == [0., 1.]
    # Independent zero and weak positive priors sit beside the unreachable root.
    independent = torch.tensor([0., 1e-30], device=device, requires_grad=True)
    weights = torch.cat((control.flatten(), independent))
    marked = torch.cat((face.flatten(), torch.tensor([False, True], device=device)))
    q = torch.ones((1, 1, 1), device=device, requires_grad=True)
    k = torch.tensor([110., 0., 0., 0.], device=device).view(1, 4, 1).requires_grad_()
    v = torch.tensor([3., 1., 4., 6.], device=device).view_as(k).requires_grad_()
    out = weighted_attention(q, k, v, weights, backend=backend, _constant_zero_prior=marked)
    ds, dr, di, dq, dk, dv = torch.autograd.grad(out.sum(), (scores, received, independent, q, k, v))
    torch.testing.assert_close(out, torch.ones_like(out))
    for gradient in (ds, dr, dq, dk, dv):
        assert torch.isfinite(gradient).all()
    torch.testing.assert_close(di.cpu(), torch.tensor([3., 5.]))
    assert ds.count_nonzero() == 0 and dv[0, 0, 0] == 0


def test_fixed_through_quad_zero_is_not_a_constant_lower_face(device):
    from tide.region import region_control
    received = torch.tensor([0., 1.], device=device, requires_grad=True)
    scores = torch.tensor([10., 1.], device=device)
    control, face = region_control(scores[:, None, None], received[:, None, None],
        torch.zeros(2, dtype=torch.bool, device=device), 1, "quad", "group", return_constant_zero=True)
    assert control[0].item() == 0 and not face[0].item()
    gradient, = torch.autograd.grad(control[0].sum(), received)
    torch.testing.assert_close(gradient.cpu(), torch.tensor([2., 0.]))


def test_timed_pending_checkpoint_and_independent_roots(tmp_path, device):
    from dataclasses import fields, replace
    from tide.placement import move_tensors
    from tide import TopologyConfig

    model = TideModel(replace(padded_model().config, topology=TopologyConfig(
        kind="multiscale", profile="compact-5-dd4", n=2, hubs=1, partition="mixed", timing="timed")))
    region = model.layer_regions[1][0]
    banks, pending = [], []
    for nodes in model.layer_nodes:
        strengths = torch.zeros(len(nodes), 1, 2, device=device)
        values = torch.zeros(*strengths.shape, 64, device=device)
        banks.append(LayerOutput.from_region(values, strengths + 1, None))
        pending.append(LayerOutput(values[:, :, :1], strengths[:, :, :1]))
    # Legacy checkpoint payloads hash the original two dataclass fields.
    path = tmp_path / "timed.pt"
    torch.save(pending, path)
    pending = torch.load(path, weights_only=False)
    assert [f.name for f in fields(pending[0])] == ["values", "strengths"]
    embedded = {"D": torch.zeros(1, 2, 64, device=device)}
    def routed():
        return model._incoming(region, banks, pending, embedded, 1, 2, device, return_constant_zero=True)
    _, mass, _, face = routed()
    assert (mass[:, :, 0] == 0).all() and face[:, :, 0].any() and not face[:, :, 1].any()
    for index, bank in enumerate(pending):
        independent = bank.strengths.detach().requires_grad_()
        rerooted = LayerOutput.from_region(bank.values, independent, torch.ones_like(independent, dtype=torch.bool))
        # Both slicing and device/index packing can turn an input leaf into a nonleaf.
        ids = torch.arange(len(independent), device=device)
        pending[index] = LayerOutput.cat([move_tensors(rerooted, device).index_select(ids).select(slice(None))])
    _, mass, _, face = routed()
    assert mass.requires_grad and face is None
