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


@pytest.mark.parametrize("batched", [False, True])
def test_regional_prior_uses_all_real_incoming_proofs(device, batched):
    from tide.region import RegionNetwork, all_real_inputs_constant_zero
    from tide.region_batch import regional_scores

    net = RegionNetwork(64, "tiled").to(device)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.q.weight[0, 0] = 1
        net.k.weight[0, 1] = 100 * 32**0.5
        net.v.weight[0, 1] = net.o.weight[0, 0] = net.readout[0] = 1
    z = torch.zeros(2, 1, 4, 32, device=device)
    z[0, 0, 0, 0] = 1
    z[1, :, :, 1] = 1
    tiny = 2.**-30
    weights = torch.tensor([[[1., 1., 1., 0.]], [[0., 0., tiny, 0.]]], device=device)
    real = torch.tensor([[True, False, False], [True, True, False]], device=device)
    edges = torch.zeros(2, 1, 4, 3, device=device, dtype=torch.bool)
    edges[1, :, :, :2] = True
    edges[1, 0, 1, 1] = False
    proof = all_real_inputs_constant_zero(edges, real)
    assert proof[1, 0].tolist() == [True, False, True, True]

    def score(prior, face):
        if batched:
            stack = lambda x: torch.stack((x, x.flip((2,))))
            values = regional_scores([net, net], stack(z), stack(torch.zeros_like(prior)),
                                     stack(prior), None if face is None else stack(face))
            # Two independent region coordinates, both with an extreme closed key.
            return (values[0, 0] + values[1, 0].flip((1,))) * 0.5
        return net(z, torch.zeros_like(prior), prior, face)[0]

    weights.requires_grad_()
    with pytest.raises(ArithmeticError, match="nonfinite tiled attention key-prior VJP"):
        score(weights, None).sum().backward()
    values = score(weights, proof)
    dw, = torch.autograd.grad(values.sum(), weights)
    torch.testing.assert_close(values.flatten(), weights.new_tensor([0., 0., tiny/(1+tiny), 0.]), atol=0, rtol=1e-5)
    torch.testing.assert_close(dw[1, 0], weights.new_tensor([0., 1., 1/(1+tiny)**2, 0.]))


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


@pytest.mark.parametrize("inject", [True, False])
def test_timed_clock_proof_follows_current_bank_after_one_tick(device, inject):
    model = TideModel(ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-7",
        n=2, hubs=1, partition="singleton", timing="timed", period=8), d=64, allocation="shannon"))
    # Original r13 edge 0 -> 7 shares its slot with a dummy gather from layer 1.
    region = model.graph.nodes[7].region
    row = model.graph.regions[region].index(7)
    slot = next(i for i,e in enumerate(model.graph.incoming[7]) if model.graph.edges[e].src == 0)
    ticks = torch.arange(16, device=device)
    active = ((ticks == 0) | (ticks == 8)) & inject
    banks, pending = [], []
    for layer, nodes in enumerate(model.layer_nodes):
        strength = torch.zeros(len(nodes), 1, 16, device=device)
        if layer == 0:
            strength[:] = active
        strength.requires_grad_()
        content = torch.zeros(*strength.shape, 64, device=device)
        banks.append(LayerOutput(content, strength))
        # A caller-supplied differentiable zero at the preceding window boundary.
        pending.append(LayerOutput(content[:, :, :1], torch.zeros_like(strength[:, :, :1], requires_grad=True)))
    embedded = {"D": banks[0].values[0]}
    _, mass, _, face = model._incoming(region, banks, pending, embedded, 1, 16, device, inject,
        return_constant_zero=True, silent_ingress=model._silent_ingress_ticks(ticks, inject))
    assert not face[row, 0, 0, slot].item()
    assert torch.equal(face[row, 0, 1:, slot], ~active[:-1])
    assert torch.equal(mass[row, 0, 1:, slot], active[:-1].float())
    grad = torch.zeros_like(mass)
    grad[row, 0, 3, slot] = torch.inf
    unrelated, = torch.autograd.grad(mass, (banks[1].strengths,), grad)
    assert torch.equal(unrelated, torch.zeros_like(unrelated))


def test_independent_zero_with_unrepresentable_tiled_prior_vjp(device):
    q = torch.ones(1, 1, 1, device=device)
    k = torch.tensor([0., 100.], device=device).view(1, 2, 1)
    v = torch.tensor([0., 1.], device=device).view_as(k)
    weights = torch.tensor([1., 0.], device=device, requires_grad=True)
    # At zero the right derivative of t*exp(100)/(1+t*exp(100)) is exp(100).
    with pytest.raises(ArithmeticError, match="tiled attention key-prior VJP.*float32"):
        weighted_attention(q, k, v, weights, backend="tiled").sum().backward()


def test_timed_shared_sender_finite_branches_overflow_only_on_a_free_root(device):
    model = TideModel(ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-7",
        n=2, hubs=1, partition="singleton", timing="timed", period=8), d=64, allocation="shannon"))
    ticks = torch.arange(8, device=device)
    silent = model._silent_ingress_ticks(ticks, True)
    nodes = model.layer_nodes[0]
    live = (~silent).float().expand(len(nodes), 1, 8).clone().requires_grad_()
    contents = torch.zeros(*live.shape, 64, device=device)
    prior = torch.zeros(len(nodes), 1, 1, device=device, requires_grad=True)
    layers = [LayerOutput(contents, live)]
    pending = [LayerOutput(contents[:, :, :1], prior)]
    samples = []
    for receiver in (3, 5):
        region = model.graph.nodes[receiver].region
        row = model.graph.regions[region].index(receiver)
        slot = next(i for i,e in enumerate(model.graph.incoming[receiver]) if model.graph.edges[e].src == 0)
        _, mass, _, _ = model._incoming(region, layers, pending, {"D":contents[0]}, 1, 8, device,
                                        return_constant_zero=True, silent_ingress=silent)
        samples.append(mass[row, 0, :, slot])
    # Each consumer returns a finite cotangent. The silent source would receive
    # 4e38 if the projection happened only after their gradients were added.
    seed = torch.tensor([1., 1., 2e38, 0., 0., 0., 0., 0.], device=device)
    g_live, g_prior = torch.autograd.grad(samples, (live, prior), (seed, seed))
    assert torch.isfinite(g_live).all() and g_live[0, 0, 1] == 0
    assert g_live[0, 0, 0] == 2 and g_prior[0, 0, 0] == 2


def test_derived_sender_only_uses_real_proven_inputs_and_follows_attached_tail(tmp_path, device):
    from dataclasses import fields
    from tide.placement import move_tensors
    model = padded_model()
    region = next(r for r in range(len(model.graph.regions)) if model._reception_mask((r,), device) is not None)
    real = model._reception_mask((region,), device)
    faces = real[:, None, None, :].expand(-1, 1, 2, -1).clone()
    faces[0, :, 0, 0] = False
    strengths = torch.zeros(faces.shape[:-1], device=device, requires_grad=True) * 2
    values = torch.zeros(*strengths.shape, 64, device=device)
    bank = model._sender_output(region, values, strengths, None, faces, None)
    assert not bank.routing_face()[0, 0, 0] and bank.routing_face()[:, :, 1].all()
    tail = move_tensors(LayerOutput.cat([bank]).tail().as_pending(), device)
    assert tail.routing_face().all()
    assert [f.name for f in fields(tail)] == ["values", "strengths"]
    path = tmp_path / "attached.pt"
    torch.save(tail, path)
    restored = torch.load(path, weights_only=False)
    assert restored.routing_face() is None
    tail.strengths = tail.strengths.detach().requires_grad_()
    assert tail.routing_face() is None


def test_mean_distinguishes_fixed_and_movable_zero_from_padding(device):
    from tide.reception import Reception
    recv = Reception([{"ext"}], 64, "source_residual").to(device)
    with torch.no_grad():
        for p in recv.parameters():
            p.zero_()
    owners = torch.tensor([0], device=device)
    classes = torch.zeros(1, 4, dtype=torch.long, device=device)
    content = torch.zeros(1, 1, 1, 4, 64, device=device)
    content[..., 0] = content.new_tensor([1., 2., 4., 8.])
    mass = content.new_tensor([0., 0., 2.**-100, 0.]).view(1, 1, 1, 4).requires_grad_()
    membership = torch.tensor([[True, True, True, False]], device=device)
    # A stale proof on the tiny *positive* sender must not remove its value.
    fixed = torch.tensor([True, False, True, False], device=device).reshape_as(mass)
    _, value = recv(content, mass, classes, owners, membership, fixed)
    grad, = torch.autograd.grad(value[..., 0].sum(), mass)
    assert value[..., 0].item() == 4.
    torch.testing.assert_close(grad.flatten(), mass.new_tensor([0., -2.**101, 0., 0.]))
