"""Independent fixed-math reception formulas, including every gradient root."""

import math

import pytest
import torch

from tide.reception import Reception


PROFILES = ("source_residual", "incoming_light_64", "incoming_light_128", "incoming_wide")
WIDTH = 64


def _fixture(profile, shared, qkv_backend="separate"):
    groups = [0, 0, 1] if shared else None
    module = Reception([{"DD", "HD", "ext"}] * 3, WIDTH, profile,
                       groups=groups, rank=4, backend="math", qkv_backend=qkv_backend).double()
    generator = torch.Generator().manual_seed(903)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64) * .09)
    return module, groups


def _formula(content, mass, classes, owners, parameters, profile, groups):
    """Literal node matrices and exponential quotients; no module helpers."""
    d_a = WIDTH if profile == "incoming_wide" else (
        int(profile.rsplit("_", 1)[1]) if profile != "source_residual" else 0)

    def normalize(x):
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)

    def linear(x, name, owner):
        group = owner if groups is None else groups[owner]
        matrix = parameters[f"{name}.weight"][group]
        if groups is not None:
            matrix = matrix + parameters[f"{name}.left"][owner] @ parameters[f"{name}.right"][owner]
        return x @ matrix.T

    totals, aggregates = [], []
    for slot, owner in enumerate(owners.tolist()):
        group = owner if groups is None else groups[owner]
        # Every fixture owner has the same three source classes, so each
        # parameter group owns one contiguous DD/HD/ext embedding triple.
        embedding_ids = group * 3 + classes[slot]
        rows, mass_rows = [], []
        for batch in range(content.shape[1]):
            positions, masses = [], []
            for time in range(content.shape[2]):
                z = content[slot, batch, time]
                weights = mass[slot, batch, time].clamp_min(0)
                total = weights.sum()
                y = z
                if profile != "source_residual":
                    x = normalize(z) + parameters["sa_embedding.weight"][embedding_ids]
                    q, k, v = [linear(x, name, owner).reshape(len(z), d_a // 32, 32).transpose(0, 1)
                               for name in ("q", "k", "v")]
                    exponential = torch.exp((q @ k.transpose(-2, -1)) / math.sqrt(32))
                    terms = exponential * weights[None, None, :]
                    denominator = terms.sum(-1, keepdim=True)
                    probability = terms / torch.where(denominator > 0, denominator, 1)
                    read = (probability @ v).transpose(0, 1).reshape(len(z), d_a)
                    y = z + linear(read, "o", owner)
                hidden = linear(normalize(y), "a", owner) + parameters["embedding.weight"][embedding_ids]
                transformed = y + linear(hidden * torch.sigmoid(hidden), "b", owner)
                value = (weights[:, None] * transformed).sum(0) / torch.where(total > 0, total, 1)
                positions.append(value)
                masses.append(total)
            rows.append(torch.stack(positions))
            mass_rows.append(torch.stack(masses))
        aggregates.append(torch.stack(rows))
        totals.append(torch.stack(mass_rows))
    return torch.stack(totals), torch.stack(aggregates)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("qkv_backend", ["separate", "fused"])
def test_all_profiles_match_literal_values_and_all_root_vjps(profile, shared, qkv_backend):
    module, groups = _fixture(profile, shared, qkv_backend)
    generator = torch.Generator().manual_seed(138)
    owners = torch.tensor([2, 0, 1, 0])  # Repeated owner and shared bases sum real contributions.
    classes = torch.tensor([[0, 1, 2], [2, 1, 0], [1, 0, 2], [0, 2, 1]])
    content = torch.randn(4, 1, 2, 3, WIDTH, generator=generator, dtype=torch.float64).requires_grad_()
    mass = (.2 + torch.rand(4, 1, 2, 3, generator=generator, dtype=torch.float64))
    mass[0, 0, 0, 2], mass[2, 0, 1, 0] = 0., -.25
    mass.requires_grad_()
    parameters = dict(module.named_parameters())
    actual = module(content, mass, classes, owners)
    reference = _formula(content, mass, classes, owners, parameters, profile, groups)
    for got, want in zip(actual, reference):
        torch.testing.assert_close(got, want, atol=2e-13, rtol=2e-12)
    seeds = [torch.randn(value.shape, generator=generator, dtype=torch.float64) for value in actual]
    roots = (content, mass, *parameters.values())
    got = torch.autograd.grad(actual, roots, grad_outputs=seeds, retain_graph=True)
    expected = torch.autograd.grad(reference, roots, grad_outputs=seeds)
    for name, a, b in zip(("content", "mass", *parameters), got, expected):
        torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-11, msg=lambda message: f"{name}: {message}")
    assert got[1][2, 0, 1, 0] == 0  # A strictly negative raw mass is clipped away.
    assert module.a.out_features == (WIDTH if profile == "incoming_wide" else WIDTH // 8)
    if profile != "source_residual":
        expected_width = WIDTH if profile == "incoming_wide" else int(profile.rsplit("_", 1)[1])
        assert module.q.out_features == expected_width and module.heads == expected_width // 32
        assert module.effective_qkv_backend == qkv_backend


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("qkv_backend", ["separate", "fused"])
@pytest.mark.parametrize("zero_real", [False, True])
def test_structural_padding_matches_deleted_edges_and_keeps_real_zero_vjp(profile, qkv_backend, zero_real):
    """The reference removes padded atoms before constructing any Attention.

    Fake padded mass is deliberately nonzero: structural membership, rather
    than a runtime mass comparison, defines the derivative domain.
    """
    module, groups = _fixture(profile, True, qkv_backend)
    module.backend = "auto"
    generator = torch.Generator().manual_seed(713)
    owners = torch.tensor([0, 2, 0])
    classes = torch.tensor([[0, 1, 2, 0], [1, 0, 2, 1], [0, 2, 1, 0]])
    edge_mask = torch.tensor([[True, False, True, False],
                              [True, True, True, False],
                              [True, True, False, False]])
    content = torch.randn(3, 1, 2, 4, WIDTH, generator=generator, dtype=torch.float64).requires_grad_()
    mass = .2 + torch.rand(3, 1, 2, 4, generator=generator, dtype=torch.float64)
    if zero_real:
        mass[0, 0, 0, 2] = 0
    mass.requires_grad_()
    parameters = dict(module.named_parameters())
    actual = module(content, mass, classes, owners, edge_mask)
    pieces = [_formula(content[i:i+1, ..., visible, :], mass[i:i+1, ..., visible],
                       classes[i:i+1, visible], owners[i:i+1], parameters, profile, groups)
              for i, visible in enumerate(edge_mask)]
    reference = tuple(torch.cat([piece[j] for piece in pieces]) for j in range(2))
    for got, expected in zip(actual, reference):
        torch.testing.assert_close(got, expected, atol=2e-13, rtol=2e-12)
    probes = [torch.randn(value.shape, generator=generator, dtype=value.dtype) for value in actual]
    roots = (content, mass, *parameters.values())
    actual_grads = torch.autograd.grad(actual, roots, grad_outputs=probes, retain_graph=True)
    expected_grads = torch.autograd.grad(reference, roots, grad_outputs=probes)
    for name, got, expected in zip(("content", "mass", *parameters), actual_grads, expected_grads):
        torch.testing.assert_close(got, expected, atol=3e-12, rtol=3e-11,
                                   msg=lambda message: f"{name}: {message}")
    assert torch.equal(actual_grads[1][~edge_mask[:, None, None].expand_as(mass)],
                       torch.zeros_like(actual_grads[1][~edge_mask[:, None, None].expand_as(mass)]))
    if zero_real:
        # A real, currently zero sender is still a differentiable prior. A
        # blanket mass>0 Attention mask would incorrectly erase this term.
        assert actual_grads[1][0, 0, 0, 2].abs().item() > 1e-4
        epsilon = 1e-7
        with torch.no_grad():
            moved = mass.detach().clone()
            moved[0, 0, 0, 2] = epsilon
            values = module(content.detach(), moved, classes, owners, edge_mask)
            right_difference = sum(((new-old.detach())*probe).sum()
                                   for new, old, probe in zip(values, actual, probes)) / epsilon
        torch.testing.assert_close(actual_grads[1][0, 0, 0, 2], right_difference, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("profile", PROFILES)
def test_absent_atom_and_zero_slot_preserve_existing_roots_but_have_different_mass_domains(profile):
    module, _ = _fixture(profile, True)
    generator = torch.Generator().manual_seed(49)
    owners, classes = torch.tensor([0]), torch.tensor([[0, 1, 2]])
    content = torch.randn(1, 1, 1, 3, WIDTH, generator=generator, dtype=torch.float64).requires_grad_()
    mass = torch.tensor([[[[.7, .3, 0.]]]], dtype=torch.float64, requires_grad=True)
    probe = torch.randn(WIDTH, generator=generator, dtype=torch.float64)
    parameters = tuple(module.parameters())
    full = module(content, mass, classes, owners)
    short_z = content.detach()[..., :2, :].clone().requires_grad_()
    short_m = mass.detach()[..., :2].clone().requires_grad_()
    absent = module(short_z, short_m, classes[:, :2], owners)
    for a, b in zip(full, absent):
        torch.testing.assert_close(a, b, atol=2e-13, rtol=2e-12)
    loss = .37 * full[0].sum() + (full[1] * probe).sum()
    short_loss = .37 * absent[0].sum() + (absent[1] * probe).sum()
    gradients = torch.autograd.grad(loss, (content, mass, *parameters))
    short_gradients = torch.autograd.grad(short_loss, (short_z, short_m, *parameters))
    torch.testing.assert_close(gradients[0][..., :2, :], short_gradients[0], atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(gradients[1][..., :2], short_gradients[1], atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(gradients[0][..., 2, :], torch.zeros_like(content[..., 2, :]))
    for a, b in zip(gradients[2:], short_gradients[2:]):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-11)
    # The inserted mass is an additional independent root. At a nonempty
    # fiber its positive-side derivative exists and must not be masked away.
    epsilon = 1e-7
    with torch.no_grad():
        positive = mass.detach().clone()
        positive[..., 2] = epsilon
        moved = module(content.detach(), positive, classes, owners)
        right_difference = (.37 * moved[0].sum() + (moved[1] * probe).sum() - loss.detach()) / epsilon
    torch.testing.assert_close(gradients[1][..., 2].squeeze(), right_difference, atol=2e-6, rtol=2e-5)
    assert gradients[1][..., 2].abs().item() > 1e-4


@pytest.mark.parametrize("profile", PROFILES)
def test_empty_and_entirely_zero_fibers_have_zero_value_and_content_parameter_vjp(profile):
    module, _ = _fixture(profile, True)
    owners = torch.tensor([0])
    content = torch.randn(1, 1, 1, 3, WIDTH, dtype=torch.float64, requires_grad=True)
    zero = module(content, torch.zeros(1, 1, 1, 3, dtype=torch.float64), torch.tensor([[0, 1, 2]]), owners)
    empty = module(content[..., :0, :], torch.zeros(1, 1, 1, 0, dtype=torch.float64),
                   torch.empty(1, 0, dtype=torch.long), owners)
    for value, absent in zip(zero, empty):
        torch.testing.assert_close(value, torch.zeros_like(value))
        torch.testing.assert_close(value, absent)
    gradients = torch.autograd.grad(zero[1].sum(), (content, *module.parameters()))
    assert all(torch.equal(gradient, torch.zeros_like(gradient)) for gradient in gradients)
    # At total mass zero normalized content has no general directional
    # derivative; this test deliberately makes no mass-VJP claim there.
