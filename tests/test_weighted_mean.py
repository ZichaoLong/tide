"""Architecture §2: independent normalization identities and directional roots."""
import torch
from tidegraph.weighted.mean import weighted_mean


def test_singleton_mean_mass_invariance_under_large_cotangent(dtype):
    mass = torch.tensor([.35], dtype=dtype, requires_grad=True)
    value = torch.tensor([[.7, -.4]], dtype=dtype, requires_grad=True)
    mean = weighted_mean(mass, value)
    # Forward keeps the original floating quotient, including its rounding.
    assert torch.equal(mean, (mass[:, None]*value).sum(0)/mass.sum())
    bar = value.new_tensor([1.e5, -2.e5])
    dm, dz = torch.autograd.grad(mean, (mass, value), bar)
    assert torch.equal(dm, torch.zeros_like(mass))
    assert torch.equal(dz, bar[None])


def test_weighted_mean_ordinary_directional_derivative():
    masses = torch.tensor([[.2, 0., .8], [.5, .7, .3]], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([[[.5, -.2], [1., .4], [-.3, .7]], [[.1, .8], [.4, .6], [-.9, .3]]], dtype=torch.float64, requires_grad=True)
    dm = torch.tensor([[.1, 0., -.2], [.05, -.03, .07]], dtype=torch.float64)
    dz = torch.linspace(-.2, .3, values.numel(), dtype=torch.float64).reshape_as(values)
    bar = torch.tensor([[.7, -.3], [.5, .9]], dtype=torch.float64)
    gradients = torch.autograd.grad(weighted_mean(masses, values), (masses, values), bar)
    analytic = (gradients[0]*dm).sum()+(gradients[1]*dz).sum()
    # Independent scalar quotients at perturbed points, no helper invocation.
    def at(delta):
        m, v = masses.detach()+delta*dm, values.detach()+delta*dz
        return sum(sum(float(m[i,j])*float(v[i,j,k]) for j in range(3))/sum(float(x) for x in m[i])*float(bar[i,k])
                   for i in range(2) for k in range(2))
    torch.testing.assert_close(analytic, analytic.new_tensor((at(1e-5)-at(-1e-5))/2e-5), atol=2e-10, rtol=1e-8)
