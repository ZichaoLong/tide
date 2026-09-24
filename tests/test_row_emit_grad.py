"""Scale RowEmit's Full and each phased port are separate first-order roots."""
import pytest
import torch
import _tide_native as core
from tidegraph.compare import equivalent


@pytest.mark.parametrize('policy', ['replay', 'batched'])
@pytest.mark.parametrize('targets', [0, 2])
@pytest.mark.parametrize('frozen', [False, True])
def test_row_emit_full_ports_vjp(dtype, policy, targets, frozen):
    def run(variant):
        x = [torch.linspace(-.5+i/9, .7, 4, dtype=dtype, requires_grad=i != 1 or not frozen) for i in range(3)]
        norm = torch.linspace(.7, 1.3, 4, dtype=dtype, requires_grad=not frozen)
        weight = torch.arange(max(1, targets)*16, dtype=dtype).reshape(-1, 4).div(35).requires_grad_(not frozen)
        w = core.NodeWeights(torch.tensor(0., dtype=dtype), torch.eye(4, dtype=dtype), torch.zeros(4, dtype=dtype), torch.ones(4, dtype=dtype))
        w.full_kind = 'lh-silu-rms-v1'; w.extra = {'lh_norm_weight': norm, 'row_emit_weight': weight}
        logical = [-1, 0, 1, 0] if targets else [-1, -1]
        phases = [0, 0, 1, 1] if targets else [0, 1]
        times = [0, 1, 2]
        if variant == 'python':
            fresh = [torch.nn.functional.rms_norm(torch.nn.functional.silu(v), (4,), norm, 1e-7) for v in x]
            projections = [torch.nn.functional.linear(v, weight) for v in fresh]
            output = [(fresh[i], {s: fresh[i] if l < 0 else projections[i][l*4:(l+1)*4]
                                 for s, l in enumerate(logical) if times[i]%2 == phases[s]}) for i in range(3)]
        else:
            output = core.row_emit_probe(w, [v*1 for v in x], times, logical, phases, 2, targets, variant)
        variables = [v for v in [*x, norm, weight] if v.requires_grad]
        vjps = []
        for i, (full, ports) in enumerate(output):
            for root in [full, *ports.values()]:
                if not root.requires_grad:
                    vjps.append(None); continue
                for zero in (False, True):
                    grads = torch.autograd.grad(root.square().sum()*(0 if zero else .7), variables, allow_unused=True, retain_graph=True)
                    for j, v in enumerate(x):
                        if i != j and v.requires_grad:
                            assert grads[next(k for k, a in enumerate(variables) if a is v)] is None
                    vjps.append(grads)
        return output, vjps, [[r.requires_grad for r in [full, *ports.values()]] for full, ports in output]
    expected = run('python')
    equivalent(expected, run('scalar'))
    equivalent(expected, run(policy))
