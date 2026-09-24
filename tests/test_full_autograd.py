"""Batched Full VJP vs the independently scheduled Python scalar interpreter."""
import dataclasses
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from isolated_cases import fixture, roots, vjp


@pytest.mark.parametrize('algorithm', ['streaming', 'frontier'])
@pytest.mark.parametrize('mode', ['hard', 'softp', 'hst'])
@pytest.mark.parametrize('full', ['tanh', 'swiglu', 'lh-silu-rms-v1', 'lh-identity-identity-v1'])
@pytest.mark.parametrize('emission', ['broadcast', 'slot_affine'])
def test_full_all_observables_and_isolated_roots(dtype, algorithm, mode, full, emission):
    def execute(native):
        g, _, q, xs, variables = fixture(dtype, 'ssm')
        g = dataclasses.replace(g, nodes=tuple(dataclasses.replace(n, full=full, emission=emission) for n in g.nodes))
        m = Model(g, width=4, dtype=dtype)
        # Share owners across nodes, with other never-touched owners kept unused.
        m.nodes[1].weight = m.nodes[0].weight
        q.identity = g.identity
        variables = dict(m.named_parameters()) | {k: v for k, v in variables.items() if k.startswith(('initial.', 'input.', 'upstream.'))}
        if native:
            result = Native(g, m, packed=True, workers=3, algorithm=algorithm, mode=mode, full_autograd='batched').run(q, xs, 4, sealed_until=4)
            assert result.stats.get('semantic_full_replays', 0) == 0
            assert result.stats['batched_full_events'] > 0
        else:
            result = run(g, m, q, xs, 4, sealed_until=4, mode=mode)
        rs = roots(result)
        first = next(e for e in result.trace if e['batch'] == 0 and e['active'])
        rs['full'] = first['full']
        rs.update({'emit.'+str(k): v for k, v in first['emitted'].items()})
        grads = {(name, zero): vjp(root, variables, zero) for name, root in rs.items() for zero in (False, True)}
        return result, grads
    equivalent(execute(False), execute(True))
