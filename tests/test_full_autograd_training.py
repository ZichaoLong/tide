import pytest
from tidegraph.compare import equivalent
from tidegraph.native import Native
from foundation_training import trajectory
from isolated_cases import fixture, vjp
from tidegraph.reference import run


@pytest.mark.parametrize('family,implementation', [('pdg', 'native-packed'), ('pdg', 'native-specialized'),
    ('dag', 'native-packed'), ('dag', 'native-specialized'), ('settle', 'native-packed')])
@pytest.mark.parametrize('memory', ['ssm', 'attention'])
@pytest.mark.parametrize('kind', ['sgd', 'adamw'])
def test_batched_full_training_cuts_and_owners(dtype, family, implementation, memory, kind):
    expected = trajectory(dtype, family, 'reference', memory, kind)
    actual = trajectory(dtype, family, implementation, memory, kind, native_optimizer=True, full_autograd='batched')
    equivalent(expected, actual)


@pytest.mark.parametrize('kind', ['ema', 'ssm', 'attention', 'linear', 'delta'])
def test_switching_policy_at_complete_cuts_retains_graph(dtype, kind):
    def execute(native):
        g, m, q, xs, variables = fixture(dtype, kind)
        if not native:
            result = run(g, m, q, xs, 9, sealed_until=9, mode='hst')
        else:
            for stop, policy in [(2, 'batched'), (4, 'replay'), (9, 'batched')]:
                chunk = [x for x in xs if q.cut <= x.time < stop]
                result = Native(g, m, workers=3, packed=True, mode='hst', full_autograd=policy).run(q, chunk, stop, sealed_until=stop)
                q = result.continuation
        state = result.continuation.states[0, 2].value
        return result.continuation, vjp(state, variables), vjp(state, variables, True)
    equivalent(execute(False), execute(True))


@pytest.mark.parametrize('policy,packed', [('unknown', True), ('batched', False)])
def test_reject_invalid_policy_before_execution(dtype, policy, packed):
    g, m, *_ = fixture(dtype, 'ssm')
    with pytest.raises(ValueError, match='Full autograd'):
        Native(g, m, packed=packed, full_autograd=policy)
