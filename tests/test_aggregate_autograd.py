"""Packed VJP versus independent scalar schedules and all public roots."""
import pytest
import torch
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.reference import run
from aggregate_cases import fixture
from isolated_cases import vjp
from foundation_training import trajectory


@pytest.mark.parametrize('kind', ['sum', 'mean', 'weighted_mean', 'active_softmax', 'all_softmax'])
@pytest.mark.parametrize('mode', ['hard', 'softp', 'hst'])
@pytest.mark.parametrize('schedule,workers', [('streaming', 1), ('streaming', 3), ('frontier', 3)])
def test_all_observables_and_each_source_root(dtype, kind, mode, schedule, workers):
    def execute(native):
        g, m, q, xs, variables = fixture(dtype, kind, budget=1)
        if native:
            result = Native(g, m, packed=True, workers=workers, algorithm=schedule, mode=mode,
                            full_autograd='batched', aggregate_autograd='batched',
                            packed_sources=mode == 'hst').run(q, xs, 5, sealed_until=5)
            assert result.stats.get('semantic_aggregate_replays', 0) == 0
            assert result.stats['batched_aggregate_events'] == result.stats['candidate_events']
        else:
            result = run(g, m, q, xs, 5, sealed_until=5, mode=mode)
        first = result.trace[0]
        roots = [objective(result), result.outputs[0][-1], first['content'],
                 *first['contributions'].values(), result.continuation.pending[0].value]
        return result, [vjp(root, variables, zero) for root in roots for zero in (False, True)]
    equivalent(execute(False), execute(True))


@pytest.mark.parametrize('family,implementation', [('pdg','native-packed'), ('pdg','native-specialized'),
    ('dag','native-packed'), ('dag','native-specialized'), ('settle','native-packed')])
@pytest.mark.parametrize('kind', ['sgd', 'adamw'])
def test_training_detach_owners_and_native_schedules(dtype, family, implementation, kind):
    expected = trajectory(dtype, family, 'reference', 'ssm', kind)
    actual = trajectory(dtype, family, implementation, 'ssm', kind, native_optimizer=True,
                        full_autograd='batched', aggregate_autograd='batched')
    equivalent(expected, actual)


def test_switch_policy_at_complete_cut_retains_gradients(dtype):
    def execute(native):
        g, m, q, xs, variables = fixture(dtype, 'all_softmax')
        if native:
            for stop, policy in [(2, 'batched'), (3, 'replay'), (5, 'batched')]:
                chunk = [x for x in xs if q.cut <= x.time < stop]
                result = Native(g, m, packed=True, aggregate_autograd=policy, full_autograd='batched', mode='hst').run(
                    q, chunk, stop, sealed_until=stop)
                q = result.continuation
        else:
            result = run(g, m, q, xs, 5, sealed_until=5, mode='hst')
        return result.continuation, vjp(objective(result, 'state'), variables)
    equivalent(execute(False), execute(True))


@pytest.mark.parametrize('context', [torch.no_grad, torch.inference_mode])
def test_nograd_keeps_original_numeric_path(dtype, context):
    with context():
        def execute(policy):
            g, m, q, xs, _ = fixture(dtype, 'all_softmax')
            return Native(g, m, packed=True, aggregate_autograd=policy).run(q, xs, 5, sealed_until=5)
        a, b = execute('replay'), execute('batched')
        equivalent(a, b)
        assert not b.stats.get('batched_aggregate_events', 0)
        assert not b.stats.get('semantic_aggregate_replays', 0)


@pytest.mark.parametrize('policy,packed', [('bad',True), ('batched',False)])
def test_reject_unsupported_aggregate_policy(dtype, policy, packed):
    g, m, *_ = fixture(dtype, 'all_softmax')
    with pytest.raises(ValueError, match='Aggregate autograd'):
        Native(g, m, packed=packed, aggregate_autograd=policy)
