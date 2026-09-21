from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region, StateClock
from tidegraph.clocked_state import with_clock
from tidegraph.ops import Model
from tidegraph.native import Native
from read_cases import execute


@pytest.mark.parametrize("policy", [(1, 0, 1), (4, 0, 3), (4, 3, 1), (7, 2, 3)])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_clock_arithmetic_and_complete_cuts(policy, implementation):
    if implementation == "native":
        import _tide_native as core
        clock = core.StateClock(*policy)
    else:
        clock = StateClock(*policy)
    period, first, count = policy
    assert clock.to_local(-1) == clock.to_global(-1) == -1
    accepted = [t for t in range(60) if first <= t % period < first+count]
    for c in range(61):
        assert clock.cut(c) == sum(t < c for t in accepted)
    for local, global_time in enumerate(accepted):
        assert clock.to_local(global_time) == local and clock.to_global(local) == global_time
    for time in set(range(60))-set(accepted):
        with pytest.raises(ValueError, match="clock phases"):
            clock.to_local(time)


@pytest.mark.parametrize("policy", [(0, 0, 1), (-1, 0, 1), (4, -1, 2), (4, 4, 1), (4, 2, 3), (4, 0, 0)])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_invalid_clock_policy(policy, implementation):
    with pytest.raises(ValueError, match="state clock"):
        if implementation == "python":
            StateClock(*policy)
        else:
            import _tide_native as core
            core.StateClock(*policy).validate()


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_clock_int64_limits(implementation):
    if implementation == "native":
        import _tide_native as core
        make = core.StateClock
    else:
        make = StateClock
    maximum = 2**63-1
    identity = make(1, 0, 1)
    assert identity.to_global(maximum) == identity.to_local(maximum) == identity.cut(maximum) == maximum
    clock = make(maximum, maximum-1, 1)
    assert clock.to_global(0) == maximum-1 and clock.cut(maximum) == 1
    with pytest.raises(ValueError, match="overflow"):
        clock.to_global(1)
    for method in (identity.to_global, identity.to_local, identity.cut):
        with pytest.raises(ValueError, match="coordinate"):
            method(-2)


@pytest.mark.parametrize("policy", [(True, 0, 1), (4, False, 1), (4, 0, 1.0), (2**63, 0, 1)])
def test_python_clock_type_guards(policy):
    with pytest.raises(ValueError, match="clock"):
        StateClock(*policy)


@pytest.mark.parametrize("implementation", ["reference", "python-frontier", "native-serial", "native-packed", "native-frontier"])
def test_off_clock_events_and_persistent_states_rejected(dtype, implementation):
    clock = StateClock(4, 0, 3)
    g = Graph((Node(0, state_clock=clock),), (), (Region(1),), (0,), (0,))
    m = Model(g, dtype=dtype); q = Continuation(g.identity, 1)
    x = External(0, 0, 0, 3, torch.ones(3, dtype=dtype))
    with pytest.raises((ValueError, RuntimeError), match="clock phases"):
        execute(implementation, g, m, q, [x], stop=4)
    q.cut = 4; q.states[0, 0] = replace(m.nodes[0].initial(), last_time=3)
    with pytest.raises((ValueError, RuntimeError), match="clock phases"):
        execute(implementation, g, m, q, [], stop=4)


def test_clock_policy_and_shared_program_guards(dtype):
    clock = StateClock(4, 3, 1)
    with pytest.raises(ValueError, match="identity boundaries"):
        Graph((Node(0, identity=True, state_clock=clock),), (), (Region(1),), (0,), (0,))
    g = Graph((Node(0, state_clock=clock),), (), (Region(1),), (0,), (0,))
    m = Model(g, dtype=dtype); program = m.nodes[0].kernel
    assert with_clock(program, clock) is program
    with pytest.raises(ValueError, match="state clock"):
        with_clock(program, StateClock(4, 0, 3))
    changed = replace(g, nodes=(replace(g.nodes[0], state_clock=StateClock()),))
    with pytest.raises(ValueError, match="state clock"):
        Native(changed, m)


def test_callable_clock_subclass_is_not_silently_exported():
    class CustomClock(StateClock):
        def to_local(self, time):
            return time  # A distinct Python behavior, not the declared native policy.
    with pytest.raises(ValueError, match="state clock"):
        Graph((Node(0, state_clock=CustomClock(4, 0, 3)),), (), (Region(1),), (0,), (0,))
    from tidegraph.memory import EMA
    with pytest.raises(ValueError, match="StateClock"):
        with_clock(EMA(), CustomClock(4, 0, 3))
