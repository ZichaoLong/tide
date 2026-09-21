import pytest
from tidegraph.compare import equivalent
from single_graph_cases import PROFILES
from single_graph_training import Case, TwoClock, Encoded, IMPLEMENTATIONS, check
from single_graph_roots import check_roots, gradients


@pytest.mark.parametrize("pool", PROFILES)
@pytest.mark.parametrize("mode", ["hard", "softp", "hst"])
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_single_pdg_isolated_training_roots(dtype, pool, mode, clear, implementation):
    case = Case(dtype, pool, clear)
    expected = TwoClock(case, mode).advance(3*case.period-1)
    actual = Encoded(case, mode, implementation).advance(3*case.period-1)
    check(case, actual, expected)
    check_roots(case, actual, expected)


@pytest.mark.parametrize("pool", ["add", "all-softmax"])
@pytest.mark.parametrize("mode", ["hard", "softp", "hst"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_single_pdg_connected_zero_is_not_absence(dtype, pool, mode, implementation):
    case = Case(dtype, pool, True)
    expected = TwoClock(case, mode).advance(3*case.period-1)
    actual = Encoded(case, mode, implementation).advance(3*case.period-1)
    check_roots(case, actual, expected, zero=True)


@pytest.mark.parametrize("pool", ["add", "all-softmax"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_single_pdg_cuts_keep_input_vjps(dtype, pool, implementation):
    case = Case(dtype, pool)
    stop = 3*case.period-1
    whole = Encoded(case, "hst", implementation).advance(stop)
    split = Encoded(case, "hst", implementation)
    first = split.advance(case.period-1)
    assert first.buffer
    tail = split.advance(stop)
    equivalent(whole.body.continuation, tail.body.continuation)
    equivalent(whole.read.continuation, tail.read.continuation)
    equivalent(whole.buffer, tail.buffer)
    for field in ("body", "read"):
        a = getattr(whole, field).continuation.states[0, 0].value
        b = getattr(tail, field).continuation.states[0, 0].value
        equivalent(gradients(a, case.variables), gradients(b, case.variables))
