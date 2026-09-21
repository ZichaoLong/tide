import pytest
from tidegraph.compare import equivalent
from single_graph_training import Case, TwoClock, Encoded, IMPLEMENTATIONS, check
from single_graph_roots import gradients
from single_graph_optimizer import train


@pytest.mark.parametrize("pool", ["add", "all-softmax"])
@pytest.mark.parametrize("mode", ["hard", "softp", "hst"])
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("optimizer_kind", ["sgd", "adamw"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_single_pdg_shared_optimizer_updates(dtype, pool, mode, clear, optimizer_kind, implementation):
    case, expected = train(dtype, pool, clear, mode, "two-clock", optimizer_kind)
    _, actual = train(dtype, pool, clear, mode, implementation, optimizer_kind)
    for a, e in zip(actual, expected):
        check(case, a[0], e[0])
        equivalent(a[1:], e[1:], "optimizer_step")
    # First update precedes the first readout; head parameters must be skipped.
    assert expected[0][1]["readout.nodes.0.extra.token_head"] is None
    assert expected[1][1]["readout.nodes.0.extra.token_head"] is not None


@pytest.mark.parametrize("pool", ["add", "all-softmax"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_partial_window_is_part_of_the_truncation_boundary(dtype, pool, implementation):
    case = Case(dtype, pool)
    expected, actual = TwoClock(case, "hst"), Encoded(case, "hst", implementation)
    incomplete = TwoClock(case, "hst")
    for runner in (expected, actual, incomplete):
        assert runner.advance(case.period-1).buffer
    expected.detach(); actual.detach(); incomplete.detach(buffer=False)
    e, a, bad = (runner.advance(case.period) for runner in (expected, actual, incomplete))
    check(case, a, e)
    equivalent(bad.read.outputs, e.read.outputs)
    roots = [next(x for b, _, _, x in case.logits(f.read.outputs) if b == 0) for f in (e, a, bad)]
    eg, ag, badg = (gradients(root, case.variables) for root in roots)
    equivalent(eg, ag)
    old = [f"input.{x.batch}.{x.port}.{x.time}" for x in case.xs if x.time < case.layers]
    assert all(eg[k] is None for k in old)
    assert any(badg[k] is not None and badg[k].abs().sum() > 0 for k in old)
