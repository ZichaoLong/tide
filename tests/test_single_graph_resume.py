import pytest
from tidegraph.compare import equivalent
from single_graph_training import IMPLEMENTATIONS, check
from single_graph_optimizer import train


@pytest.mark.parametrize("pool", ["add", "all-softmax"])
@pytest.mark.parametrize("mode", ["hard", "softp", "hst"])
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("optimizer_kind", ["sgd", "adamw"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_single_pdg_resume_matches_two_clock_updates(dtype, pool, mode, clear, optimizer_kind,
                                                    implementation, tmp_path):
    case, expected = train(dtype, pool, clear, mode, "two-clock", optimizer_kind)
    _, actual = train(dtype, pool, clear, mode, implementation, optimizer_kind, checkpoints=tmp_path)
    for a, e in zip(actual, expected):
        check(case, a[0], e[0])
        equivalent(a[1:], e[1:], "resumed_optimizer_step")
