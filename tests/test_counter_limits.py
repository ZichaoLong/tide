from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, History, Node, Region
from tidegraph.ops import Model
from read_cases import execute


@pytest.mark.parametrize("kind", ["ema", "ssm", "linear", "delta", "attention"])
@pytest.mark.parametrize("implementation", ["reference", "python-frontier", "native-serial", "native-packed", "native-frontier"])
def test_builtin_observation_counter_limits(dtype, kind, implementation):
    g = Graph((Node(0, memory=kind),), (), (Region(1),), (0,), ())
    m = Model(g, width=2, dtype=dtype)
    state = replace(m.nodes[0].initial(), observations=2**63-2)
    q = Continuation(g.identity, 1, states={(0, 0): state})
    xs = [External(0, 0, i, i, torch.ones(2, dtype=dtype)) for i in range(2)]
    result = execute(implementation, g, m, q, xs[:1], stop=1)
    assert result.continuation.states[0, 0].observations == 2**63-1
    with pytest.raises(ValueError, match="counter overflow"):
        execute(implementation, g, m, q, xs, stop=2)
    assert q.states[0, 0].observations == 2**63-2


@pytest.mark.parametrize("implementation", ["reference", "python-frontier", "native-serial", "native-packed", "native-frontier"])
def test_selected_counter_limits(dtype, implementation):
    g = Graph((Node(0),), (), (Region(1),), (0,), ())
    m = Model(g, width=1, dtype=dtype)
    q = Continuation(g.identity, 1, history={(0, 0): History(node_maps={"selected": {0: 2**63-2}})})
    xs = [External(0, 0, i, i, torch.ones(1, dtype=dtype)) for i in range(2)]
    result = execute(implementation, g, m, q, xs[:1], stop=1)
    assert result.continuation.history[0, 0].node_maps["selected"][0] == 2**63-1
    with pytest.raises(ValueError, match="counter overflow"):
        execute(implementation, g, m, q, xs, stop=2)
    assert q.history[0, 0].node_maps["selected"][0] == 2**63-2


@pytest.mark.parametrize("implementation", ["reference", "python-frontier", "native-packed", "native-frontier"])
def test_nonincrementing_counters_can_remain_at_the_int64_limit(dtype, implementation):
    g = Graph((Node(0, identity=True),), (), (Region(1, selector="positive-v1", read_mode="content"),), (0,), ())
    m = Model(g, width=1, dtype=dtype)
    state = replace(m.nodes[0].initial(), observations=2**63-1)
    q = Continuation(g.identity, 1, states={(0, 0): state},
                     history={(0, 0): History(node_maps={"selected": {0: 2**63-1}})})
    result = execute(implementation, g, m, q, [External(0, 0, 0, 0, torch.ones(1, dtype=dtype))], stop=1)
    assert not result.trace[0]["active"]
    assert result.continuation.states[0, 0].observations == 2**63-1
    assert result.continuation.history[0, 0].node_maps["selected"][0] == 2**63-1
