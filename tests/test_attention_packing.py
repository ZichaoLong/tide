from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region
from tidegraph.compare import equivalent
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.packing import PackedSequence
from tidegraph.reference import run


@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("implementation", ["python", "native-serial", "native-packed"])
def test_attention_packing_has_real_batch_and_sequence(dtype, ragged, implementation):
    g = Graph((Node(0, memory="attention", query_heads=2, kv_heads=1, window=2),), (), (Region(1),), (0,), (0,))
    m = Model(g, width=4, dtype=dtype); q = Continuation(g.identity, 4)
    lengths = (6, 4 if ragged else 6, 6)
    xs = [External(b, 0, t, t, torch.full((4,), (b+1) * (t+1) / 10, dtype=dtype))
          for b, length in enumerate(lengths) for t in range(length)]
    with torch.inference_mode():
        expected = run(g, m, q, xs, 8, sealed_until=8)
        actual = frontier(g, m, q, xs, 8, sealed_until=8) if implementation == "python" else Native(
            g, m, algorithm="frontier", packed=implementation == "native-packed", workers=3).run(q, xs, 8, sealed_until=8)
    equivalent(expected, actual)
    assert (3, 0) not in actual.continuation.states  # no candidate from padding or idle sample
    calls = 3 if implementation == "native-serial" else 2 if ragged else 1
    assert actual.stats["state_sequence_calls"] == calls
    assert actual.stats["state_blocks"] == 3 and actual.stats["state_steps"] == 0
    if implementation.startswith("native"):
        assert actual.stats["max_state_batch"] == (1 if implementation == "native-serial" else 2 if ragged else 3)
        assert actual.stats["max_state_sequence"] == 6
        assert actual.stats["attention_score_elements"] == sum(2 * length**2 for length in lengths)
    for state in actual.continuation.states.values():
        for t in (state.value, *state.slots.values()):
            assert t.untyped_storage().nbytes() == t.numel() * t.element_size()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("invalid", ["length", "heads", "slot", "clock"])
def test_invalid_cache_is_rejected(dtype, native, invalid):
    g = Graph((Node(0, memory="attention", query_heads=2, window=2),), (), (Region(1),), (0,), (0,))
    m = Model(g, width=4, dtype=dtype); state = m.nodes[0].initial()
    state.observations = 3
    shape = (3, 1, 2) if invalid == "length" else (1, 2, 2) if invalid == "heads" else (1, 1, 2)
    state.slots = {name: torch.ones(shape, dtype=dtype) for name in ("key", "value")}
    if invalid == "slot":
        del state.slots["value"]
    if invalid == "clock":
        state.observations = 0
    q = Continuation(g.identity, 1, states={(0, 0): state})
    with pytest.raises(ValueError, match="attention"):
        if native:
            Native(g, m).run(q, [], 1, sealed_until=1)
        else:
            run(g, m, q, [], 1, sealed_until=1)


def test_attention_config_and_graph_identity(dtype):
    for node in (Node(0, query_heads=3, kv_heads=2), Node(0, window=-1), Node(0, kv_heads=0)):
        with pytest.raises(ValueError, match="attention"):
            Graph((node,), (), (Region(1),), (0,), (0,))
    g = Graph((Node(0, memory="attention", query_heads=2),), (), (Region(1),), (0,), (0,))
    with pytest.raises(ValueError, match="divisible"):
        Model(g, width=3, dtype=dtype)
    different = replace(g, nodes=(replace(g.nodes[0], window=2),))
    assert different.identity != g.identity
    assert Native(g, Model(g, width=4, dtype=dtype)).compiled.identity != Native(
        different, Model(different, width=4, dtype=dtype)).compiled.identity


@pytest.mark.parametrize("change", ["offsets", "owners", "times", "views"])
def test_packed_metadata_validation(change):
    from tidegraph.content import Content
    values = torch.ones(3, 2)
    p = PackedSequence(values, [0, 1, 3], [(0, 0), (1, 0)], [0, 1, 2], [Content(v) for v in values])
    p.validate()
    setattr(p, change, {"offsets": [], "owners": [(0, 0), (0, 0)], "times": [0, 2, 1], "views": []}[change])
    with pytest.raises(ValueError, match="packed"):
        p.validate()
