"""Single-batch padding must preserve sparse events, masks, state and public VJPs."""
import pytest
import torch
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.reference import run
from tidegraph.records import Result
from fiber_cases import fixture, physical
from fiber_pool_cases import profile
from isolated_cases import vjp
from test_fiber_packing import case


def ragged(dtype):
    g, m, q, xs, leaves, lengths = case(dtype, 0, True)
    for b, cache in enumerate((0, 2, 5)):
        state = q.states[b, 0]
        state.slots = {
            "key": torch.full((cache, 2, 2), .1+b/20, dtype=dtype, requires_grad=True),
            "value": torch.full((cache, 2, 2), .3+b/10, dtype=dtype, requires_grad=True),
            "log_bias": torch.full((cache,), -.13-b/20, dtype=dtype, requires_grad=True)}
        leaves.update({f"initial.{b}.{name}": value for name, value in state.slots.items()})
    return g, m, q, xs, leaves, lengths


@pytest.mark.parametrize("algorithm", ["streaming", "frontier"])
@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("packing", ["exact", "single"])
@pytest.mark.parametrize("grad", [False, True])
def test_ragged_query_cache_masks_and_public_gradients(dtype, algorithm, workers, packing, grad):
    g, m, q, xs, leaves, lengths = ragged(dtype)
    with torch.set_grad_enabled(grad):
        expected = run(g, m, q, xs, 12, sealed_until=12)
        actual = Native(g, m, algorithm=algorithm, workers=workers, packed=True,
                        attention_packing=packing).run(q, xs, 12, sealed_until=12)
    equivalent(expected, actual)
    assert (3, 0) not in actual.continuation.states
    for state in actual.continuation.states.values():
        for value in (state.value, *state.slots.values()):
            assert value.untyped_storage().nbytes() == value.numel()*value.element_size()
    if algorithm == "frontier":
        assert actual.stats["state_sequence_calls"] == (1 if packing == "single" else 3)
        assert actual.stats["max_state_batch"] == (3 if packing == "single" else 1)
        # Different event boundaries: future keys remain masked inside this work.
        scores = 16*11 if packing == "single" else 6*6+4*6+6*11
        assert actual.stats["attention_score_elements"] == 2*scores
    if grad:
        equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
        eg, ag = vjp(expected.outputs[0][3], leaves), vjp(actual.outputs[0][3], leaves)
        equivalent(eg, ag)
        assert ag["input.0.0.1"] is None and ag["input.1.0.0"] is None
        assert ag["input_scale.2"] is None and ag["nodes.0.extra.fiber_decay"] is None


@pytest.mark.parametrize("kind", ["sum", "mean", "linear", "active-softmax", "all-softmax"])
@pytest.mark.parametrize("policy", ["all", "selected", "clear", "old"])
def test_single_cyclic_selection_clear_pooling_and_all_roots(dtype, kind, policy):
    g, m, q, xs, leaves = fixture(dtype, policy, cyclic=True, profile=profile(kind))
    expected = run(g, m, q, xs, 9, sealed_until=9, mode="hst")
    actual = Native(g, m, workers=3, packed=True, attention_packing="single", mode="hst",
                    parallel_regions=True, compact_events=True).run(q, xs, 9, sealed_until=9)
    equivalent(expected, actual); equivalent(physical(expected, m), physical(actual, m))
    for root in ("output", "pending", "state"):
        equivalent(vjp(objective(expected, root), leaves), vjp(objective(actual, root), leaves))


@pytest.mark.parametrize("first", ["exact", "single"])
def test_policy_switch_across_cursor_checkpoint_and_idle_cut(dtype, first, tmp_path):
    g, m, q, xs, leaves = fixture(dtype, "clear", cyclic=True, profile=profile("all-softmax"))
    with torch.no_grad():
        expected = run(g, m, q, xs, 12, sealed_until=12)
        cursor = Native(g, m, workers=3, packed=True, attention_packing=first).cursor(q)
        prefix = cursor.advance([x for x in xs if x.time < 4], 4, sealed_until=4)
        snapshot = cursor.snapshot()
        path = tmp_path/"state.pt"; save(path, g, m, snapshot)
        restored = load(path, g, m)
        equivalent(snapshot, restored)
        second = "single" if first == "exact" else "exact"
        cursor = Native(g, m, workers=1, packed=True, attention_packing=second).cursor(restored)
        suffix = cursor.advance([x for x in xs if x.time >= 4], 12, sealed_until=12)
        actual = Result(cursor.snapshot(), prefix.trace+suffix.trace, prefix.outputs+suffix.outputs,
                        prefix.messages+suffix.messages, suffix.stats)
        equivalent(expected, actual)
        before = cursor.snapshot()
        # A zero-length cut preserves all pending feedback and retained state.
        empty = Native(g, m, packed=True, attention_packing=second).cursor(before)
        zero = empty.advance([], 12, sealed_until=12)
        equivalent(before, empty.snapshot()); assert not zero.outputs


def training(dtype, packing):
    g, m, q, xs, _ = fixture(dtype, profile=profile("all-softmax"))
    m.nodes[1] = m.nodes[0]
    opt = torch.optim.AdamW(m.parameters(), lr=.002, eps=1e-5)
    records = []
    for step, stop in enumerate((4, 12)):
        opt.zero_grad(set_to_none=True)
        inputs = [x for x in xs if q.cut <= x.time < stop]
        if packing is None:
            r = run(g, m, q, inputs, stop, sealed_until=stop)
        else:
            r = Native(g, m, algorithm="frontier", workers=3, packed=True,
                       attention_packing=packing if step == 0 else "exact").run(q, inputs, stop, sealed_until=stop)
        objective(r).backward(); opt.step(); q = r.continuation.detach()
        records.append((q, {k: None if p.grad is None else p.grad.clone() for k, p in m.named_parameters()},
                        {k: v.clone() for k, v in m.state_dict().items()}))
    return records


def test_shared_training_and_execution_policy_switch(dtype):
    equivalent(training(dtype, None), training(dtype, "single"))


def test_invalid_attention_policy_rejected_before_execution(dtype):
    g, m, *_ = fixture(dtype)
    with pytest.raises(ValueError, match="packing"):
        Native(g, m, attention_packing="unknown")
