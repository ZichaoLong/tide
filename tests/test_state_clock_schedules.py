from dataclasses import replace
import pytest
import torch
from tidegraph import External, StateClock
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.lazy_add import decode
from tidegraph.fiber_attention import decode_bias
from tidegraph.native import Native
from tidegraph.records import Result
from clock_cases import PROFILES, fixture, project, loss_grad
from isolated_cases import vjp
from read_cases import execute


@pytest.mark.parametrize("memory", PROFILES)
@pytest.mark.parametrize("clock", [StateClock(4, 0, 3), StateClock(4, 3, 1)])
@pytest.mark.parametrize("implementation", ["reference", "python-frontier", "native-serial", "native-packed", "native-frontier"])
def test_local_clock_projection_and_public_root_vjps(dtype, memory, clock, implementation):
    base, m, q, g, em, eq, xs, exs, leaves = fixture(dtype, clock, memory)
    expected = execute("reference", base, m, q, xs, stop=7)
    raw = execute(implementation, g, em, eq, exs, stop=clock.to_global(7))
    actual = project(raw, base, clock); equivalent(expected, actual)
    equivalent(loss_grad(expected, leaves), loss_grad(actual, leaves))
    roots = [(expected.outputs[0][-1], actual.outputs[0][-1]),
             (expected.continuation.states[0, 0].value, actual.continuation.states[0, 0].value)]
    roots += [(value, actual.continuation.states[0, 0].slots[name])
              for name, value in expected.continuation.states[0, 0].slots.items()]
    for a, b in roots:
        for zero in (False, True):
            equivalent(vjp(a, leaves, zero), vjp(b, leaves, zero))


@pytest.mark.parametrize("memory", ["lh-add-repeat-v1", "lh-fiber-attention-all-softmax-repeat-v1"])
@pytest.mark.parametrize("clock", [StateClock(4, 0, 3), StateClock(4, 3, 1)])
@pytest.mark.parametrize("policy", ["all", "selected", "clear"])
def test_every_global_cut_and_physical_idle_decay(dtype, memory, clock, policy):
    base, m, q, g, em, eq, xs, exs, leaves = fixture(dtype, clock, memory, policy)
    cursor = Native(g, em, packed=True, workers=3).cursor(eq)
    traces, outputs, messages = [], [], []
    decode_state = decode if memory == "lh-add-repeat-v1" else decode_bias
    for stop in range(clock.to_global(7)+1):
        part = cursor.advance([x for x in exs if cursor.cut <= x.time < stop], stop, sealed_until=stop)
        traces += part.trace; outputs += part.outputs; messages += part.messages
        raw = Result(cursor.snapshot(), traces, outputs, messages, {})
        actual = project(raw, base, clock)
        expected = execute("reference", base, m, q, [x for x in xs if x.time < clock.cut(stop)], stop=clock.cut(stop), mode="hard")
        equivalent(expected, actual)
        for owner, state in raw.continuation.states.items():
            equivalent(decode_state(em.nodes[owner[1]], state, stop),
                       decode_state(m.nodes[owner[1]], expected.continuation.states[owner], clock.cut(stop)))
    equivalent(loss_grad(expected, leaves), loss_grad(actual, leaves))


@pytest.mark.parametrize("memory", ["ssm", "lh-add-repeat-v1", "lh-fiber-attention-all-softmax-repeat-v1"])
def test_clock_optimizer_checkpoint_and_resume(dtype, memory, tmp_path):
    clock = StateClock(4, 0, 3)
    def train(embedded):
        base, m, q, g, em, eq, xs, exs, _ = fixture(dtype, clock, memory)
        graph, model, state, inputs = (g, em, eq, exs) if embedded else (base, m, q, xs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002); records = []
        for stop in (2, 7):
            cut = clock.to_global(stop) if embedded else stop
            optimizer.zero_grad(set_to_none=True)
            result = execute("native-frontier" if embedded else "reference", graph, model, state,
                             [x for x in inputs if state.cut <= x.time < cut], stop=cut)
            objective(result).backward(); optimizer.step(); state = result.continuation.detach()
            records.append(({k: p.detach().clone() for k, p in model.state_dict().items()},
                            {k: None if p.grad is None else p.grad.clone() for k, p in model.named_parameters()}))
        return records, graph, model, state, optimizer
    expected, *_ = train(False); actual, g, m, q, optimizer = train(True)
    equivalent(expected, actual)
    path = tmp_path/"clock.pt"; save(path, g, m, q, optimizer)
    _, _, _, _, restored, _, _, _, _ = fixture(dtype, clock, memory)
    ro = torch.optim.AdamW(restored.parameters(), lr=.002)
    resumed = load(path, g, restored, ro)
    equivalent(q, resumed); equivalent(m.state_dict(), restored.state_dict())
    equivalent(optimizer.state_dict(), ro.state_dict())
    assert restored.nodes[0] is restored.nodes[1]
    inputs = [External(b, p, q.ledger[b, p][0]+1, clock.to_global(9),
                       torch.full((4,), .12+b/20+p/10, dtype=dtype)) for b in range(3) for p in (0, 2)]
    expected = execute("reference", g, m, q, inputs, stop=clock.to_global(10))
    actual = execute("native-frontier", g, restored, resumed, inputs, stop=clock.to_global(10))
    equivalent(expected, actual)
    equivalent(loss_grad(expected, dict(m.named_parameters())), loss_grad(actual, dict(restored.named_parameters())))
    changed = replace(g, nodes=tuple(replace(n, state_clock=StateClock(4, 1, 2)) for n in g.nodes))
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    with pytest.raises(ValueError, match="graph"):
        load(path, changed, restored)
    equivalent(before, restored.state_dict())


@pytest.mark.parametrize("memory", PROFILES)
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_clock_wrapper_retains_prefill_without_inference_replay(dtype, memory, context):
    with context():
        clock = StateClock(5, 2, 2)
        base, m, q, g, em, eq, xs, exs, _ = fixture(dtype, clock, memory)
        expected = execute("reference", base, m, q, xs, stop=7)
        raw = execute("native-frontier", g, em, eq, exs, stop=clock.to_global(7))
        equivalent(expected, project(raw, base, clock))
    assert raw.stats["state_blocks"] > 0
    assert raw.stats["max_state_sequence"] >= 3
    assert not any(v for k, v in raw.stats.items() if k.startswith("semantic_") and "replay" in k)
