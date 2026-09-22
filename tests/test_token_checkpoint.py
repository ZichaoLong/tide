from copy import deepcopy
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.token_checkpoint import TokenApplication
from single_graph_training import TwoClock
from single_graph_compare import compare
from single_graph_optimizer import make_optimizer
from token_checkpoint_cases import fixture, state, train


@pytest.mark.parametrize("pool", ["add", "all-softmax"])
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("mode", ["hard", "softp", "hst"])
@pytest.mark.parametrize("kind", ["sgd", "adamw"])
@pytest.mark.parametrize("implementation", ["reference", "serial", "parallel", "packed"])
def test_two_clock_bundle_preserves_cross_graph_optimizer_updates(dtype, pool, clear, mode, kind,
                                                               implementation, tmp_path):
    case, expected = train(dtype, pool, clear, mode, kind, "reference")
    _, actual = train(dtype, pool, clear, mode, kind, implementation, tmp_path)
    for a, e in zip(actual, expected):
        compare(case.body, a[0].body, e[0].body, "application.body")
        compare(case.readout, a[0].read, e[0].read, "application.readout")
        equivalent(a[0].buffer, e[0].buffer, "application.buffer")
        equivalent(a[1:], e[1:], "application_update")


@pytest.mark.parametrize("corruption", ["identity", "policy", "missing_ledger", "cut", "body_clock", "read_clock", "batch", "read_pending",
                                       "buffer_time", "buffer_port", "buffer_batch", "buffer_duplicate", "buffer_nan",
                                       "buffer_shape", "alias_value", "optimizer_order", "optimizer_shape"])
def test_invalid_bundle_never_changes_live_owners(dtype, corruption, tmp_path):
    case, app = fixture(dtype, "add")
    runner = TwoClock(case, "hst"); runner.advance(3*case.period-1)
    optimizer = make_optimizer(case, "sgd")
    for p in case.owner.parameters(): p.grad = torch.ones_like(p)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    path = tmp_path/"bad.pt"; app.save(path, state(runner), optimizer)
    record = torch.load(path, weights_only=True); payload = record["state"]
    row = list(payload["buffer"][0])
    if corruption == "identity": record["identity"]["layers"] += 1
    elif corruption == "policy": record["identity"]["mode"] = "softp"
    elif corruption == "missing_ledger": del payload["readout"]["ledger"]
    elif corruption == "cut": payload["cut"] = True
    elif corruption == "body_clock": payload["body"]["cut"] -= 1
    elif corruption == "read_clock": payload["readout"]["cut"] += 1
    elif corruption == "batch": payload["readout"]["batch_size"] += 1
    elif corruption == "read_pending": payload["readout"]["pending"] = payload["body"]["pending"]
    elif corruption.startswith("buffer_"):
        if corruption == "buffer_time": row[1] = payload["body"]["cut"]
        elif corruption == "buffer_port": row[2] = 999
        elif corruption == "buffer_batch": row[0] = payload["body"]["batch_size"]
        elif corruption == "buffer_nan": row[3] = torch.full_like(row[3], float("nan"))
        elif corruption == "buffer_shape": row[3] = row[3][:-1]
        if corruption == "buffer_duplicate": payload["buffer"].append(payload["buffer"][0])
        else: payload["buffer"][0] = tuple(row)
    elif corruption == "alias_value":
        record["weights"]["readout.input_scale.0"] = record["weights"]["body.output_scale.0"]+1
    elif corruption == "optimizer_order":
        record["optimizer_layout"]["groups"][0].reverse()
    else:
        record["optimizer"]["state"][0]["momentum_buffer"] = torch.ones(19, dtype=dtype)
    torch.save(record, path)
    with torch.no_grad():
        for p in app.models.parameters(): p.add_(.05)
    before = deepcopy((app.models.state_dict(), optimizer.state_dict()))
    with pytest.raises(ValueError): app.load(path, optimizer)
    equivalent(before, (app.models.state_dict(), optimizer.state_dict()))


def test_bundle_retains_occurrence_ledger_and_truncates_partial_buffer(dtype, tmp_path):
    case, app = fixture(dtype, "all-softmax")
    runner = TwoClock(case, "hst"); runner.advance(3*case.period-1)
    value = state(runner); assert value.buffer and value.readout.ledger
    assert any(position != time for position, time in value.readout.ledger.values())
    detached = value.detach(); equivalent(value, detached)
    assert any(x.requires_grad for _, _, _, x in value.buffer)
    path = tmp_path/"state.pt"; app.save(path, value)
    restored = app.load(path); equivalent(value, restored)
    for q in (restored.body, restored.readout):
        assert all(not s.value.requires_grad and all(not t.requires_grad for t in s.slots.values())
                   for s in q.states.values())
        assert all(not a.value.requires_grad for a in q.pending)
    assert all(not x.requires_grad for _, _, _, x in restored.buffer)
    assert all(not x.requires_grad for _, _, _, x in detached.buffer)


@pytest.mark.parametrize("mismatch", ["aliases", "optimizer_missing", "policy"])
def test_bundle_restore_requires_same_application_owners_and_policy(dtype, mismatch, tmp_path):
    case, app = fixture(dtype, "add")
    runner = TwoClock(case, "hst"); runner.advance(case.period-1)
    path = tmp_path/"application.pt"; app.save(path, state(runner))
    if mismatch == "aliases":
        case.read_model.input_scale[0] = torch.nn.Parameter(case.read_model.input_scale[0].detach().clone())
    elif mismatch == "policy":
        app.zeta = .5
    optimizer = make_optimizer(case, "sgd") if mismatch == "optimizer_missing" else None
    before = deepcopy(app.models.state_dict())
    with pytest.raises(ValueError): app.load(path, optimizer)
    equivalent(before, app.models.state_dict())


def test_invalid_application_state_is_not_published(dtype, tmp_path):
    case, app = fixture(dtype, "add")
    runner = TwoClock(case, "hst"); runner.advance(case.period-1)
    value = state(runner); value.buffer = value.buffer+value.buffer
    path = tmp_path/"invalid.pt"
    with pytest.raises(ValueError, match="buffer coordinate"): app.save(path, value)
    assert not path.exists()


@pytest.mark.parametrize("bad", ["layers", "ports", "dtype", "output", "mode", "zeta"])
def test_token_application_boundary_is_explicit(dtype, bad):
    case, _ = fixture(dtype, "add")
    layers, output, mode, zeta = case.layers, 0, "hard", 1.0
    if bad == "layers": layers = True
    elif bad == "ports": layers += 1
    elif bad == "output": output = 999
    elif bad == "mode": mode = "unknown"
    elif bad == "zeta": zeta = float("nan")
    else: case.read_model.to(torch.float32 if dtype == torch.float64 else torch.float64)
    with pytest.raises(ValueError):
        TokenApplication(case.body, case.model, case.readout, case.read_model, layers,
                         output_port=output, mode=mode, zeta=zeta)
