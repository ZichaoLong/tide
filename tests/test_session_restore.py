"""Session constraints must validate the one deserialized value before mutation."""
import copy

import pytest
import torch

from tidegraph import GraphConfig, GraphRuntime
from tidegraph.compare import equivalent
from tidegraph import checkpoint


def owners():
    config = GraphConfig.from_dict(dict(schema_version=1, family="settle",
        topology=dict(kind="layered", layers=[1, 1], budget=1),
        model=dict(width=2, dtype="float64")))
    runtime = GraphRuntime(config, device="cpu")
    optimizer = torch.optim.AdamW(runtime.model.parameters(), lr=.01)
    for parameter in runtime.model.parameters():
        parameter.grad = torch.full_like(parameter, .1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return runtime, runtime.session(2), optimizer


def test_session_load_reads_and_decodes_once_and_restores_next_window(tmp_path, monkeypatch):
    runtime, session, optimizer = owners()
    inputs = torch.ones(2, 1, 2, dtype=torch.float64)
    session.advance(inputs)
    session.detach()
    path = tmp_path/"checkpoint.pt"
    session.save(path, optimizer)
    actual_runtime, actual_session, actual_optimizer = owners()
    with torch.no_grad():
        next(actual_runtime.model.parameters()).add_(2)
    counts = dict(read=0, decode=0)
    original_load, original_decode = torch.load, checkpoint.decode

    def read(*args, **kwargs):
        counts["read"] += 1
        assert counts["read"] == 1, "the committed state must be the one validated read"
        return original_load(*args, **kwargs)

    def decode(*args, **kwargs):
        counts["decode"] += 1
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(torch, "load", read)
    monkeypatch.setattr(checkpoint, "decode", decode)
    actual_session.load(path, actual_optimizer)
    assert counts == dict(read=1, decode=1)
    equivalent(session.continuation, actual_session.continuation, atol=0, rtol=0)
    equivalent(optimizer.state_dict(), actual_optimizer.state_dict(), atol=0, rtol=0)
    equivalent(session.advance(inputs), actual_session.advance(inputs), atol=0, rtol=0)


@pytest.mark.parametrize("corruption,match", [("batch", "batch size"), ("cut", "complete position"),
    ("decode", "malformed checkpoint continuation"), ("ownership", "malformed checkpoint ownership"),
    ("weight", "parameter mismatch"), ("schema", "schema")])
def test_rejected_session_restore_preserves_all_live_owners(tmp_path, corruption, match):
    runtime, session, optimizer = owners()
    path = tmp_path/"checkpoint.pt"
    session.save(path, optimizer)
    record = torch.load(path, weights_only=True)
    if corruption == "batch":
        record["batch_size"] = 1
    elif corruption == "cut":
        record["cut"] = 1  # Legal empty PDG cut; not a complete Settle position.
    elif corruption == "decode":
        del record["states"]
    elif corruption == "ownership":
        del record["aliases"]
    elif corruption == "weight":
        record["weights"][next(iter(record["weights"]))] = torch.tensor(float("nan"))
    else:
        record = []
    torch.save(record, path)
    with torch.no_grad():
        next(runtime.model.parameters()).add_(2)
    optimizer.param_groups[0]["lr"] = .2
    before = copy.deepcopy((runtime.model.state_dict(), optimizer.state_dict(), session.continuation))
    original_continuation = session.continuation
    with pytest.raises(ValueError, match=match):
        session.load(path, optimizer)
    assert session.continuation is original_continuation
    equivalent(before, (runtime.model.state_dict(), optimizer.state_dict(), session.continuation), atol=0, rtol=0)
