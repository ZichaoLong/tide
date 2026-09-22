from copy import deepcopy
import pytest
import torch
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent
from coordinate_cases import fixture, corrupt
from token_checkpoint_cases import fixture as token_fixture, state
from single_graph_training import TwoClock


@pytest.mark.parametrize("format", ["single", "token"])
@pytest.mark.parametrize("name", ["ledger_batch", "ledger_port", "ledger_position", "ledger_time",
                                  "state_node", "observations", "history_count", "pending_time"])
@pytest.mark.parametrize("convert", [bool, float], ids=["bool", "float"])
def test_checkpoint_coordinate_rejection_preserves_all_live_owners(dtype, format, name, convert, tmp_path):
    if format == "single":
        g, m, q, _ = fixture(dtype, continued=True)
        owner = m
        def write(path, optimizer): save(path, g, m, q, optimizer)
        def read(path, optimizer): return load(path, g, m, optimizer)
    else:
        case, app = token_fixture(dtype, "add")
        controller = TwoClock(case, "hst"); controller.advance(3*case.period-1)
        q = state(controller); owner = app.models
        def write(path, optimizer): app.save(path, q, optimizer)
        def read(path, optimizer): return app.load(path, optimizer)
    optimizer = torch.optim.SGD(owner.parameters(), lr=.01, momentum=.9)
    for p in owner.parameters(): p.grad = torch.ones_like(p)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    good = tmp_path/"valid.pt"; write(good, optimizer)
    record = torch.load(good, weights_only=True)
    # Use the shared record codec to preserve all other checkpoint fields.
    from tidegraph.checkpoint_values import decode, encode
    payload = record if format == "single" else record["state"]["body"]
    malformed = decode(payload); corrupt(malformed, name, convert); payload.update(encode(malformed))
    bad = tmp_path/"invalid.pt"; torch.save(record, bad)
    with torch.no_grad():
        for p in owner.parameters(): p.add_(.2)
    optimizer.param_groups[0]["lr"] = .7
    before = deepcopy((owner.state_dict(), optimizer.state_dict()))
    with pytest.raises(ValueError, match="int64"): read(bad, optimizer)
    equivalent(before, (owner.state_dict(), optimizer.state_dict()))
    equivalent(q, read(good, optimizer))
    malformed = deepcopy(q.detach())
    corrupt(malformed if format == "single" else malformed.body, name, convert)
    q = malformed
    unpublished = tmp_path/"unpublished.pt"
    with pytest.raises(ValueError, match="int64"): write(unpublished, optimizer)
    assert not unpublished.exists()
