"""Actual arbitrary planned consumer windows, AdamW and rejected restore identity."""
from copy import copy
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"examples/consumer_core5"))
from planned_windows import child, input_window
from tidegraph.planning.recipes import prepare
from tidegraph.comparison import equivalent


def plan():
    return prepare(dict(topology="lh", family="kv", width=4, depth=1, region_width=2,
        sparse_denominator=2, profile="textbook"), samples=2, positions=3, seed=7,
        input_pattern="staggered-v1")[0]


@pytest.mark.parametrize("client", ["python", "native"])
def test_planned_three_adamw_windows_exact_checkpoint(tmp_path, client):
    a = SimpleNamespace(device="cpu", dtype="float64", client=client,
        native_library=os.environ.get("TIDE_BUILD_DIR") if client == "native" else None,
        placement="mixed-c", threads=1, steps=3, learning_rate=1e-5,
        max_materialize_bytes=2**26, max_workspace_bytes=2**26,
        deterministic_algorithms=True, resume=None, save_checkpoint=False)
    paths = {name: tmp_path/name for name in ("whole", "prefix", "resumed", "bad")}
    for path in paths.values(): path.mkdir()
    whole = child(a, plan(), paths["whole"])
    prefix_args = copy(a); prefix_args.steps=2; prefix_args.save_checkpoint=True
    child(prefix_args, plan(), paths["prefix"])
    resumed_args = copy(a); resumed_args.steps=1; resumed_args.resume=paths["prefix"]/"checkpoint.pt"
    resumed = child(resumed_args, plan(), paths["resumed"])
    equivalent(torch.load(paths["whole"]/"observation.pt", weights_only=True),
               torch.load(paths["resumed"]/"observation.pt", weights_only=True), atol=0, rtol=0)
    assert whole["steps"][-1] == resumed["steps"][-1]
    assert whole["steps"][-1]["execution"]["schedule"] == "settle-columns-v1"
    resumed_args.learning_rate *= 2
    with pytest.raises(ValueError, match="identity differs"):
        child(resumed_args, plan(), paths["bad"])


def test_planned_input_uses_sparse_physical_ports_and_raw_position():
    p = plan()
    x, layout, ports = input_window(p, torch.zeros(4, dtype=torch.float64), 5)
    assert x.is_leaf and x.requires_grad
    assert torch.equal(layout.times, torch.arange(15, 18).repeat(2))
    assert layout.times.dtype == torch.int64
    assert any(len(m.rows) < 6 for m in ports.values())
    first = next(m for m in ports.values() if len(m.rows))
    g, = torch.autograd.grad(first.content.sum(), x)
    expected = torch.zeros_like(x).reshape(-1, 4)
    expected[first.rows] = 1
    assert torch.equal(g, expected.reshape_as(x))
