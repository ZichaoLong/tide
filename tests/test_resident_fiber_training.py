"""Public fiber training, numerical/cache continuation and independent restart."""
from functools import partial
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from resident_training_cases import inputs, tree_equal
from resident_fiber_cases import runtime, roots, terms
from resident_cache_training import training_case


@pytest.fixture
def target():
    device = os.environ.get("TIDE_RESIDENT_DEVICE")
    if device is None:
        pytest.skip("optional resident NPU target not requested")
    import torch_npu
    assert device.startswith("npu") and os.environ.get("TIDE_RESIDENT_LIBRARY")
    return device


@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("schedule", ["streaming", "greedy"])
@pytest.mark.parametrize("kind", ["sgd", "adamw"])
def test_fiber_training(target, family, schedule, kind, tmp_path):
    training_case(target, family, schedule, kind, tmp_path, runtime=runtime, roots=roots, terms=terms)


@pytest.mark.parametrize("pooling", ["mean", "linear", "active-softmax", "all-softmax"])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
def test_fiber_pool_control_clear(target, pooling, mode, tmp_path):
    training_case(target, "pdg", "greedy", "adamw", tmp_path, mode=mode, clear=True,
                  runtime=partial(runtime, pooling=pooling), roots=roots, terms=terms)


def test_fiber_new_process(target, tmp_path):
    r = runtime("pdg", target, pooling="all-softmax")
    values = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) * .005
    path, output = tmp_path / "prefix.pt", tmp_path / "suffix.pt"
    with torch.no_grad(), r.training_session(1, optimizer="adamw") as session:
        for start, stop in ((0, 1), (1, 2)):
            args, kw = inputs(session, values, start, stop)
            w = session.advance_device(args, **kw)
            session.backward([roots(session, w, "all")])
            assert session.step().applied
        session.save(path)
        args, kw = inputs(session, values, 2, 4)
        w = session.advance_device(args, **kw)
        session.backward([roots(session, w, "all")])
        assert session.step().applied
        expected = session.checkpoint()
    subprocess.run([sys.executable, str(Path(__file__).with_name("resident_training_worker.py")),
        "--device", target, "--checkpoint", str(path), "--output", str(output),
        "--memory", "fiber", "--pooling", "all-softmax"], check=True, timeout=180)
    tree_equal(expected, torch.load(output, weights_only=True))
