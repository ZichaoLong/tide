"""Complete event attention/KV training through the public Python client."""
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from resident_test_target import target_device, device_api, owner_devices
from resident_training_cases import inputs, tree_equal
from resident_event_cases import runtime, roots, terms
from resident_cache_training import training_case


@pytest.fixture
def target():
    return target_device()


@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("schedule", ["streaming", "greedy"])
@pytest.mark.parametrize("kind", ["sgd", "adamw"])
def test_event_training(target, family, schedule, kind, tmp_path):
    training_case(target, family, schedule, kind, tmp_path, runtime=runtime, roots=roots, terms=terms)


@pytest.mark.parametrize("mode", ["hst", "softp"])
@pytest.mark.parametrize("clear", [False, True])
def test_event_control_and_clear(target, mode, clear, tmp_path):
    training_case(target, "pdg", "greedy", "adamw", tmp_path, runtime=runtime, roots=roots, terms=terms, mode=mode, clear=clear)



def test_event_new_process(target, tmp_path):
    r = runtime("pdg", target)
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
        "--device", target, "--checkpoint", str(path), "--output", str(output), "--memory", "event"],
        check=True, timeout=180)
    tree_equal(expected, torch.load(output, weights_only=True))
