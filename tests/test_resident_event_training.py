"""Complete event attention/KV training through the public Python client."""
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from tidegraph.compare import equivalent
from resident_training_cases import inputs, compare_parameters, compare_gradients, tree_equal
from resident_event_cases import runtime, roots, terms


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
def test_event_training(target, family, schedule, kind, tmp_path):
    training_case(target, family, schedule, kind, tmp_path)


@pytest.mark.parametrize("mode", ["hst", "softp"])
@pytest.mark.parametrize("clear", [False, True])
def test_event_control_and_clear(target, mode, clear, tmp_path):
    training_case(target, "pdg", "greedy", "adamw", tmp_path, mode=mode, clear=clear)


def training_case(target, family, schedule, kind, tmp_path, mode="hard", clear=False):
    r, cpu = runtime(family, target, schedule, mode, clear), runtime(family, "cpu", mode=mode, clear=clear)
    names, parameters = zip(*((n, p) for n, p in cpu.execution_model.named_parameters() if p.requires_grad))
    settings = dict(lr=.001, weight_decay=.01)
    settings.update(momentum=.5) if kind == "sgd" else settings.update(eps=1e-5, amsgrad=True)
    optimizer = getattr(torch.optim, "SGD" if kind == "sgd" else "AdamW")(parameters, **settings)
    values = torch.sin(torch.arange(48, dtype=torch.float32).reshape(2, 6, 4) * .37) * .1
    values[0, 1].zero_()
    oracle = cpu.session(2)
    with torch.no_grad():
        session = r.training_session(2, optimizer=kind, groups=[dict(parameters=list(names), **settings)])
        for step, root_mode in enumerate(("all", "zero", "none")):
            x = values.clone().requires_grad_(True)
            cotangents, objectives = [], []
            for start, stop in ((step*2, step*2+1), (step*2+1, step*2+2)):
                with torch.enable_grad():
                    args, kw = inputs(oracle, x, start, stop)
                    reference = oracle.advance(args, **kw)
                    objectives.extend(terms(reference, oracle.continuation, root_mode))
                args, kw = inputs(session, values.to(target) if step == 1 else values, start, stop)
                window = session.advance_device(args, **kw)
                cotangents.append(roots(session, window, root_mode))
                equivalent(reference, session.result())
                with pytest.raises(ValueError, match="cache"):
                    session.cotangents(window, cache=[])
            with torch.enable_grad():
                expected = (torch.autograd.grad(torch.stack(objectives).sum(), (*parameters, x), allow_unused=True)
                            if objectives else (None,) * (len(parameters)+1))
            gradient = session.backward(cotangents)
            compare_gradients(gradient, cpu.execution_model, {id(p): g for p, g in zip(parameters, expected)}, x, expected[-1])
            if step == 0:
                # No initial state was supplied, even though later caches exist.
                assert all(not g.key_connected.any() and not g.value_connected.any() for g in gradient.initial_cache)
            optimizer.zero_grad(set_to_none=True)
            for p, g in zip(parameters, expected):
                p.grad = g
            optimizer.step()
            assert session.step().applied
            compare_parameters(session.checkpoint(), cpu.execution_model)
            if step == 0:
                path = tmp_path / "event.pt"
                session.save(path)
                checkpoint = session.checkpoint()
                session.close()
                session = r.training_session(2, checkpoint=path)
                tree_equal(checkpoint, session.checkpoint())
            oracle.detach()
        session.close()


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
