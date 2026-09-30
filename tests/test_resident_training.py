"""Public Python-owned resident VJP, real loss cotangents, updates and disk resume."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from tidegraph import ResidentTrainingLimits
from tidegraph.compare import equivalent
from resident_training_cases import (runtime, inputs, roots, terms, compare_parameters,
                                     compare_gradients, tree_equal)


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
def test_resident_python_training(target, family, schedule, kind, tmp_path):
    training_case(target, family, schedule, kind, tmp_path)


def training_case(target, family, schedule, kind, tmp_path, full="tanh", aggregation="sum"):
    r = runtime(family, target, schedule, full, aggregation)
    cpu = runtime(family, "cpu", full=full, aggregation=aggregation)
    names, parameters = zip(*((n, p) for n, p in cpu.execution_model.named_parameters() if p.requires_grad))
    options = dict(lr=.001, weight_decay=.01)
    options.update(momentum=.5) if kind == "sgd" else options.update(eps=.0001, amsgrad=True)
    optimizer = getattr(torch.optim, "SGD" if kind == "sgd" else "AdamW")(parameters, **options)
    values = torch.sin(torch.arange(48, dtype=torch.float32).reshape(2, 6, 4) * .37) * .1
    values[0, 1].zero_()
    oracle = cpu.session(2)
    with torch.no_grad():
        session = r.training_session(2, optimizer=kind, groups=[dict(parameters=list(names), **options)])
        original = session.checkpoint()
        for step, mode in enumerate(("all", "zero", "none")):
            x = values.clone().requires_grad_(True)
            cotangents, objectives = [], []
            for start, stop in ((step*2, step*2+1), (step*2+1, step*2+2)):
                with torch.enable_grad():
                    args, kw = inputs(oracle, x, start, stop)
                    reference = oracle.advance(args, **kw)
                    objectives.extend(terms(reference, oracle.continuation, mode))
                source = values.to(target) if step == 1 else values
                args, kw = inputs(session, source, start, stop)
                stream = torch.npu.Stream(device=target)
                with torch.npu.stream(stream):
                    window = session.advance_device(args, **kw)
                    cotangents.append(roots(session, window, mode))
                equivalent(reference, session.result())
            with pytest.raises(RuntimeError, match="outstanding|unconsumed|detach"):
                session.checkpoint()
            with torch.enable_grad():
                expected = (torch.autograd.grad(torch.stack(objectives).sum(), (*parameters, x), allow_unused=True)
                            if objectives else (None,) * (len(parameters)+1))
            grad = session.backward(cotangents)
            compare_gradients(grad, cpu.execution_model, {id(p): g for p, g in zip(parameters, expected)}, x, expected[-1])
            optimizer.zero_grad(set_to_none=True)
            for p, g in zip(parameters, expected):
                p.grad = g
            optimizer.step()
            result = session.step()
            assert result.applied and result.generation == step + 1
            checkpoint = session.checkpoint()
            compare_parameters(checkpoint, cpu.execution_model)
            if step == 0:
                assert not torch.equal(original["state"]["values"], checkpoint["state"]["values"])
                assert (checkpoint["state"]["steps"] > 0).any()
                path = tmp_path / "train.pt"
                session.save(path)
                with pytest.raises(FileExistsError):
                    session.save(path)
                session.close()
                session = r.training_session(2, checkpoint=path)
                tree_equal(checkpoint, session.checkpoint())
                checkpoint["state"]["values"].fill_(123)
            oracle.detach()
        session.close()


def test_resident_training_rejection_and_export(target, tmp_path):
    r = runtime("pdg", target)
    with pytest.raises(ValueError, match="no_grad"):
        r.training_session(1)
    with torch.no_grad():
        s = r.training_session(1, limits=ResidentTrainingLimits(windows=1))
        initial = s.checkpoint()
        for name, bad in (("generation", True), ("next_token", 1.0), ("offsets", [False])):
            c = dict(initial, **{name: bad})
            with pytest.raises(ValueError, match="int64"):
                r.training_session(1, checkpoint=c)
        bad = copy.deepcopy(initial)
        bad["state"]["steps"].fill_(-1)
        with pytest.raises(ValueError, match="counter"):
            r.training_session(1, checkpoint=bad)
        tree_equal(initial, s.checkpoint())
        with pytest.raises(ValueError, match="exclusive"):
            r.training_session(1, checkpoint=initial, optimizer="sgd")
        values = torch.ones(1, 2, 4) * .01
        args, kw = inputs(s, values, 0, 1)
        w = s.advance_device(args, **kw)
        with pytest.raises(ValueError, match="capacity"):
            s.advance_device([], stop=1, sealed_until=1)
        assert s.cut == 1
        s.detach()
        args, kw = inputs(s, values, 1, 2)
        current = s.advance_device(args, **kw)
        with pytest.raises(ValueError, match="token"):
            s.backward([s.cotangents(w)])
        s.backward([roots(s, current, "all")])
        assert s.step().applied
        path = tmp_path / "updated.pt"
        s.save(path)
        expected = s.checkpoint()
        r.model.nodes[0].bias = torch.nn.Parameter(r.model.nodes[0].bias.clone())
        with pytest.raises(RuntimeError, match="parameters changed"):
            s.checkpoint()
        s.close()
        fresh = runtime("pdg", target)
        with fresh.training_session(1, checkpoint=path) as restored:
            tree_equal(expected, restored.checkpoint())


@pytest.mark.parametrize("full,aggregation", [("tanh", "sum"), ("swiglu", "sum"),
    ("lh-silu-layer-v1", "sum"), *(('tanh', a) for a in
        ("mean", "weighted_mean", "active_softmax", "all_softmax"))])
def test_resident_training_new_process(target, tmp_path, full, aggregation):
    r = runtime("pdg", target, full=full, aggregation=aggregation)
    values = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) * .005
    path, output = tmp_path / "prefix.pt", tmp_path / "suffix.pt"
    with torch.no_grad(), r.training_session(1, optimizer="adamw") as s:
        for start, stop in ((0, 1), (1, 2)):
            args, kw = inputs(s, values, start, stop)
            w = s.advance_device(args, **kw)
            # One update per window here; retained windows are covered above.
            s.backward([roots(s, w, "all")]); assert s.step().applied
        s.save(path)
        args, kw = inputs(s, values, 2, 4)
        w = s.advance_device(args, **kw)
        s.backward([roots(s, w, "all")]); assert s.step().applied
        expected = s.checkpoint()
    command = [sys.executable, str(Path(__file__).with_name("resident_training_worker.py")),
               "--device", target, "--checkpoint", str(path), "--output", str(output), "--full", full,
               "--aggregation", aggregation]
    subprocess.run(command, check=True, timeout=180)
    tree_equal(expected, torch.load(output, weights_only=True))


def test_training_cpu_safe_options():
    for value in (False, 0, -1, 1.0, 1 << 63):
        with pytest.raises(ValueError, match="capacity"):
            ResidentTrainingLimits(windows=value)
    r = runtime("pdg", "cpu")
    with pytest.raises(ValueError, match="resident placement"):
        r.training_session(1)


@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("schedule", ["streaming", "greedy"])
@pytest.mark.parametrize("full,kind", [("swiglu", "adamw"), ("lh-silu-layer-v1", "sgd")])
def test_resident_extended_full_training(target, family, schedule, full, kind, tmp_path):
    training_case(target, family, schedule, kind, tmp_path, full)


@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("schedule", ["streaming", "greedy"])
@pytest.mark.parametrize("kind", ["sgd", "adamw"])
@pytest.mark.parametrize("aggregation", ["mean", "weighted_mean", "active_softmax", "all_softmax"])
def test_resident_normalized_aggregate_training(target, family, schedule, kind, aggregation, tmp_path):
    training_case(target, family, schedule, kind, tmp_path, aggregation=aggregation)
