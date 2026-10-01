"""Public half loss roots, master/checkpoint lifecycle and independent scalar oracle."""
import copy
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from tidegraph import Graph, Node, Region, GraphConfig
from test_resident_library import target, inputs, runtime as make_runtime
from resident_training_cases import tree_equal
from resident_half_training_cases import runtime, limits, roots, update, payload_master

MEMORIES = ("ema", "attention", "lh-fiber-attention-all-softmax-repeat-v1")


@pytest.mark.parametrize("family", ("pdg", "timed-dag", "settle"))
@pytest.mark.parametrize("schedule", ("streaming", "greedy"))
@pytest.mark.parametrize("kind", ("sgd", "adamw"))
@pytest.mark.parametrize("memory", MEMORIES)
def test_half_client_lifecycle(target, family, schedule, kind, memory, tmp_path):
    r = runtime(family, target, schedule, memory, "softp")
    values = (torch.sin(torch.arange(64).reshape(2, 8, 4)*.37)*.1).half()
    values[0, 1].zero_()
    with torch.no_grad():
        session = r.training_session(2, optimizer=kind, limits=limits())
        before = session.checkpoint()
        for step, mode in enumerate(("all", "zero", "none", "all")):
            cot = []
            for start, stop in ((step*2, step*2+1), (step*2+1, step*2+2)):
                args, kw = inputs(session, values.to(target) if step == 1 else values, start, stop)
                window = session.advance_device(*args, **kw)
                assert window.outputs.values.dtype == torch.float16
                cot.append(roots(session, window, mode))
            gradient = session.backward(cot)
            assert gradient.values.dtype == torch.float32
            if mode == "none":
                assert not gradient.connected.any()
            assert session.step().applied
            saved = session.checkpoint()
            payload_master(saved)
            if mode == "none":
                tree_equal(before["state"], saved["state"])
            if step == 1:
                path = tmp_path / "half.pt"
                session.save(path)
                session.close()
                other = runtime(family, target, "streaming" if schedule == "greedy" else "greedy", memory, "softp")
                session = other.training_session(2, checkpoint=path, limits=limits())
                tree_equal(saved, session.checkpoint())
                wrong = copy.deepcopy(saved)
                name = next(iter(wrong["parameters"]))
                wrong["parameters"][name] = wrong["parameters"][name].float()
                with pytest.raises(ValueError, match="checkpoint parameter"):
                    other.training_session(2, checkpoint=wrong, limits=limits())
            before = saved
        assert not torch.equal(before["state"]["values"], before["state"]["values"].half().float())
        session.close()


@pytest.mark.parametrize("memory", MEMORIES)
def test_half_training_fresh_process(target, memory, tmp_path):
    r = runtime("pdg", target, memory=memory, mode="softp")
    values = (torch.arange(16).reshape(1, 4, 4)*.005).half()
    prefix, suffix = tmp_path / "prefix.pt", tmp_path / "suffix.pt"
    with torch.no_grad(), r.training_session(1, optimizer="adamw", limits=limits()) as session:
        update(session, values, 0, 1)
        update(session, values, 1, 2)
        session.save(prefix)
        update(session, values, 2, 4)
        expected = session.checkpoint()
    subprocess.run([sys.executable, str(Path(__file__).with_name("resident_half_training_worker.py")),
        "--device", target, "--memory", memory, "--checkpoint", str(prefix), "--output", str(suffix)], check=True, timeout=180)
    tree_equal(expected, torch.load(suffix, weights_only=True))


@pytest.mark.parametrize("kind", ("sgd", "adamw"))
def test_half_training_independent_scalar_oracle(target, kind):
    # A public one-node linear graph has a separate analytic Torch schedule:
    # output = half(half(x * half(input_scale)) * half(output_scale)).
    # The reference produces no runtime input/route/gradient for the candidate.
    graph = Graph((Node(0, identity=True),), (), (Region(1),), (0,), (0,))
    r = make_runtime(GraphConfig("pdg", graph, width=3, dtype="float16"), target)
    named = dict(r.execution_model.named_parameters())
    master = {n: p.detach().cpu().float().requires_grad_(True) for n, p in named.items() if p.requires_grad}
    settings = dict(lr=.001, weight_decay=.01)
    settings.update(momentum=.5) if kind == "sgd" else settings.update(eps=.001, amsgrad=True)
    optimizer = getattr(torch.optim, "SGD" if kind == "sgd" else "AdamW")(list(master.values()), **settings)
    values = torch.tensor([[[.113, -.0523, .281], [0., 0., 0.], [.71, -.14, .09], [.12, .61, -.41], [.31, -.29, .78], [.55, .43, .19]]]).half()
    rounded = lambda x: x + (x.detach().half().float() - x.detach())
    with torch.no_grad(), r.training_session(1, optimizer=kind,
            groups=[dict(parameters=list(master), **settings)]) as session:
        for step in range(3):
            start, stop = step*2, step*2+2
            with torch.enable_grad():
                y = rounded(rounded(values[0, start:stop].float()*rounded(master["input_scale.0"])) * rounded(master["output_scale.0"]))
                loss = y.square().sum()*.0625
                expected = torch.autograd.grad(loss, tuple(master.values()), allow_unused=True)
            args, kw = inputs(session, values, start, stop)
            window = session.advance_device(*args, **kw)
            with torch.enable_grad():
                leaf = window.outputs.values.detach().float().requires_grad_(True)
                actual_loss = torch.where(window.outputs.valid[:, None], leaf, torch.zeros_like(leaf)).square().sum()*.0625
                cotangent, = torch.autograd.grad(actual_loss, (leaf,))
            gradient = session.backward([session.cotangents(window, outputs=cotangent.detach())])
            by_name = dict(zip(master, expected))
            for i, name in enumerate(gradient.names):
                ref = by_name[name]
                assert gradient.connected[i].item() == (ref is not None), name
                if ref is not None:
                    actual = gradient.values.cpu().narrow(0, gradient.offsets[i], ref.numel()).reshape(ref.shape)
                    torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-3)
            optimizer.zero_grad(set_to_none=True)
            for p, g in zip(master.values(), expected):
                p.grad = g
            optimizer.step()
            assert session.step().applied
            saved = session.checkpoint()
            payload_master(saved)
            for i, name in enumerate(gradient.names):
                if saved["offsets"][i] >= 0:
                    ref = master[name]
                    actual = saved["state"]["values"].narrow(0, saved["offsets"][i], ref.numel()).reshape(ref.shape)
                    torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-3)
