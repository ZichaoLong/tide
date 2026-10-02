"""Independent CPU autograd with explicit truncated groups and one update."""
import pytest
import torch
from tidegraph import ResidentPlacement, ResidentTrainingLimits
from tidegraph.compare import equivalent
from test_resident_training import target
from resident_training_cases import (runtime, inputs, roots, terms, compare_gradients,
                                     compare_parameters, tree_equal)


@pytest.mark.parametrize("family,schedule,kind,cards,mode", [
    ("pdg", "streaming", "sgd", 1, "hard"),
    ("pdg", "greedy", "adamw", 2, "hst"),
    ("timed-dag", "streaming", "adamw", 2, "softp"),
    ("timed-dag", "greedy", "sgd", 1, "hard"),
    ("settle", "streaming", "sgd", 2, "hst"),
    ("settle", "greedy", "adamw", 1, "softp"),
])
def test_accumulated_updates(target, family, schedule, kind, cards, mode, tmp_path):
    r = runtime(family, target, schedule, mode=mode, model_device="cpu", resident_workspace_bytes=1024**3)
    cpu = runtime(family, "cpu", mode=mode)
    first = torch.device(target).index
    placement = ResidentPlacement(devices=tuple(f"npu:{first+i}" for i in range(cards))) if cards>1 else None
    names, parameters = zip(*((n,p) for n,p in cpu.execution_model.named_parameters() if p.requires_grad))
    settings = dict(lr=.001, weight_decay=.01)
    settings.update(momentum=.5) if kind == "sgd" else settings.update(eps=.0001, amsgrad=True)
    opt = getattr(torch.optim, "SGD" if kind == "sgd" else "AdamW")(parameters, **settings)
    values = torch.sin(torch.arange(2*18*4).reshape(2, 18, 4)*.37)*.1
    values[0, 1].zero_()
    oracle = cpu.session(2)
    limits = ResidentTrainingLimits(windows=2, backward_bytes=8*1024**3)
    with torch.no_grad():
        session = r.training_session(2, optimizer=kind, groups=[dict(parameters=list(names), **settings)],
                                     placement=placement, limits=limits)
        before = session.checkpoint()
        position = 0
        # First update sums two real gradients; the following updates separate
        # connected zero from all-None, including momentum/decay/slot counters.
        for step, modes in enumerate((("all", "none", "all"), ("none", "zero", "none"), ("none",)*3)):
            opt.zero_grad(set_to_none=True)
            retained_exports = []
            for group, root_mode in enumerate(modes):
                x = values.clone().requires_grad_(True)
                cotangents, objectives = [], []
                for _ in range(2):
                    with torch.enable_grad():
                        args, kw = inputs(oracle, x, position, position+1)
                        expected = oracle.advance(args, **kw)
                        objectives.extend(terms(expected, oracle.continuation, root_mode))
                    args, kw = inputs(session, values, position, position+1)
                    window = session.advance_device(args, **kw)
                    cotangents.append(roots(session, window, root_mode))
                    equivalent(expected, session.result())
                    position += 1
                with pytest.raises(RuntimeError, match="outstanding|backward"):
                    session.step()
                with torch.enable_grad():
                    grad = (torch.autograd.grad(torch.stack(objectives).sum(), (*parameters, x), allow_unused=True)
                            if objectives else (None,)*(len(parameters)+1))
                actual = session.backward(cotangents)
                compare_gradients(actual, cpu.execution_model, {id(p): g for p, g in zip(parameters, grad)}, x, grad[-1])
                for p, g in zip(parameters, grad):
                    if g is not None:
                        p.grad = g if p.grad is None else p.grad+g
                with pytest.raises(ValueError, match="budget"):
                    session.accumulate(max_bytes=1)
                assert session.accumulated_batches == group
                if group:
                    with pytest.raises(RuntimeError, match="accumulate"):
                        session.step()
                old = [(s.values.clone(), s.connected.clone()) for s in (actual.parameter_shards or [actual])]
                retained_exports.extend(zip(actual.parameter_shards or [actual], old))
                session.accumulate()
                assert session.accumulated_batches == group+1 and session.generation == step
                # Keep all preceding exports alive across later accumulations.
                for s, (v, c) in retained_exports:
                    torch.testing.assert_close(s.values, v, atol=0, rtol=0)
                    assert torch.equal(s.connected, c)
                with pytest.raises(RuntimeError, match="backward"):
                    session.accumulate()
                with pytest.raises(RuntimeError, match="detach"):
                    session.checkpoint()
                oracle.detach()
            opt.step()
            assert session.step().applied
            assert session.generation == step+1 and session.accumulated_batches == 0
            saved = session.checkpoint()
            compare_parameters(saved, cpu.execution_model)
            if step == 2:
                tree_equal(before["state"], saved["state"])
            if step == 0:
                path = tmp_path/"accumulated.pt"
                session.save(path)
                session.close()
                session = r.training_session(2, checkpoint=path, placement=placement, limits=limits)
                tree_equal(saved, session.checkpoint())
            before = saved
        session.close()


@pytest.mark.parametrize("cards", [1, 2])
def test_accumulation_failure_and_discard(target, cards):
    r = runtime("pdg", target, model_device="cpu", resident_workspace_bytes=1024**3)
    first = torch.device(target).index
    placement = ResidentPlacement(devices=tuple(f"npu:{first+i}" for i in range(cards))) if cards>1 else None
    with torch.no_grad(), r.training_session(1, optimizer="adamw", placement=placement,
            limits=ResidentTrainingLimits(backward_bytes=8*1024**3)) as session:
        before = session.checkpoint()
        for budget in (True, 0, -1, 1.5, 2**63):
            with pytest.raises(ValueError):
                session.accumulate(max_bytes=budget)
        with pytest.raises(RuntimeError, match="backward"):
            session.accumulate()
        values = torch.ones(1, 2, 4)*.1
        args, kw = inputs(session, values, 0, 1)
        window = session.advance_device(args, **kw)
        root = roots(session, window, "all")
        root.outputs.fill_(float("nan"))
        if window.states:
            for state in root.states:
                state.final.fill_(float("nan"))
        else:
            root.final.fill_(float("nan"))
        session.backward([root])
        session.accumulate()
        refused = session.step()
        assert not refused.applied and refused.refusal_code == 20 and session.generation == 0
        assert session.accumulated_batches == 1
        session.detach()
        assert session.accumulated_batches == 0 and session.cut == 1
        tree_equal(before["state"], session.checkpoint()["state"])
        with pytest.raises(RuntimeError, match="backward"):
            session.step()
        args, kw = inputs(session, values, 1, 2)
        window = session.advance_device(args, **kw)
        actual = session.backward([roots(session, window, "all")])
        session.accumulate()
        # An adversarial caller must not be able to rewrite the frozen bank by
        # retaining a backward export. Normal clients treat views as read-only.
        for shard in actual.parameter_shards or [actual]:
            shard.values.fill_(float("nan"))
        assert session.step().applied
