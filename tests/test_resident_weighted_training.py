"""Public Python-owned weighted training versus independent Python recursion.

The standalone complete-field/cross-cut checks and local formula anchors are
separate. This gate exercises the Python adapter, real cotangents and disk cuts.
"""
import os
import subprocess
import sys
from pathlib import Path
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.compare import equivalent
from tidegraph.reference import run as reference
from tidegraph.weighted import WeightedResidentTraining
from resident_test_target import target_device
from resident_training_cases import tree_equal
from test_weighted_execution import setup
from test_weighted_recurrent_execution import model as recurrent_model


def binding(family, topology):
    return setup(torch.float32, topology) if family == "kv" else recurrent_model(torch.float32, family, topology=topology)


def owner(b, device, **kw):
    if "groups" in kw:
        from tidegraph.native_loader import load_native
        from tidegraph.native_model import encode_model
        core = load_native(os.environ["TIDE_BUILD_DIR"], backend=str(device).split(":")[0])
        _, model = encode_model(core, b.graph, b.model)
        names = [p.canonical for p in model.parameters().owners()]
        kw["groups"] = [dict(parameters=names, **group) for group in kw["groups"]]
    return WeightedResidentTraining(b.graph, b.model, device=device,
        native_library=os.environ["TIDE_BUILD_DIR"], resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],
        limits=dict(queue=64, stages=32, cache=16), **kw)


def cotangents(session, w, mode):
    # A separate consumer autograd context creates detached task-loss roots.
    f = w.fields["outputs"]
    with torch.enable_grad():
        x = f.values.detach().clone().requires_grad_()
        loss = torch.where(f.present.unsqueeze(-1), x, torch.zeros_like(x)).sum() * (.07 if mode == "outputs" else 0.)
        bar, = torch.autograd.grad(loss, x)
    return session.cotangents(w, {"outputs": (bar, f.present)})


@pytest.mark.parametrize("family,topology,kind", [
    ("kv", "fc", "sgd"), ("kv", "hb", "adamw"), ("kv", "dag", "sgd"), ("kv", "pdg", "adamw"),
    ("affine", "pdg", "sgd"), ("delta", "pdg", "adamw"), ("gated_delta", "pdg", "sgd"), ("ssd", "pdg", "adamw")])
def test_python_owned_complete_cut_updates_and_disk_resume(family, topology, kind, tmp_path):
    device = target_device()
    candidate, _, base, span = binding(family, topology)
    cpu, _, _, _ = binding(family, topology)
    parameters = tuple(p for p in cpu.model.parameters() if p.requires_grad)
    names = dict(cpu.model.named_parameters(remove_duplicate=False))
    options = dict(lr=.001, weight_decay=.01)
    if kind == "sgd": options["momentum"] = .3
    else: options["amsgrad"] = True
    optimizer = getattr(torch.optim, "SGD" if kind == "sgd" else "AdamW")(parameters, **options)
    q = Continuation(cpu.graph.identity, 1)
    with torch.no_grad():
        session = owner(candidate, device, optimizer=kind, groups=[options])
        assert session.manifest()["effective_optimizer"] == kind
        assert session.manifest()["effective_groups"][0]["parameters"]
        initial = session.checkpoint()["state"]["values"].clone()
        for update, mode in enumerate(("outputs", "zero", "outputs")):
            xs = [replace(x, time=x.time+update*span, position=x.position+2*update,
                          value=x.value.detach().clone().requires_grad_()) for x in base]
            split = update*span+span//2
            groups = ([x for x in xs if x.time < split], [], [x for x in xs if x.time >= split])
            roots, losses = [], []
            for part, stop in zip(groups, (split, split, (update+1)*span)):
                w = session.advance(part, stop=stop, sealed_until=stop)
                roots.append(cotangents(session, w, mode))
                with torch.enable_grad():
                    expected = reference(cpu.graph, cpu.model, q, part, stop, sealed_until=stop)
                    q = expected.continuation
                    losses.extend(x[-1].sum()*(.07 if mode == "outputs" else 0.) for x in expected.outputs)
                equivalent(session.result(), expected)
                for name, value in w.metadata.items():
                    if not name.endswith("present"): assert value.dtype == torch.int64
            with torch.enable_grad():
                expected = torch.autograd.grad(torch.stack(losses).sum(), (*parameters, *(x.value for x in xs)), allow_unused=True)
            actual = session.backward(roots)
            lookup = {id(p): value for p, value in zip(parameters, expected)}
            p = actual.parameters
            for name, offset, flag in zip(p.names, p.offsets, p.connected.cpu().tolist()):
                parameter = names[name.replace(".extra.", ".")]
                value = lookup[id(parameter)]
                assert flag == (value is not None), name
                if flag:
                    torch.testing.assert_close(p.values[offset:offset+parameter.numel()].cpu().reshape(parameter.shape), value, rtol=3e-5, atol=3e-6)
            for i, x in enumerate(xs):
                found = []
                for boundary in actual.boundaries:
                    for j, (meta, valid) in enumerate(zip(boundary.coordinates.cpu().tolist(), boundary.valid.cpu().tolist())):
                        if valid and meta == [x.batch, cpu.graph.inputs[x.port], x.time, 0, x.port, x.position]: found.append((boundary, j))
                assert len(found) == 1
                boundary, j = found[0];value = expected[len(parameters)+i]
                assert boundary.connected[j].item() == (value is not None)
                if value is not None: torch.testing.assert_close(boundary.values[j].cpu(), value, rtol=3e-5, atol=3e-6)
            optimizer.zero_grad(set_to_none=True)
            for p, value in zip(parameters, expected): p.grad = value
            optimizer.step(); assert session.step().applied
            checkpoint = session.checkpoint()
            for name, value in checkpoint["parameters"].items():
                translated = name.replace(".extra.", ".")
                if translated in names: torch.testing.assert_close(value, names[translated], rtol=1e-5, atol=1e-6)
            if update == 0:
                assert not torch.equal(initial, checkpoint["state"]["values"])
                assert checkpoint["state"]["steps"].gt(0).any()
                path = tmp_path/"weighted-training.pt";session.save(path)
                with pytest.raises(FileExistsError): session.save(path)
                session.close();session = owner(candidate, device, checkpoint=path)
                tree_equal(checkpoint, session.checkpoint())
                assert session.manifest()["initialization"] == "resume"
                assert session.manifest()["effective_groups"] == checkpoint["groups"]
            q = q.detach()
        assert session.manifest()["training"] and session.generation == 3
        if family == "kv" and topology == "fc":
            from weighted_training_worker import continue_cut
            source, result = tmp_path/"before-process.pt", tmp_path/"after-process.pt"
            session.save(source);continue_cut(session, base, span)
            expected = session.checkpoint();session.close()
            subprocess.run([sys.executable, str(Path(__file__).with_name("weighted_training_worker.py")),
                            device, str(source), str(result)], check=True, timeout=120)
            tree_equal(expected, torch.load(result, map_location="cpu", weights_only=True))
        session.close()


def test_weighted_training_requires_explicit_no_grad_before_backend_access():
    b, _, _, _ = setup(torch.float32, "fc")
    with pytest.raises(ValueError, match="no_grad"):
        WeightedResidentTraining(b.graph, b.model, device="cpu", native_library="unused", resident_library="unused")


def test_weighted_training_checkpoint_rejects_wrong_binding_before_native_access():
    from types import SimpleNamespace
    from tidegraph.weighted.resident_training_checkpoint import prepare
    with pytest.raises(ValueError, match="schema"):
        prepare(SimpleNamespace(), {"schema": "tide-resident-training-v1"})
    for zeta in (True, float("nan"), 2.):
        with pytest.raises(ValueError, match="binding"):
            prepare(SimpleNamespace(), dict(schema="tide-weighted-resident-training-v1", mode="weighted", zeta=zeta))
