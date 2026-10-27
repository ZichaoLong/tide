"""Python-owned sample replicas against independent CPU recursions and SGD/AdamW."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.reference import run
from tidegraph.weighted import WeightedResidentTraining
from resident_test_target import target_device
from resident_training_cases import tree_equal
from test_resident_weighted_training import binding, owner, cotangents
from weighted_precision_reference import reference_model, assert_records


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("family,topology,kind", [("kv", "pdg", "sgd"), ("ssd", "hb", "adamw")])
def test_public_sample_replicas_share_one_update(family, topology, kind, storage_dtype, tmp_path):
    device = torch.device(target_device())
    assert device.index == 0
    devices = [str(device), str(torch.device(device.type, 1))]
    candidate, _, base, span = binding(family, topology, storage_dtype)
    model = reference_model(candidate.model) if storage_dtype == torch.float16 else binding(family, topology)[0].model
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    masters = tuple(p.detach().clone().requires_grad_() for p in parameters)
    names = dict(model.named_parameters(remove_duplicate=False))
    options = dict(lr=.001, weight_decay=.01, **(dict(momentum=.3) if kind == "sgd" else dict(amsgrad=True)))
    optimizer = getattr(torch.optim, "SGD" if kind == "sgd" else "AdamW")(masters, **options)
    qs = [Continuation(candidate.graph.identity, 1) for _ in devices]
    with torch.no_grad():
        sessions = [owner(candidate, d, optimizer=kind, groups=[options]) for d in devices]
        for update in range(3):
            losses, inputs, partials = [], [], []
            for replica, session in enumerate(sessions):
                xs = [replace(x, time=x.time+update*span, position=x.position+2*update,
                    value=(x.value*torch.tensor([1., 1.+.1*replica, 1.], dtype=storage_dtype)).to(storage_dtype)) for x in base]
                reference_xs = [replace(x, value=x.value.float().requires_grad_()) for x in xs]
                inputs.extend(reference_xs)
                w = session.advance(xs, stop=(update+1)*span, sealed_until=(update+1)*span)
                mode = "zero" if update == 2 else "outputs"
                roots = cotangents(session, w, mode)
                with torch.enable_grad():
                    expected = run(candidate.graph, model, qs[replica], reference_xs, (update+1)*span, sealed_until=(update+1)*span)
                    qs[replica] = expected.continuation
                    losses.extend(x[-1].sum()*(0. if mode == "zero" else .07) for x in expected.outputs)
                assert_records(session.result(), expected, storage_dtype)
                partials.append(session.backward([roots]))
                if update == 1: session.accumulate()
            with torch.enable_grad():
                expected = torch.autograd.grad(torch.stack(losses).sum(), (*parameters, *(x.value for x in inputs)), allow_unused=True)
            if update == 0:
                with pytest.raises(ValueError): WeightedResidentTraining.step_replicas(sessions, communication_bytes=1)
                with pytest.raises(ValueError): WeightedResidentTraining.step_replicas([sessions[0], sessions[0]])
            result = WeightedResidentTraining.step_replicas(sessions)
            assert result.step.applied and result.transferred_bytes > 0
            assert 0 < result.reserved_tensor_bytes <= 256*1024*1024
            lookup = {id(p): value for p, value in zip(parameters, expected)}
            for reduced in result.parameters:
                assert reduced.values.dtype == torch.float32
                for name, offset, connected in zip(reduced.names, reduced.offsets, reduced.connected.cpu().tolist()):
                    p = names[name.replace(".extra.", ".")]; value = lookup[id(p)]
                    assert connected == (value is not None)
                    if connected:
                        torch.testing.assert_close(reduced.values[offset:offset+p.numel()].cpu().reshape(p.shape), value, rtol=3e-5, atol=3e-6)
            cursor = len(parameters)
            for partial, xs in zip(partials, [inputs[:len(base)], inputs[len(base):]]):
                boundary = partial.boundaries[0]
                for x in xs:
                    coordinate = [x.batch, candidate.graph.inputs[x.port], x.time, 0, x.port, x.position]
                    matches = [j for j, (c, valid) in enumerate(zip(boundary.coordinates.cpu().tolist(), boundary.valid.cpu().tolist())) if valid and c == coordinate]
                    assert len(matches) == 1
                    j = matches[0]; value = expected[cursor]; cursor += 1
                    assert boundary.connected[j].item() == (value is not None)
                    if value is not None: torch.testing.assert_close(boundary.values[j].cpu(), value, rtol=3e-5, atol=3e-6)
            optimizer.zero_grad(set_to_none=True)
            for master, value in zip(masters, expected): master.grad = value
            optimizer.step()
            for p, master in zip(parameters, masters): p.copy_(master.to(storage_dtype).float())
            checkpoints = [session.checkpoint() for session in sessions]
            tree_equal(checkpoints[0]["state"], checkpoints[1]["state"])
            tree_equal(checkpoints[0]["parameters"], checkpoints[1]["parameters"])
            for name, value in checkpoints[0]["parameters"].items():
                translated = name.replace(".extra.", ".")
                if translated in names: torch.testing.assert_close(value.float(), names[translated], rtol=1e-5, atol=1e-6)
            if update == 0:
                for i, session in enumerate(sessions):
                    path = tmp_path/f"replica-{i}.pt";session.save(path);session.close()
                    sessions[i] = owner(candidate, devices[i], checkpoint=path)
                    tree_equal(checkpoints[i], sessions[i].checkpoint())
            qs = [q.detach() for q in qs]
        # A replica can carry a valid local checkpoint but disagree with the
        # other replica's optimizer history. Neither may commit this collective.
        checkpoints = [s.checkpoint() for s in sessions]
        for s in sessions: s.close()
        checkpoints[1]["state"]["first"] += .001
        sessions = [owner(candidate, d, checkpoint=c) for d, c in zip(devices, checkpoints)]
        for s in sessions:
            w = s.advance([], stop=s.cut, sealed_until=s.cut)
            s.backward([s.cotangents(w)])
        refused = WeightedResidentTraining.step_replicas(sessions)
        assert not refused.step.applied and refused.step.refusal_code == 25
        for s, original in zip(sessions, checkpoints):
            s.detach(); after = s.checkpoint()
            tree_equal(after["state"], original["state"])
            tree_equal(after["parameters"], original["parameters"])
            assert after["generation"] == original["generation"]
            s.close()
