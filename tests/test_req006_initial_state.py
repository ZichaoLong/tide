"""Paired diagnostic material: exact next update and fail-closed publication."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import req006_initial as initial
from req006_material import synthetic_inputs, tensor_digest
from req006_runtime import capture, restore

CPU = torch.device("cpu")


def owner(dtype=torch.float32):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.25)).to(dtype=dtype)
    return model, torch.optim.AdamW(model.parameters(), lr=.01, foreach=False)


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    output = model(torch.randn(2, 3, dtype=next(model.parameters()).dtype))
    output.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return output.detach().clone()


@pytest.fixture
def material():
    model, optimizer = owner()
    update(model, optimizer)
    arguments = dict(config=dict(seed=11, dtype="float32", training=dict(batch=2)),
        material=synthetic_inputs(2, 3, 17), progress=29, input_seed=17, prime_updates=1,
        runtime=dict(backend="cpu", dtype="float32", torch=torch.__version__, threads=dict(intraop=1, interop=1)),
        downstream=dict(commit="a"*40, content_sha256="b"*64))
    return capture(model, optimizer, CPU), arguments


def write(path, state, identity):
    return initial.save_initial(path, state, identity, producer=dict(source="c"*40, dirty=True))


def test_populated_state_restores_exact_next_random_update(tmp_path, dtype):
    model, optimizer = owner(dtype)
    update(model, optimizer)
    state = capture(model, optimizer, CPU)
    assert all(int(slot["step"]) == 1 for slot in state["optimizer"]["state"].values())
    identity = dict(dtype=str(dtype), torch=torch.__version__)
    artifact = tmp_path/"initial.pt"
    published = write(artifact, state, identity)
    loaded, metadata = initial.load_initial(artifact, identity)
    assert metadata == published
    assert tensor_digest(loaded) == tensor_digest(state)
    clone, clone_optimizer = owner(dtype)
    restore(state, model, optimizer, CPU)
    expected_output = update(model, optimizer)
    expected_state = capture(model, optimizer, CPU)
    restore(loaded, clone, clone_optimizer, CPU)
    assert tensor_digest(capture(clone, clone_optimizer, CPU)) == published["initial_state_sha256"]
    actual_output = update(clone, clone_optimizer)
    torch.testing.assert_close(actual_output, expected_output, atol=0, rtol=0)
    assert tensor_digest(capture(clone, clone_optimizer, CPU)) == tensor_digest(expected_state)
    assert tensor_digest(loaded) == published["initial_state_sha256"]


@pytest.mark.parametrize("field", ["config", "material", "progress", "input_seed", "prime_updates",
                                  "dtype", "runtime", "source", "source_content", "threads"])
def test_mismatched_identity_is_rejected(tmp_path, material, field):
    state, arguments = material
    artifact = tmp_path/"initial.pt"
    identity = initial.initial_identity(**arguments)
    published = write(artifact, state, identity)
    changed = copy.deepcopy(arguments)
    if field == "config":
        changed["config"]["training"]["batch"] += 1
    elif field == "material":
        changed["material"]["labels"][0, 0] += 1
    elif field in {"progress", "input_seed", "prime_updates"}:
        changed[field] += 1
    elif field == "dtype":
        changed["config"]["dtype"] = "float64"
    elif field == "runtime":
        changed["runtime"]["torch"] = "another-version"
    elif field == "threads":
        changed["runtime"]["threads"]["intraop"] += 1
    else:
        changed["downstream"]["commit" if field == "source" else "content_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity mismatch"):
        initial.load_initial(artifact, initial.initial_identity(**changed))
    _, metadata = initial.load_initial(artifact, identity)
    assert metadata == published


def test_physical_ordinal_and_action_are_not_material_identity(material):
    _, arguments = material
    identity = initial.initial_identity(**arguments)
    changed = copy.deepcopy(arguments)
    changed["runtime"].update(resolved_device="cpu:7", logical_device_index=7,
                              action="profile", mode="grad-forward")
    assert initial.initial_identity(**changed) == identity
    assert type(identity["runtime"]["torch"]) is str


@pytest.mark.parametrize("field", ["state", "initial_state_sha256", "weights_sha256", "identity_sha256", "schema"])
def test_payload_and_identity_hashes_are_verified(tmp_path, material, field):
    state, arguments = material
    identity = initial.initial_identity(**arguments)
    artifact = tmp_path/"initial.pt"
    write(artifact, state, identity)
    document = torch.load(artifact, weights_only=True)
    if field == "state":
        document["state"]["model"]["0.weight"][0, 0] += 1
    else:
        document[field] = "changed"
    torch.save(document, artifact)
    with pytest.raises(ValueError, match="digest mismatch|schema"):
        initial.load_initial(artifact, identity)


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_output_is_never_replaced(tmp_path, material, symlink):
    state, arguments = material
    original = tmp_path/"original.pt"
    original.write_bytes(b"previous known-good bytes")
    artifact = tmp_path/"initial.pt"
    if symlink:
        artifact.symlink_to(original)
    else:
        artifact.write_bytes(original.read_bytes())
    with pytest.raises(FileExistsError):
        write(artifact, state, initial.initial_identity(**arguments))
    assert artifact.read_bytes() == b"previous known-good bytes"
    assert original.read_bytes() == artifact.read_bytes()


@pytest.mark.parametrize("failure", ["write", "concurrent_publish", "no_space"])
def test_failed_publication_preserves_existing_material(tmp_path, material, monkeypatch, failure):
    state, arguments = material
    artifact = tmp_path/"initial.pt"
    if failure == "write":
        def failed_save(document, stream):
            stream.write(b"partial")
            raise OSError("injected write failure")
        monkeypatch.setattr(torch, "save", failed_save)
    elif failure == "concurrent_publish":
        real_link = initial.os.link
        def raced_link(staging, destination):
            destination.write_bytes(b"other writer")
            real_link(staging, destination)
        monkeypatch.setattr(initial.os, "link", raced_link)
    else:
        monkeypatch.setattr(initial.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(OSError):
        write(artifact, state, initial.initial_identity(**arguments))
    assert not list(tmp_path.glob(".*.tmp"))
    if failure == "concurrent_publish":
        assert artifact.read_bytes() == b"other writer"
    else:
        assert not artifact.exists()


def test_byte_bound_is_checked_before_loading(tmp_path, material, monkeypatch):
    state, arguments = material
    artifact = tmp_path/"initial.pt"
    identity = initial.initial_identity(**arguments)
    write(artifact, state, identity)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("oversize file was loaded"))
    with pytest.raises(ValueError, match="byte bound"):
        initial.load_initial(artifact, identity, max_bytes=artifact.stat().st_size-1)
