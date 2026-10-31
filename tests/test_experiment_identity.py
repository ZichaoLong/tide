import json
from pathlib import Path
import zipfile
import pytest
from tidegraph.experiment import identity
from tidegraph.experiment.records import digest, read_json, write_json


def test_installed_identity_covers_nested_source_and_ignores_cache(tmp_path):
    root = tmp_path/"package"
    (root/"weighted").mkdir(parents=True)
    (root/"__init__.py").write_text("# package\n")
    source = root/"weighted"/"runtime.py"
    source.write_text("VALUE = 1\n")
    first = identity.package_identity(root)
    (root/"__pycache__").mkdir()
    (root/"__pycache__"/"cache.pyc").write_bytes(b"cache")
    assert identity.package_identity(root)["installed_files_sha256"] == first["installed_files_sha256"]
    source.write_text("VALUE = 2\n")
    assert identity.package_identity(root)["installed_files_sha256"] != first["installed_files_sha256"]
    assert "requires an explicit dependency lock" in first["source_provenance"]


def test_build_identity_checks_exact_bytes_not_just_source_label(tmp_path):
    binary = tmp_path/"_tide_native.so"
    binary.write_bytes(b"example binary")
    manifest = dict(source="historical-fixed-source", backend="cpu", binary_sha256={binary.name: digest(binary)})
    write_json(tmp_path/"build-manifest.json", manifest)
    result = identity.build_identity(tmp_path)
    assert result["source"] == "historical-fixed-source"
    binary.write_bytes(b"replacement binary")
    with pytest.raises(ValueError, match="binary changed"):
        identity.build_identity(tmp_path)


def test_unsafe_manifest_paths_are_refused(tmp_path):
    write_json(tmp_path/"build-manifest.json", dict(binary_sha256={"../other": "fake"}))
    with pytest.raises(ValueError, match="invalid binary path"):
        identity.build_identity(tmp_path)


def test_freeze_verification_rejects_installation_tamper(monkeypatch):
    value = {"installed_files_sha256": "first"}
    monkeypatch.setattr(identity, "package_identity", lambda: dict(value))
    frozen = identity.capture()
    identity.verify(frozen)
    value["installed_files_sha256"] = "second"
    with pytest.raises(ValueError, match="content changed"):
        identity.verify(frozen)


def test_atomic_record_failure_keeps_previous_complete_bytes(tmp_path):
    path = tmp_path/"status.json"
    write_json(path, dict(state="running"))
    with pytest.raises(ValueError):
        write_json(path, dict(value=float("nan")))
    assert read_json(path) == dict(state="running")
    with pytest.raises(FileExistsError):
        write_json(path, dict(state="passed"), exclusive=True)


def test_lock_wheel_matches_installed_bytes_and_content_is_checked(tmp_path, monkeypatch):
    package = tmp_path/"package"
    package.mkdir()
    (package/"__init__.py").write_text("VALUE = 1\n")
    actual_identity = identity.package_identity
    monkeypatch.setattr(identity, "package_identity", lambda: actual_identity(package))
    wheel = tmp_path/"example.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(package/"__init__.py", "tidegraph/__init__.py")
    lock = identity.create_lock(wheel=wheel)
    assert lock["wheel"]["installed_files_verified"]
    assert identity.verify_lock(lock)["state"] == "passed"
    (package/"__init__.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="content changed"):
        identity.verify_lock(lock)
    with pytest.raises(ValueError, match="wheel tidegraph files differ"):
        identity.create_lock(wheel=wheel)


def test_lock_refuses_dependency_metadata_change(monkeypatch):
    monkeypatch.setattr(identity, "package_identity", lambda: {"installed_files_sha256": "fixed"})
    state = {"torch": {"version": "one", "record_sha256": "record"}}
    monkeypatch.setattr(identity, "dependencies", lambda: json.loads(json.dumps(state)))
    lock = identity.create_lock()
    state["torch"]["version"] = "two"
    with pytest.raises(ValueError, match="dependency metadata changed"):
        identity.verify_lock(lock)
