"""All public package identities must observe actual nested implementation bytes."""
import hashlib
import json
import subprocess
import sys

import pytest

from tidegraph.package_content import CONTENT_SCOPE, content_identity


def populate(root, *, reverse=False):
    items = [("__init__.py", b"# package\n"), ("weighted/kernel.py", b"VALUE = 1\n"),
             ("planning/data.json", b'{"version":1}\n')]
    for name, value in reversed(items) if reverse else items:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    return items


def test_content_hash_is_sorted_recursive_location_independent_and_matches_existing_lock(tmp_path):
    first, second = tmp_path/"first", tmp_path/"second"
    items = populate(first)
    populate(second, reverse=True)
    actual = content_identity(first)
    assert actual == content_identity(second)
    # Independent formula pins the pre-existing experiment-lock hash, including
    # non-Python resources; this is not just two entry points sharing a helper.
    expected = {name: hashlib.sha256(value).hexdigest() for name, value in items}
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert actual["files_sha256"] == expected and actual["installed_files_sha256"] == digest
    assert actual["content_scope"] == CONTENT_SCOPE
    (first/"weighted/__pycache__").mkdir()
    (first/"weighted/__pycache__/kernel.pyc").write_bytes(b"cache")
    (first/"orphan.pyo").write_bytes(b"cache")
    assert content_identity(first) == actual
    (first/"planning/data.json").write_text('{"version":2}\n')
    assert content_identity(first)["installed_files_sha256"] != digest
    with pytest.raises(ValueError, match="no readable files"):
        content_identity(tmp_path/"absent")


def test_public_runtime_manifest_covers_subpackages_and_matches_experiment_identity(tmp_path, monkeypatch):
    from tidegraph import GraphConfig, GraphRuntime
    from tidegraph import library
    from tidegraph.experiment.identity import package_identity
    populate(tmp_path)
    config = GraphConfig.from_dict(dict(schema_version=1, family="pdg", topology=dict(kind="chain")))
    runtime = GraphRuntime(config, device="cpu")
    monkeypatch.setattr(library, "__file__", str(tmp_path/"library.py"))
    before = runtime.manifest()
    assert before["schema"] == "tide-runtime-v2"
    assert before["package_hash_scope"] == CONTENT_SCOPE
    assert before["package_sha256"] == package_identity(tmp_path)["installed_files_sha256"]
    (tmp_path/"weighted/kernel.py").write_text("VALUE = 2\n")
    after = runtime.manifest()
    assert before["package_sha256"] != after["package_sha256"]
    assert after["package_sha256"] == package_identity(tmp_path)["installed_files_sha256"]


def test_content_helper_requires_no_torch_or_site_packages(tmp_path):
    import tidegraph.package_content as module
    populate(tmp_path)
    script = "import runpy,sys; f=runpy.run_path(sys.argv[1]); print(f['content_identity'](sys.argv[2])['installed_files_sha256']); assert 'torch' not in sys.modules"
    result = subprocess.check_output([sys.executable, "-S", "-c", script, module.__file__, str(tmp_path)], text=True)
    assert result.strip() == content_identity(tmp_path)["installed_files_sha256"]
