"""Explicit opt-in archive retrieval; preserve checksums and output safety."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def test_reader_explicit_archive_and_failure_boundaries(tmp_path):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "docs/evidence").mkdir(parents=True)
    script = root / "scripts/read_retired_record.py"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/read_retired_record.py", script)
    content = b"historical patch bytes\n"
    record = {"path": "/original/path:patch", "kind": "source-patch", "bytes": len(content),
              "sha256": hashlib.sha256(content).hexdigest(), "data_b64": base64.b64encode(content).decode()}
    archive = tmp_path / "archive.gz"
    archive.write_bytes(gzip.compress((json.dumps(record) + "\n").encode()))
    (root / "docs/evidence/artifact-retirement-20261012.json").write_text(json.dumps({"archive": {
        "path": "docs/evidence/archive/absent.gz", "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}}))
    def run(*args):
        return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)
    missing = run("--list-prefix", "/original")
    assert missing.returncode != 0 and "--archive PATH" in missing.stderr
    output = tmp_path / "patch"
    args = ("--archive", str(archive), "--path", record["path"], "--output", str(output))
    assert run(*args).returncode == 0
    assert output.read_bytes() == content
    assert run(*args).returncode != 0 and output.read_bytes() == content
    assert run("--archive", str(archive), "--path", "absent").returncode != 0
    archive.write_bytes(b"corrupt")
    other = tmp_path / "other"
    failed = run(*args[:-1], str(other))
    assert failed.returncode != 0 and "checksum mismatch" in failed.stderr and not other.exists()
