import json
from pathlib import Path
import subprocess
import sys
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_original_iocortex_scope_inventory():
    import _tide_native
    binary = Path(_tide_native.__file__).with_name("tidegraph-lh-scope-check")
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    assert "full=204, assertions-on-fp64=180, smoke=6; passed" in result.stdout


@pytest.mark.parametrize("scope", [None, "smoke"])
def test_python_fixtures_require_an_explicit_full_gate(scope, tmp_path):
    manifest = tmp_path/"oracle.json"
    manifest.write_text(json.dumps({"state": "passed", "scope": scope}))
    out = tmp_path/"unchecked"
    result = subprocess.run([sys.executable, str(ROOT/"scripts/check_lh_iocortex_python.py"),
                             "--device", "cpu", "--oracle-result", str(manifest), "--output-dir", str(out)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "full-scope" in result.stderr
    assert not out.exists()
