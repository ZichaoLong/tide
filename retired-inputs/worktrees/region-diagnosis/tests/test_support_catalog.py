"""The large catalog is a metadata check, never a blanket device qualification."""
import json
from pathlib import Path
import subprocess
import sys


def test_fixed_support_catalog_is_complete_and_keeps_qualification_boundaries(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = tmp_path/"catalog.json"
    result = subprocess.run([sys.executable, "-B", "-m", "scripts.support_catalog", "--output", str(path)],
                            cwd=root, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    record = json.loads(path.read_text())
    main = record["main_menu"]
    assert [main[name] for name in ("spatial_profiles", "bound_profiles", "geometry_resolutions",
                                   "parameter_configs", "allocation_rows")] == [41, 55, 475, 225400, 676200]
    assert main["by_variant"] == {"fc-settle": 9, "lattice-settle": 6, "exchange-settle": 12,
                                  "multiscale-timed": 14, "multiscale-settle": 14}
    assert record["qualification"]["metadata"]["state"] == "passed"
    assert record["qualification"]["metadata"]["torch_imported"] is False
    assert all(record["qualification"][key]["state"] == "not_run_by_catalog"
               for key in ("execution", "numerical", "performance"))
    assert all(not row["catalog_member"] for row in record["optional_bindings"])
    assert len(record["implementation"]["source_sha256"]) == 64
