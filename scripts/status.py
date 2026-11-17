"""Small current-work handoff; no imports from a model implementation."""
from pathlib import Path
import json
import subprocess

root = Path(__file__).resolve().parents[1]
subprocess.run(["git", "status", "--short", "--branch"], cwd=root, check=True)
print((root / "docs/STATUS.md").read_text())
for path in sorted((root/"artifacts/instances").glob("*/status.json")):
    record=json.loads(path.read_text())
    print(json.dumps({"record":str(path.relative_to(root)),
                      **{key:record.get(key) for key in ("state","source","elapsed","exit_code")}}))

