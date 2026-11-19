"""Check the immutable mathematical snapshot without importing Torch."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lock = json.loads((root / "docs/upstream.json").read_text())
for name, expected in lock["files"].items():
    actual = hashlib.sha256((root / lock["snapshot"] / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"snapshot mismatch: {name}")
print(json.dumps({"state": "passed", "revision": lock["revision"], "files": len(lock["files"])}))

