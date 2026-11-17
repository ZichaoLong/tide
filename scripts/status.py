"""Small current-work handoff; no imports from a model implementation."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
subprocess.run(["git", "status", "--short", "--branch"], cwd=root, check=True)
print((root / "docs/STATUS.md").read_text())

