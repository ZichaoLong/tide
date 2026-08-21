"""Small, sanitized run manifests for TIDE experiments."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def sha256_file(path: str | pathlib.Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def repository_identity(root: str | pathlib.Path = ".") -> dict[str, Any]:
    root = pathlib.Path(root)

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=normal")
        return {"commit": commit, "dirty": bool(status), "status_lines": status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status_lines": []}


def model_identity(model_path: str | pathlib.Path) -> dict[str, Any]:
    root = pathlib.Path(model_path)
    candidates = sorted(root.glob("*.safetensors")) + sorted(root.glob("pytorch_model*.bin"))
    files = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in candidates
    ]
    config = root / "config.json"
    return {
        "path_hint": str(root),
        "weight_files": files,
        "config_sha256": sha256_file(config) if config.is_file() else None,
    }


def base_manifest(*, runtime: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    visible = {
        name: name in os.environ
        for name in ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")
    }
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository_identity(),
        "runtime": runtime.to_dict(),
        "host": {
            "architecture": platform.machine(),
            "os": platform.platform(),
            "python": sys.version.replace("\n", " "),
        },
        "visibility_environment_set": visible,
        "arguments": arguments,
    }


def atomic_write_json(path: str | pathlib.Path, payload: dict[str, Any]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
