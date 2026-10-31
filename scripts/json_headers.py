"""Resolve and fingerprint an explicit external nlohmann JSON header dependency."""
import hashlib
from pathlib import Path
import re


def resolve(explicit=None):
    if explicit is not None:
        path, provider = Path(explicit).resolve(), "explicit include directory"
    else:
        try:
            import nlohmann_json
        except ImportError:
            path, provider = Path("/usr/include"), "system include directory"
        else:
            path, provider = Path(nlohmann_json.get_include()).resolve(), "private Python header package"
    if not (path / "nlohmann/json.hpp").is_file():
        raise ValueError("nlohmann/json.hpp missing: pass --json-include or install nlohmann_json==3.11.2 in the project environment")
    digest = hashlib.sha256()
    files = sorted((path / "nlohmann").rglob("*.hpp"))
    for file in files:
        digest.update(str(file.relative_to(path)).encode() + b"\0" + file.read_bytes() + b"\0")
    versions = {}
    for name in ("json.hpp", "detail/abi_macros.hpp"):
        file = path / "nlohmann" / name
        if file.is_file():
            versions.update(re.findall(r"#define\s+NLOHMANN_JSON_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)", file.read_text()))
    version = ".".join(versions.get(part, "unknown") for part in ("MAJOR", "MINOR", "PATCH"))
    return path, dict(provider=provider, version=version, header_count=len(files), sha256=digest.hexdigest())
