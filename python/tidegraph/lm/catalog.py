"""Pinned public model identities; loading this module never imports Torch."""
import json
from pathlib import Path, PurePosixPath


def catalog():
    return json.loads(Path(__file__).with_name("catalog.json").read_text())["models"]


def model(key):
    values = catalog()
    if key not in values:
        raise ValueError(f"model: unknown {key!r}; choose {', '.join(values)}")
    return values[key]


def local_path(root, key):
    return Path(root).resolve()/f"{key}-{model(key)['revision']}"


def safe_relative(name):
    value = PurePosixPath(name)
    if value.is_absolute() or not value.parts or ".." in value.parts or "\\" in name:
        raise ValueError(f"invalid asset path: {name!r}")
    return Path(*value.parts)
