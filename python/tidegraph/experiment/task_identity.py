"""Task code identity without importing Torch or executing a plugin factory."""
import importlib.util
from copy import deepcopy
from pathlib import Path
from .records import digest, fingerprint


def identity(task):
    plugin = task.get("plugin")
    if plugin is None:
        return dict(kind="built-in", objective=task["objective"], optimizer=task["optimizer"])
    package = plugin["module"].split(".")[0]
    spec = importlib.util.find_spec(package)
    if spec is None:
        raise ValueError(f"task.plugin.module: install the downstream package containing {package}")
    roots = list(spec.submodule_search_locations or ())
    if roots:
        if len(roots) != 1:
            raise ValueError("task.plugin.module: use one installed package root for immutable task identity")
        root = Path(roots[0])
        files = {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    elif spec.origin and Path(spec.origin).is_file():
        files = {Path(spec.origin).name: digest(spec.origin)}
    else:
        raise ValueError("task.plugin.module: task code must be readable installed files")
    if not files:
        raise ValueError("task.plugin.module: no readable task package files")
    return dict(kind="downstream", **deepcopy(plugin), package=package,
                package_files=files, package_source_sha256=fingerprint(files))


def verify(task, expected):
    if identity(task) != expected:
        raise ValueError("task.plugin: installed task code/configuration differs; restore the fixed task package or start a new run")
