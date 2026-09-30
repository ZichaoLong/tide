"""Explicit optional CANN plugin loading, separate from standalone NPU SDKs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def load_resident(location, core):
    if location is None:
        raise ValueError("resident placement requires resident_library=DEVICE_BUILD_DIR")
    build = Path(location).expanduser().resolve()
    record = json.loads((build / "control-build.json").read_text())
    if record.get("npu_runtime") != "python":
        raise ValueError("Python resident execution requires a Python-owned backend build")
    core_path = Path(core.__file__).resolve()
    expected = record["core"]["binary_sha256"].get(core_path.name)
    if hashlib.sha256(core_path.read_bytes()).hexdigest() != expected:
        raise ValueError("resident backend requires the exact native core recorded by its build")
    for name in ("_tide_resident.so", "libtide-resident.so"):
        if hashlib.sha256((build / name).read_bytes()).hexdigest() != record["binary_sha256"].get(name):
            raise ValueError("resident binary differs from its build manifest")
    path = build / "_tide_resident.so"
    loaded = sys.modules.get("_tide_resident")
    if loaded is not None:
        if Path(loaded.__file__).resolve() != path:
            raise RuntimeError("a different resident backend is loaded; use a fresh process")
        return loaded, record
    spec = importlib.util.spec_from_file_location("_tide_resident", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["_tide_resident"] = module
    return module, record
