"""Explicit optional native dependency loading; never modifies sys.path or builds."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import torch


def load_native(location=None):
    if location is None:
        try:
            return importlib.import_module("_tide_native")
        except ImportError as error:
            raise RuntimeError("native implementation unavailable; build the matching adapter and pass native_library=BUILD_DIR") from error
    path = Path(location).expanduser().resolve()
    if path.is_dir():
        matches = list(path.glob("_tide_native*.so"))
        if len(matches) != 1:
            raise ValueError("native_library must contain exactly one _tide_native shared library")
        path = matches[0]
    if not path.is_file():
        raise ValueError("native library does not exist")
    manifest_path = path.parent / "build-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("explicit native library requires adjacent build-manifest.json")
    record = json.loads(manifest_path.read_text())
    if (record.get("architecture") != platform.machine() or record.get("torch") != torch.__version__
            or record.get("cxx11_abi") != torch.compiled_with_cxx11_abi()
            or record.get("python", "").split(".")[:2] != platform.python_version().split(".")[:2]):
        raise ValueError("native build does not match this host/Python/Torch/ABI")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if record.get("binary_sha256", {}).get(path.name) != digest:
        raise ValueError("native library content does not match its build manifest")
    loaded = sys.modules.get("_tide_native")
    if loaded is not None:
        if Path(loaded.__file__).resolve() != path:
            raise RuntimeError("a different native library is already loaded; use a fresh process")
        return loaded
    spec = importlib.util.spec_from_file_location("_tide_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["_tide_native"] = module
    return module
