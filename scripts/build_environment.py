"""Reproducible native build identity, separate from compiler orchestration."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def cache_values(target):
    path = target / "CMakeCache.txt"
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            name, value = line.split("=", 1)
            result[name.split(":", 1)[0]] = value
    return result


def check_reuse(target, backend, npu_runtime, torch=None, *, torch_dir=None, torch_version=None):
    values = cache_values(target)
    if torch is not None:
        torch_dir = str(Path(torch.__file__).parent / 'share/cmake/Torch')
        torch_version = torch.__version__
    expected = {"TIDE_BACKEND":backend.upper(), "Torch_DIR":torch_dir}
    if backend == "npu":
        expected["TIDE_NPU_RUNTIME"] = npu_runtime
        if npu_runtime == 'standalone' and os.environ.get('Torch_npu_DIR'):
            expected['Torch_npu_DIR'] = os.environ['Torch_npu_DIR']
    for key, value in expected.items():
        if key in values and values[key] != value:
            raise ValueError(f"{key} differs from existing build; use a separate build directory")
    manifest = target / "build-manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get('torch') != torch_version:
            raise ValueError("Torch distribution differs from existing build; use a separate build directory")
        if backend == 'npu' and previous.get('environment', {}).get('cann_version') != os.environ.get('ASCEND_CANN_VERSION'):
            raise ValueError('CANN differs from existing build; use a separate build directory')


def environment(target):
    cache = cache_values(target)
    compiler = cache["CMAKE_CXX_COMPILER"]
    record = dict(os=platform.platform(), compiler=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
                  cxx_standard=17, build_type=cache["CMAKE_BUILD_TYPE"], torch_prefix=cache["Torch_DIR"],
                  cann_version=os.environ.get("ASCEND_CANN_VERSION"))
    if cache.get("TIDE_BACKEND") == "NPU" and cache.get("TIDE_NPU_RUNTIME") == "standalone":
        config = Path(cache["Torch_npu_DIR"]) / "Torch_npuConfig.cmake"
        # Official SDK layout: share/cmake/Torch_npu plus lib/libtorch_npu.so.
        library = config.parents[3] / "lib/libtorch_npu.so"
        if not library.is_file():
            raise ValueError("standalone SDK library is missing from its declared prefix")
        record["npu_sdk"] = dict(config=str(config), config_sha256=digest(config),
                                 library_sha256=digest(library))
    return record
