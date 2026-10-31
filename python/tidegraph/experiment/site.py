"""Site-specific paths and a deliberate, credential-free launcher environment."""
from copy import deepcopy
import os
from pathlib import Path
from .records import read_json


ENVIRONMENT_KEYS = frozenset({
    "PATH", "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG",
    "TORCH_DEVICE_BACKEND_AUTOLOAD", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_CUSTOM_OPP_PATH",
    "ASCEND_LAUNCH_BLOCKING", "TASK_QUEUE_ENABLE", "HCCL_CONNECT_TIMEOUT",
    "HCCL_EXEC_TIMEOUT", "PYTORCH_NPU_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
    "TIDE_COMMUNICATION_POLICY",
})
BUILD_PATH_KEYS = frozenset({"native_library", "resident_library", "standalone_binary"})
# CANN can require a module-owned Python runtime path. It must be explicit in
# the site; never inherit an ambient source checkout into an installed job.
EXPLICIT_ENVIRONMENT_KEYS = ENVIRONMENT_KEYS | {"PYTHONPATH"}


def fields(value, allowed, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected a JSON object")
    extra = set(value)-set(allowed)
    if extra:
        raise ValueError(f"{where}.{sorted(extra)[0]}: unknown field; allowed: {', '.join(sorted(allowed))}")
    return value


def positive(value, where, minimum=1):
    if type(value) is not int or not minimum <= value < 2**63:
        raise ValueError(f"{where}: expected an integer in [{minimum}, 2**63)")
    return value


def load(value=None, *, base_dir=None):
    base_dir = Path(base_dir or Path.cwd())
    if isinstance(value, (str, os.PathLike)):
        path = (base_dir/Path(value)).resolve()
        value, base_dir = read_json(path), path.parent
    value = deepcopy(value or {})
    fields(value, {"schema", "name", "description", "launcher", "paths", "environment", "standalone_environment", "threads",
                   "resources", "requirements", "notes"}, "site")
    if value.get("schema", "tide-site-v1") != "tide-site-v1":
        raise ValueError("site.schema: expected tide-site-v1")
    launcher = value.get("launcher", "foreground")
    if launcher not in {"foreground", "systemd", "container"}:
        raise ValueError("site.launcher: choose foreground, systemd or container")
    paths = fields(value.get("paths", {}), BUILD_PATH_KEYS, "site.paths")
    for name, path in paths.items():
        if not isinstance(path, str) or not path:
            raise ValueError(f"site.paths.{name}: expected a nonempty path")
        paths[name] = str((base_dir/Path(path)).resolve())
    environments = {}
    for group in ("environment", "standalone_environment"):
        environments[group] = fields(value.get(group, {}), EXPLICIT_ENVIRONMENT_KEYS, "site."+group)
        for key, val in environments[group].items():
            if not isinstance(val, str) or "\x00" in val or "\n" in val:
                raise ValueError(f"site.{group}.{key}: expected a single-line string")
    threads = fields(value.get("threads", {}), {"workers", "intraop_threads", "interop_threads"}, "site.threads")
    for key, val in threads.items():
        positive(val, f"site.threads.{key}")
    for key in ("resources", "requirements"):
        if not isinstance(value.get(key, {}), dict):
            raise ValueError(f"site.{key}: expected an object")
    return dict(schema="tide-site-v1", name=value.get("name", "explicit-local"),
                launcher=launcher, paths=paths, **environments, threads=threads,
                resources=value.get("resources", {}), requirements=value.get("requirements", {}),
                description=value.get("description", ""), notes=value.get("notes", []))


def launch_environment(site, execution, *, environ=None):
    """Copy only reviewed runtime variables, never the full interactive shell."""
    current = os.environ if environ is None else environ
    result = {name: current[name] for name in ENVIRONMENT_KEYS if name in current}
    result.update(site.get("environment", {}))
    result.setdefault("PATH", os.defpath)
    result["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    result["TIDE_COMMUNICATION_POLICY"] = execution["effective"].get("communication", "ordinary")
    count = str(execution["effective"]["intraop_threads"])
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        result[key] = count
    return result
