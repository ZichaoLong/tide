"""Installed content and explicit build identities, independent of a checkout."""
from pathlib import Path
import hashlib
import importlib.metadata
import platform
import sys
import zipfile
from .records import digest, fingerprint, read_json


def package_identity(root=None):
    import tidegraph
    root = Path(root) if root is not None else Path(tidegraph.__file__).resolve().parent
    files = {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}}
    if not files:
        raise ValueError("installed package contains no readable files")
    return dict(package="tidegraph", version=tidegraph.__version__, path=str(root.resolve()),
                installed_files_sha256=fingerprint(files), files_sha256=files,
                source_provenance="installation content; Git/wheel provenance requires an explicit dependency lock")


def build_identity(location, *, kind="native"):
    path = Path(location).resolve()
    root = path if path.is_dir() else path.parent
    filename = {"native": "build-manifest.json", "resident": "control-build.json",
                "standalone": "result.json"}[kind]
    if kind == "standalone" and not (root/filename).is_file():
        root = root.parent
    manifest = root/filename
    record = read_json(manifest)
    if kind == "standalone":
        if record.get("state") != "passed" or not path.is_file():
            raise ValueError("site.paths.standalone_binary: require a completed build and its executable")
        expected, observed = record.get("binary_sha256"), digest(path)
        if expected != observed:
            raise ValueError("site.paths.standalone_binary: executable differs from build manifest")
        binaries = {path.name: observed}
    else:
        expected = record.get("binary_sha256", {})
        if not isinstance(expected, dict) or not expected:
            raise ValueError(f"site.paths.{kind}_library: missing binary identities")
        binaries = {}
        for name, value in expected.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("build manifest contains an invalid binary path")
            actual = digest(root/relative)
            if actual != value:
                raise ValueError(f"site.paths.{kind}_library: binary changed: {name}")
            binaries[name] = actual
    return dict(kind=kind, path=str(path), source=record.get("source"),
                manifest_sha256=digest(manifest), binary_sha256=binaries,
                backend=record.get("backend", record.get("core", {}).get("backend")),
                architecture=record.get("architecture", record.get("core", {}).get("architecture")),
                cpp_source_sha256=record.get("cpp_source_sha256", record.get("core", {}).get("cpp_source_sha256")),
                component_sha256=record.get("component_sha256"), client_source_sha256=record.get("sources"),
                loader_sha256=record.get("loader_sha256"),
                dependency_scope="build manifest provenance; external loader libraries require the selected site lock")


def capture(paths=None):
    result = dict(schema="tide-installed-identity-v1", package=package_identity(),
                  python=platform.python_version(), executable=sys.executable,
                  architecture=platform.machine(), dependencies=dependencies(), builds={})
    for key, kind in (("native_library", "native"), ("resident_library", "resident"),
                      ("standalone_binary", "standalone")):
        if (paths or {}).get(key):
            result["builds"][key] = build_identity(paths[key], kind=kind)
    return result


def dependencies():
    """Installed distribution metadata identity, not a full dependency byte audit."""
    result = {}
    for name in ("torch", "torch-npu", "numpy"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
            continue
        record = distribution.read_text("RECORD")
        result[name] = dict(version=distribution.version,
            record_sha256=None if record is None else hashlib.sha256(record.encode()).hexdigest(),
            scope="installed version and RECORD metadata; dependency binaries are not rehashed")
    return result


def verify(expected, paths=None):
    actual = capture(paths)
    if expected["package"]["installed_files_sha256"] != actual["package"]["installed_files_sha256"]:
        raise ValueError("installed tidegraph content changed after submission; freeze the installation before running")
    if (expected["python"], expected["architecture"]) != (actual["python"], actual["architecture"]):
        raise ValueError("Python/architecture changed after submission")
    if expected.get("dependencies", {}) != actual["dependencies"]:
        raise ValueError("installed dependency metadata changed after submission; adopt a new explicit lock")
    # Locations are hints; the exact manifests/binaries determine identity.
    strip = lambda rows: {k: {n: v for n, v in row.items() if n != "path"} for k, row in rows.items()}
    if strip(expected.get("builds", {})) != strip(actual["builds"]):
        raise ValueError("native/resident/standalone build changed after submission")
    return actual


def create_lock(*, site=None, wheel=None):
    from .site import load
    selected = load(site)
    installed = capture(selected["paths"])
    artifact = None
    if wheel is not None:
        path = Path(wheel).resolve()
        with zipfile.ZipFile(path) as archive:
            files = {name[len("tidegraph/"):]: hashlib.sha256(archive.read(name)).hexdigest()
                     for name in archive.namelist() if name.startswith("tidegraph/") and not name.endswith("/")}
        if not files or files != installed["package"]["files_sha256"]:
            raise ValueError("wheel tidegraph files differ from the installed package; install this exact wheel first")
        artifact = dict(path=str(path), sha256=digest(path), installed_files_verified=True)
    body = dict(schema="tide-experiment-lock-v1", installed_identity=installed,
                build_paths=selected["paths"], wheel=artifact,
                source="supply a reviewed source commit/build record with the wheel; no Git provenance inferred from cwd")
    return dict(body, lock_id=fingerprint(body))


def verify_lock(value, *, site=None):
    from .site import load
    lock = read_json(value) if isinstance(value, (str, Path)) else value
    if lock.get("schema") != "tide-experiment-lock-v1" or lock.get("lock_id") != fingerprint({k: v for k, v in lock.items() if k != "lock_id"}):
        raise ValueError("lock schema/content identity mismatch")
    paths = load(site)["paths"] if site is not None else lock["build_paths"]
    verify(lock["installed_identity"], paths)
    return dict(state="passed", lock_id=lock["lock_id"], scope="installed package/build content and recorded dependency metadata",
                qualification="dependency identity only; run base and downstream checks separately")
