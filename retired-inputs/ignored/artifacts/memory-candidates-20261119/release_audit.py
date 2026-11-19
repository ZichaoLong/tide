"""Read-only release verification for this finite memory-candidate campaign."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.job import atomic


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(command):
    p = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return dict(argv=command, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)


def alive(identity):
    try:
        fields = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19]) == identity["start_ticks"] and fields[0] != "Z"
    except FileNotFoundError:
        return False


def identities(value):
    if isinstance(value, dict):
        if type(value.get("pid")) is int and type(value.get("start_ticks")) is int:
            yield value
        for item in value.values():
            yield from identities(item)
    elif isinstance(value, list):
        for item in value:
            yield from identities(item)


def cgroup_pids(group):
    # Both the published legacy hierarchy and cgroup v2 are supported here.
    found = set()
    if not group:
        return []
    for prefix in (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/systemd")):
        base = prefix/group.lstrip("/")
        if base.exists():
            for p in base.rglob("cgroup.procs"):
                found.update(int(x) for x in p.read_text().split())
    return sorted(found)


def expected_files(commit):
    archive = subprocess.check_output(["git", "archive", commit], cwd=ROOT)
    out = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for item in stream:
            if item.isfile():
                out[item.name] = hashlib.sha256(stream.extractfile(item).read()).hexdigest()
    return out


def main():
    here = ROOT/"artifacts/memory-candidates-20261119"
    record = dict(schema="tide-memory-release-audit-v1", observed=datetime.now(timezone.utc).isoformat(),
                  script_sha256=digest(__file__), jobs=[], failures=[], raw_manifest=[])
    expected = {}
    for launch_path in sorted((ROOT/"artifacts").glob("memory-*/launch.json")):
        # The final audit's own terminal state is checked by the caller.
        if launch_path.parent.name.startswith("memory-release-audit"):
            continue
        launch = json.loads(launch_path.read_text())
        status_path = launch_path.parent/"status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        observed = run(["systemctl", "--user", "show", launch["unit"], "-p", "MainPID", "-p", "ControlGroup",
                        "-p", "ActiveState", "-p", "SubState", "-p", "Result", "-p", "ExecMainStatus", "-p", "Slice"])
        properties = dict(line.split("=", 1) for line in observed["stdout"].splitlines() if "=" in line)
        descendants = {(x["pid"], x["start_ticks"]): x for x in identities(status)}
        for name in ("results/suite.json", "results/programme.json"):
            p = launch_path.parent/name
            if p.exists():
                descendants.update({(x["pid"], x["start_ticks"]): x for x in identities(json.loads(p.read_text()))})
        live = [x for x in descendants.values() if alive(x)]
        group_pids = cgroup_pids(properties.get("ControlGroup", ""))
        terminal = properties.get("MainPID", "0") == "0" and not live and not group_pids and properties.get("ActiveState") not in ("active", "activating", "deactivating")
        commit = launch["source"]
        if commit not in expected:
            expected[commit] = expected_files(commit)
        mismatches = []
        for name, sha in expected[commit].items():
            path = Path(launch["source_path"])/name
            if not path.is_file() or digest(path) != sha:
                mismatches.append(name)
        job = dict(name=launch_path.parent.name, source=commit, lifecycle_state=status.get("state"),
                   exit_code=status.get("exit_code"), unit=launch["unit"], unit_properties=properties,
                   all_processes_terminal=terminal, recorded_identities=len(descendants), live_identities=live,
                   cgroup_pids=group_pids, frozen_regular_files=len(expected[commit]), frozen_mismatches=mismatches)
        record["jobs"].append(job)
        if not terminal or mismatches:
            record["failures"].append(dict(job=job["name"], terminal=terminal, frozen_mismatches=mismatches))
        # Large checkpoints, wheels and environments retain their dedicated
        # qualification identities. This manifest fingerprints raw observations.
        for p in sorted(launch_path.parent.rglob("*")):
            rel = p.relative_to(launch_path.parent)
            if any(part in {"source", "venv", "wheels", "__pycache__", "build"} for part in rel.parts):
                continue
            if p.is_file() and p.suffix in {".json", ".log", ".sh"}:
                record["raw_manifest"].append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=digest(p)))
    protected = json.loads((here/"preflight.json").read_text())["protected"]
    record["protected_original"] = protected
    record["protected_current"] = {path: digest(ROOT/path) for path in protected}
    record["protected_unchanged"] = all(record["protected_current"][path] == (value if isinstance(value, str) else value["sha256"]) for path, value in protected.items())
    record["checks"] = [run(command) for command in (["git", "status", "--short", "--branch"],
        ["git", "diff", "--check"], ["python", "scripts/check_upstream.py"], ["quota", "-s"])]
    if not record["protected_unchanged"] or any(c["exit_code"] for c in record["checks"]):
        record["failures"].append(dict(protected_unchanged=record["protected_unchanged"], checks=record["checks"]))
    record["state"] = "passed" if not record["failures"] else "failed"
    atomic(here/"release-audit.json", record)
    print(json.dumps(dict(state=record["state"], jobs=len(record["jobs"]), raw_files=len(record["raw_manifest"]), failures=record["failures"])))
    return 0 if record["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
