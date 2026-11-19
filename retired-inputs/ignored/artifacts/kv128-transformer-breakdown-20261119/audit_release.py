"""Terminal/frozen-input audit of the four explicitly named breakdown workloads."""
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.job import atomic
spec = importlib.util.spec_from_file_location("prior_release_helpers", ROOT/"artifacts/memory-candidates-20261119/release_audit.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def main():
    result = dict(observed=datetime.now(timezone.utc).isoformat(), script_sha256=m.digest(__file__),
                  helper_sha256=m.digest(ROOT/"artifacts/memory-candidates-20261119/release_audit.py"),
                  jobs=[], raw_manifest=[], failures=[])
    expected = {}
    directories = []
    for mode in ("infer", "train"):
        for subject in ("kv128", "tf32"):
            path = ROOT/f"artifacts/breakdown-{mode}-{subject}-v1"
            directories.append(path)
            launch = json.loads((path/"launch.json").read_text())
            status = json.loads((path/"status.json").read_text())
            data = json.loads((path/"breakdown.json").read_text())
            unit = m.run(["systemctl", "--user", "show", launch["unit"], "-p", "MainPID", "-p", "ControlGroup", "-p", "ActiveState", "-p", "SubState", "-p", "Result", "-p", "Slice"])
            props = dict(line.split("=", 1) for line in unit["stdout"].splitlines() if "=" in line)
            launch_props = dict(line.split("=", 1) for line in launch["unit_observation"].splitlines() if "=" in line)
            ids = {(x["pid"], x["start_ticks"]): x for x in m.identities(status)}
            live = [x for x in ids.values() if m.alive(x)]
            groups = {group: m.cgroup_pids(group) for group in (props.get("ControlGroup"), launch_props.get("ControlGroup")) if group}
            terminal = props.get("MainPID") == "0" and not live and not any(groups.values()) and props.get("ActiveState") not in ("active", "activating", "deactivating")
            commit = launch["source"]
            if commit not in expected:
                expected[commit] = m.expected_files(commit)
            changed = [name for name, sha in expected[commit].items() if not (Path(launch["source_path"])/name).is_file() or m.digest(Path(launch["source_path"])/name) != sha]
            row = dict(name=path.name, source=commit, unit=launch["unit"], properties=props, lifecycle=status["state"], result=data["state"],
                       terminal=terminal, recorded_identities=len(ids), live_identities=live, cgroup_processes=groups, frozen_regular_files=len(expected[commit]), frozen_changes=changed)
            result["jobs"].append(row)
            if not terminal or changed or status["state"] != "passed" or data["state"] != "completed_finite":
                result["failures"].append(row)
            for p in sorted(path.iterdir()):
                if p.is_file() and p.suffix in (".json", ".log", ".sh"):
                    result["raw_manifest"].append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=m.digest(p)))
    protected = json.loads((HERE/"preflight.json").read_text())["protected"]
    result["protected"] = {name: dict(original=sha, current=m.digest(ROOT/name)) for name, sha in protected.items()}
    if any(x["original"] != x["current"] for x in result["protected"].values()):
        result["failures"].append("protected files changed")
    result["checks"] = [m.run(cmd) for cmd in (["git", "diff", "--check"], ["python", "scripts/check_upstream.py"], ["quota", "-s"])]
    usage = m.run(["du", "-s", "-B1", *[str(p) for p in directories], str(HERE)])
    size = sum(int(line.split()[0]) for line in usage["stdout"].splitlines())
    result["artifact_usage"] = dict(command=usage, allocated_bytes=size, allocated_gib=size/(1<<30), limit_gib=8)
    if size > 8*(1<<30) or usage["exit_code"] or any(x["exit_code"] for x in result["checks"]):
        result["failures"].append("checks or artifact budget failed")
    result["state"] = "passed" if not result["failures"] else "failed"
    atomic(HERE/"release-audit.json", result)
    print(json.dumps(dict(state=result["state"], jobs=len(result["jobs"]), files=len(result["raw_manifest"]), artifact_gib=result["artifact_usage"]["allocated_gib"], failures=result["failures"])))
    return int(result["state"] != "passed")

if __name__ == "__main__":
    raise SystemExit(main())
