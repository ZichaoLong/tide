"""Read-only release audit; writes only this cycle's audit and job index."""
import hashlib
import io
import json
import re
import subprocess
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from scripts.job import atomic


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True)
    return {"argv": argv, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


base = Path("artifacts/agg-inference-preflight-20261118")
preflight = json.loads((base / "preflight.json").read_text())
record = {"schema": "tide-reception-release-audit-1",
          "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
          "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "production_tree": subprocess.check_output(["git", "rev-parse", "HEAD:src/tide"], text=True).strip(),
          "upstream": command(["python", "scripts/check_upstream.py"]),
          "quota": command(["quota", "-s"]),
          "git_status": command(["git", "status", "--short", "--branch"]),
          "diff_check": command(["git", "diff", "--check"]),
          "preserved_user_files": [], "jobs": [], "links": {},
          "scope": "Frozen-source bytes and terminal lifecycle do not replace each inner numerical outcome."}
for original in preflight["user_files"]:
    actual = identity(original["path"])
    record["preserved_user_files"].append({"original": original, "actual": actual,
        "unchanged": all(actual[k] == original[k] for k in ("bytes", "sha256"))})

archives, jobs = {}, []
for launch_path in sorted(Path("artifacts").glob("agg-*/launch.json")):
    directory = launch_path.parent
    launch = json.loads(launch_path.read_text())
    status_path = directory / "status.json"
    status = json.loads(status_path.read_text())
    unit = launch["unit"].removesuffix(".service") + ".service"
    observation = command(["systemctl", "--user", "show", unit,
        "--property=LoadState,ActiveState,SubState,MainPID,Result,ControlGroup,Slice"])
    properties = dict(line.split("=", 1) for line in observation["stdout"].splitlines() if "=" in line)
    live = []
    for process in status.get("descendant_identities", []):
        path = Path(f"/proc/{process['pid']}/stat")
        try:
            tail = path.read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        if int(tail[19]) == process["start_ticks"] and tail[0] not in ("Z", "X"):
            live.append(dict(process, state=tail[0]))
    source = launch["source"]
    if source not in archives:
        data = subprocess.check_output(["git", "archive", "--format=tar", source])
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archives[source] = {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                                for member in archive if member.isfile()}
    mismatches, missing = [], []
    for name, expected in archives[source].items():
        path = Path(launch["source_path"]) / name
        if not path.exists():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append({"path": name, "expected_sha256": expected,
                               "actual_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    item = {"name": directory.name, "source": source, "launch": identity(launch_path),
        "status": identity(status_path), "outer_state": status["state"],
        "exit_code": status.get("exit_code"), "unit": unit, "unit_properties": properties,
        "live_recorded_descendants": live, "frozen_regular_files": len(archives[source]),
        "source_missing": missing, "source_mismatches": mismatches,
        "terminal": status["state"] not in ("running", "queued", "starting") and
                    properties.get("MainPID") == "0" and not live and
                    properties.get("ActiveState") in ("inactive", "failed")}
    for filename in ("queue.json", "environment.json"):
        if (directory / filename).exists():
            item[filename.removesuffix(".json")] = identity(directory / filename)
    record["jobs"].append(item)
    jobs.append({"name": directory.name, "source": source, "unit": unit,
                 "arguments": launch["arguments"], "state": status["state"],
                 "status": str(status_path.resolve())})

documents = sorted(set(Path(p) for p in subprocess.check_output(
    ["git", "diff", "--name-only", "df2a0f18", "--", "*.md"], text=True).splitlines()))
documents += [Path("docs/evidence/reception-training-20261118.md"),
              Path("docs/evidence/reception-inference-20261118.md")]
broken, checked = [], 0
for path in sorted(set(documents)):
    if not path.exists():
        broken.append({"source": str(path), "reason": "document missing"})
        continue
    for target in re.findall(r"\[[^\]]+\]\(([^)\n]+)\)", path.read_text()):
        target = target.strip().strip("<>").split("#", 1)[0]
        if not target or re.match(r"^[a-zA-Z]+://", target) or target.startswith("mailto:"):
            continue
        checked += 1
        if not (path.parent / unquote(target)).exists():
            broken.append({"source": str(path), "target": target})
record["links"] = {"documents": len(set(documents)), "checked": checked, "broken": broken,
                   "scope": "local target existence; fragment anchors not checked"}
record["summary"] = {"outer_states": dict(Counter(x["outer_state"] for x in record["jobs"])),
    "all_terminal": all(x["terminal"] for x in record["jobs"]),
    "all_frozen_regular_files_match": all(not x["source_missing"] and not x["source_mismatches"] for x in record["jobs"]),
    "all_user_files_unchanged": all(x["unchanged"] for x in record["preserved_user_files"]),
    "local_links_valid": not broken}
atomic(base / "jobs.json", jobs)
atomic(base / "release-audit.json", record)
print(json.dumps(record["summary"], ensure_ascii=False))
