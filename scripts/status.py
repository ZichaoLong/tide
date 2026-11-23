"""Read this checkout's handoff and the observations supplied by its owner."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def git(*arguments):
    return subprocess.check_output(
        ["git", "--no-optional-locks", *arguments], cwd=ROOT, text=True,
        errors="replace", stderr=subprocess.PIPE,
    ).rstrip("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print a tide-worktree-status-v1 record")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
    from status_runtime import REPOSITORY, observe

    started = datetime.now(timezone.utc).isoformat()
    handoff_path = "docs/STATUS.md"
    try:
        handoff = (ROOT / handoff_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        handoff = f"[{handoff_path}] {error}"
    runtime = observe(ROOT)
    record = {
        "schema": "tide-worktree-status-v1",
        "repository": REPOSITORY,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "revision": git("rev-parse", "HEAD"),
        "changes": git("status", "--short", "--untracked-files=normal").splitlines(),
        "sample_started_at": started,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "current_task": next(iter(handoff.splitlines()), "").removeprefix("# "),
        "handoff": {"path": handoff_path, "text": handoff},
        "runtime": runtime,
    }
    if args.json:
        print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(f"## {record['branch']} @ {record['revision'][:12]}")
        for change in record["changes"]:
            print(change)
        print(f"\n{record['observed_at']} [{runtime['kind']}] {runtime['summary']}")
        for item in runtime.get("records", []):
            print(json.dumps(item, ensure_ascii=False))
        print("Evidence:", ", ".join(runtime["sources"]))
        print("Next read-only check:", runtime["next_check"])
        print(f"\n[{handoff_path} · recorded handoff]\n{handoff.rstrip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
