#!/usr/bin/env python3
"""Resolve a foundation old/new commit prefix without fetching the old branch."""
import argparse
import json
from pathlib import Path
import subprocess
from history_map import MANIFEST, ROOT, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    try:
        result = resolve(args.commit, args.repo, args.manifest)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"history resolution failed: {error}\n")
    print(json.dumps(result) if args.json else result["commit"])


if __name__ == "__main__":
    main()
