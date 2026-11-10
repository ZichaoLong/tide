"""Reproducible Git-source audit inventory; automatic extraction is not review.

Examples (does not import Torch or initialize a device)::

    python scripts/review_inventory.py --revision HEAD --output artifacts/audit.json
    python scripts/review_inventory.py --revision SHA --include 'cpp/include/tide/*' \
        --cpp clang --cpp-arg=-I/path/to/sdk/include --output artifacts/headers.json

Review claims are optional, hash-bound observations from a particular audit,
not another live backlog. Current work continues to belong in ROADMAP/STATUS.
"""
import argparse
from collections import Counter
from fnmatch import fnmatchcase
import json
from pathlib import Path
import subprocess
import tempfile

from durable_records import write_json
from review_symbols import clang_symbols, digest, python_symbols


SOURCE_SUFFIXES = {".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".cu", ".cuh", ".sh", ".cmake", ".inl", ".inc"}
CPP_SUFFIXES = {".cpp", ".cc", ".c", ".h", ".hpp", ".cu", ".cuh", ".inl", ".inc"}


def git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root)


def source_tree(root, revision):
    commit = git(root, "rev-parse", "--verify", revision+"^{commit}").decode().strip()
    entries = git(root, "ls-tree", "-r", "--full-tree", "-z", commit).split(b"\0")
    blobs, omitted = [], []
    for entry in entries:
        if not entry:
            continue
        info, raw_name = entry.split(b"\t", 1)
        mode, kind, oid = info.decode().split()
        name = raw_name.decode("utf-8", errors="surrogateescape")
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe Git path: "+name)
        if kind != "blob" or mode == "120000":
            omitted.append(dict(path=name, mode=mode, reason="nonregular Git entry"))
        else:
            blobs.append((name, oid))
    response = subprocess.check_output(["git", "cat-file", "--batch"], cwd=root,
        input=b"".join(oid.encode()+b"\n" for _, oid in blobs))
    values, offset = {}, 0
    for name, oid in blobs:
        end = response.index(b"\n", offset)
        actual, kind, size = response[offset:end].split()
        if actual.decode() != oid or kind != b"blob":
            raise ValueError("unexpected Git object response")
        begin, length = end+1, int(size)
        values[name] = response[begin:begin+length]
        offset = begin+length+1
    return commit, values, omitted


def responsibility(path):
    """Stable navigational grouping, not a claim about dependency correctness."""
    if path.startswith(("tests/", "cpp/test/")):
        return "verification"
    if path.startswith("examples/"):
        return "consumer-examples"
    if path.startswith("scripts/"):
        return "development-and-diagnostics"
    if path.startswith("tools/device_online/"):
        return "resident-backend"
    if path.startswith("tools/"):
        return "native-consumers-and-tools"
    if "/planning/" in path:
        return "configuration-and-planning"
    if "/experiment/" in path:
        return "experiment-runtime"
    if "/lm/" in path:
        return "language-model-consumer"
    name = Path(path).stem
    if "checkpoint" in name or "optimizer" in name:
        return "training-and-persistence"
    if "/weighted/" in path or name.startswith("weighted_"):
        return "weighted-local-programs"
    if name.startswith(("bind", "native")):
        return "bindings"
    if name.startswith(("reference", "stream", "frontier", "greedy", "component", "window", "settle", "block")):
        return "scheduling-and-graphs"
    if path.startswith(("python/", "cpp/")):
        return "core-runtime-and-local-programs"
    return "build-and-packaging"


def apply_reviews(files, claims):
    paths = {row["path"]: row for row in files}
    for claim in claims:
        path = claim["path"]
        if path not in paths:
            raise ValueError("review path is outside inventory: "+path)
        row = paths[path]
        if claim["sha256"] != row["sha256"]:
            raise ValueError("review source changed: "+path)
        if not claim.get("reviewer") or not claim.get("evidence") or not claim.get("scope"):
            raise ValueError("review requires reviewer, evidence and scope")
        selected = claim.get("symbols", [])
        available = {symbol["id"]: symbol for symbol in row["symbols"]}
        if selected == "*":
            selected = list(available)
            row["review"]["state"] = "reviewed"
        else:
            if not selected:
                raise ValueError("partial review requires exact symbol IDs")
            if row["review"]["state"] != "reviewed":
                row["review"]["state"] = "partially_reviewed"
        for identity in selected:
            if identity not in available:
                raise ValueError("review symbol is missing: "+identity)
            available[identity]["review"] = "reviewed"
        row["review"].setdefault("claims", []).append(claim)


def inventory(root, revision="HEAD", *, includes=(), cpp="skip", driver="clang++", arguments=(),
              timeout=15, reviews=None):
    commit, tree, omitted = source_tree(root, revision)
    selected = [name for name in sorted(tree) if (Path(name).suffix in SOURCE_SUFFIXES
                or Path(name).name == "CMakeLists.txt") and (not includes or any(fnmatchcase(name, p) for p in includes))]
    rows = []
    with tempfile.TemporaryDirectory(prefix="tide-review-") as temporary:
        if cpp == "clang":
            for name, raw in tree.items():
                target = Path(temporary)/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
        for name in selected:
            raw, suffix = tree[name], Path(name).suffix
            if suffix == ".py":
                symbols, parsing = python_symbols(raw, name)
            elif suffix in CPP_SUFFIXES and cpp == "clang":
                symbols, parsing = clang_symbols(raw, name, temporary, driver=driver,
                    arguments=arguments, timeout=timeout)
            else:
                symbols, parsing = [], dict(method="none", state="not_parsed",
                    reason="C++ parsing not requested" if suffix in CPP_SUFFIXES else "language has no configured symbol parser")
            for symbol in symbols:
                symbol["review"] = "pending"
            rows.append(dict(path=name, sha256=digest(raw), bytes=len(raw), lines=len(raw.splitlines()),
                responsibility=responsibility(name), parsing=parsing, symbols=symbols,
                review=dict(state="static_only" if parsing["state"] == "parsed" else "pending")))
    if reviews:
        if reviews.get("schema") != "tide-implementation-review-v1" or reviews.get("source") != commit:
            raise ValueError("review manifest schema/source mismatch")
        apply_reviews(rows, reviews["files"])
    counts = dict(files=len(rows), symbols=sum(len(r["symbols"]) for r in rows),
        definitions=sum(s["definition"] for r in rows for s in r["symbols"]),
        parsing=dict(Counter(r["parsing"]["state"] for r in rows)),
        file_review=dict(Counter(r["review"]["state"] for r in rows)),
        symbol_review=dict(Counter(s["review"] for r in rows for s in r["symbols"])),
        responsibility=dict(Counter(r["responsibility"] for r in rows)))
    manifest = {r["path"]: r["sha256"] for r in rows}
    return dict(schema="tide-implementation-inventory-v1", source=commit,
        source_scope="immutable Git objects; working-tree and untracked changes excluded",
        manifest_sha256=digest(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()),
        tool_sha256={p.name: digest(p.read_bytes()) for p in (Path(__file__), Path(__file__).with_name("review_symbols.py"))},
        requested=dict(includes=list(includes), cpp=cpp, driver=driver, arguments=list(arguments),
                       timeout_seconds=timeout), summary=counts, omitted_nonregular_entries=omitted,
        qualification="inventory and explicit review scope only; no runtime, numerical or performance qualification", files=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--cpp", choices=("skip", "clang"), default="skip")
    parser.add_argument("--cpp-driver", default="clang++")
    parser.add_argument("--cpp-arg", action="append", default=[])
    parser.add_argument("--cpp-timeout", type=float, default=15)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cpp_timeout <= 0:
        parser.error("--cpp-timeout must be positive")
    if args.output.exists():
        parser.error("--output already exists; choose a new audit record")
    reviews = None if args.reviews is None else json.loads(args.reviews.read_text())
    result = inventory(args.root, args.revision, includes=args.include, cpp=args.cpp,
        driver=args.cpp_driver, arguments=args.cpp_arg, timeout=args.cpp_timeout, reviews=reviews)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(source=result["source"], manifest_sha256=result["manifest_sha256"], **result["summary"]), sort_keys=True))


if __name__ == "__main__":
    main()
