#!/usr/bin/env python3
"""Check Tide document links, local formulas, and the canonical boundary.

Run from any directory with Python 3.10 or later. This checks repository
structure and notation; mathematical claims still require their proofs.
Canonical hashes record the reviewed edition. An intentional canonical edit
requires a reviewed diff, a revision record, and an explicit hash update.
"""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(ROOT / 'tools'))
from check_artifacts import ArtifactRegistry
from document_links import anchors, prose_lines, references, resolve

CANONICAL_HASHES = {
    "settlegraph-learning-note.md":
        "631bb877a197c2d5179b0fa17d4a74d2a294ed9e492278f4f040bb0b1925e362",
    "timed-dag-region-selector-learning-note.md":
        "5a9157fb8f76219d1aaa91a1dc9ed1c1bfc1902f4966e5e03ddee0abf2b24866",
    "positive-delay-graph-finite-cut-learning-note.md":
        "042782c97a4e1f7ba357d3aca097196b00bb9bf25c816f02ed4c8e35238a6be2",
    "timed-dag-chunk-prefill-learning-note.md":
        "5fc0deef0980ca4ea640fb165e2974e023a0754fca23bbbd051153cc9c2280b2",
}
REMOVED_ENTRIES = {
    "semantics-anchor",
    "weighted-kv-sparse-activation-example",
    "weighted-kv-layered-example",
}
# Historical records keep the names used in their respective editions.
RETIRED_CONFIG_NAMES = re.compile(
    r"\b(?:budget_family|tau_soft)\b"
    r"|\bselection_mode\s*=\s*(?:soft|direct|attention|pooled|dense|hard)\b"
)


def check_formulas(path, errors):
    """Check local numbered equations and TeX grouping, not their semantics."""
    text = "\n".join(line for _, line in prose_lines(path.read_text()))
    tags = [int(value) for value in re.findall(r"\\tag\{(\d+)\}", text)]
    if tags != list(range(1, len(tags) + 1)):
        errors.append(f"{path.relative_to(ROOT)}: equation tags are not consecutive")
    for number, line in prose_lines(text):
        for value in re.findall(r"式\s*\((\d+)\)", line):
            if int(value) not in tags:
                errors.append(f"{path.relative_to(ROOT)}:{number}: missing equation {value}")
    for match in re.finditer(r"\$\$[\s\S]*?\$\$|\$(?!\$)[^\n$]*\$", text):
        formula = match[0]
        level = 0
        for brace in re.finditer(r"(?<!\\)[{}]", formula):
            level += 1 if brace[0] == "{" else -1
            if level < 0:
                break
        if level != 0:
            number = text.count("\n", 0, match.start()) + 1
            errors.append(f"{path.relative_to(ROOT)}:{number}: unbalanced TeX braces")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-built', action='store_true',
                        help='Also require every registered artifact after a complete build.')
    args = parser.parse_args()
    try:
        registry = ArtifactRegistry(ROOT)
        errors, pending = registry.audit(args.require_built)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f'artifact manifest: {exc}', file=sys.stderr)
        return 1
    # Include new documents before staging, while respecting the repository's ignores.
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.md"],
        cwd=ROOT,
    ).decode().split("\0")
    # Generated notes are ignored by Git but are still checked when available.
    names += [name for name in registry.outputs if name.endswith('.md')]
    paths = sorted({(ROOT / name).resolve() for name in names if name
                    and (ROOT / name).is_file()
                    and not any(part in {".playwright-cli", "appendices"}
                                for part in Path(name).parts)})
    by_name = defaultdict(list)
    for path in paths:
        by_name[path.stem].append(path)
    canonical = {BASE / name for name in CANONICAL_HASHES}
    for name, expected in CANONICAL_HASHES.items():
        path = BASE / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"{path.relative_to(ROOT)}: canonical edition changed; review the diff")
    for name in REMOVED_ENTRIES:
        if (BASE / f"{name}.md").exists():
            errors.append(f"{name}.md: obsolete redirect entry has returned")

    links = 0
    inbound = 0
    for source in paths:
        text = source.read_text()
        inside = source.is_relative_to(BASE)
        if inside and any(ord(char) < 32 and char not in "\n\t\r" for char in text):
            errors.append(f"{source.relative_to(ROOT)}: unexpected control character")
        if inside and source not in canonical and not source.is_relative_to(BASE / "records"):
            for number, line in enumerate(text.splitlines(), 1):
                for match in RETIRED_CONFIG_NAMES.finditer(line):
                    errors.append(
                        f"{source.relative_to(ROOT)}:{number}: retired configuration name: {match[0]}"
                    )
        for number, raw, wiki in references(text):
            target = unquote(raw.strip().strip("<>"))
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("//"):
                continue
            path, separator, anchor = target.partition("#")
            destination = resolve(source, path, wiki, by_name, registry)
            concerns_tide = (inside or BASE.name in path or Path(path).stem in REMOVED_ENTRIES
                             or destination is not None and destination.is_relative_to(BASE))
            if not concerns_tide:
                continue
            links += 1
            inbound += not inside
            location = f"{source.relative_to(ROOT)}:{number}"
            if destination is None:
                errors.append(f"{location}: missing target {raw}")
            elif (separator and destination.suffix == ".md" and destination.is_file()
                  and anchor not in anchors(destination)):
                errors.append(f"{location}: missing anchor {raw}")
            if source in canonical and destination not in canonical:
                errors.append(f"{location}: canonical textbook refers to downstream material: {raw}")

    local_notes = sorted((BASE / "local-semantics" / "weighted-message").rglob("*.md"))
    for path in local_notes:
        check_formulas(path, errors)
    for path in BASE.rglob("*"):
        if path.is_dir() and not any(path.iterdir()) and not registry.recognizes(path):
            errors.append(f"{path.relative_to(ROOT)}: empty directory")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    documents = sum(path.is_relative_to(BASE) for path in paths)
    print("PASS: four canonical editions match; canonical local references stay within the four textbooks")
    print(f"PASS: {links} local Markdown/wiki links ({inbound} incoming) across {documents} Tide documents")
    print(f"PASS: numbered equations and TeX grouping in {len(local_notes)} local textbooks; no empty directories")
    print("PASS: no retired sending configuration names in current noncanonical documents")
    print(f'{"PENDING" if pending else "PASS"}: {len(pending)} registered outputs await local build')
    return 0


if __name__ == "__main__":
    sys.exit(main())
