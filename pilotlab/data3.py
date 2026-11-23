"""Read-only third-round views of the current pilot's article corpus.

The vocabulary and train stream retain their existing identity.  Holdout
documents are split by a fixed content-hash ordering; all copies of an article
stay together.  Confirmation data is unavailable to ordinary training workers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .io import atomic_json, sha256

SCHEMA = "tide-pilot-data-v3"
SPLIT_SALT = "tide-pilot-r3-selection-confirmation-v1"


def _asset(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("dataset asset escapes its declared directory")
    return path


def prepare(source: str | Path, output: str | Path) -> dict:
    """Create only split indices/metadata, verifying existing source bytes once."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("third-round data must not be written inside its source")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing nonempty third-round data output")
    original = json.loads((source / "manifest.json").read_text())
    if original.get("schema_version") != "tide-pilot-data-v1":
        raise ValueError("expected the current pilot dataset schema")
    assets = {}
    for name, declared in original["files"].items():
        path = _asset(source, name)
        observed = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        if observed != declared:
            raise ValueError(f"source data asset changed: {name}")
        assets[name] = observed
    records = {}
    for split in ("train", "validation"):
        filename = original["splits"][split]["document_ids"]
        records[split] = [json.loads(line) for line in _asset(source, filename).read_text().splitlines()]
    train_ids = {row["document_id"] for row in records["train"]}
    holdout_ids = {row["document_id"] for row in records["validation"]}
    if train_ids & holdout_ids:
        raise ValueError("train and holdout article identities overlap")
    tokenizer_ids = set(_asset(source, "tokenizer-training-document-ids.txt").read_text().splitlines())
    if not tokenizer_ids <= train_ids:
        raise ValueError("tokenizer training includes holdout articles")
    ordered = sorted(holdout_ids, key=lambda identity: hashlib.sha256(
        (SPLIT_SALT + "\0" + identity).encode()).digest())
    selection_ids = set(ordered[:len(ordered) // 2])
    confirmation_ids = set(ordered[len(ordered) // 2:])
    if not selection_ids or not confirmation_ids:
        raise ValueError("at least two unique holdout articles are required")
    output.mkdir(parents=True, exist_ok=True)
    split_specs = {"train": {"tokens_source": original["splits"]["train"]["tokens"],
                             "documents_source": original["splits"]["train"]["documents"],
                             "document_count": len(records["train"]),
                             "unique_document_count": len(train_ids),
                             "token_count": original["splits"]["train"]["token_count"]}}
    generated = {}
    for name, identities in (("selection", selection_ids), ("confirmation", confirmation_ids)):
        rows = [row for row in records["validation"] if row["document_id"] in identities]
        ranges = np.asarray([[row["start"], row["stop"]] for row in rows], dtype=np.int64)
        np.save(output / f"{name}.documents.npy", ranges, allow_pickle=False)
        with (output / f"{name}.documents.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        split_specs[name] = {"tokens_source": original["splits"]["validation"]["tokens"],
                             "documents": f"{name}.documents.npy",
                             "document_ids": f"{name}.documents.jsonl",
                             "document_count": len(rows), "unique_document_count": len(identities),
                             "token_count": int((ranges[:, 1] - ranges[:, 0]).sum())}
        for filename in (f"{name}.documents.npy", f"{name}.documents.jsonl"):
            path = output / filename
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            generated[filename] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {"schema_version": SCHEMA, "source_directory": str(source),
                "source_manifest_sha256": sha256(source / "manifest.json"),
                "source_dataset_id": original["dataset_id"], "source_files": assets,
                "vocab_size": original["vocab_size"], "token_dtype": original["token_dtype"],
                "tokenizer_source": original["tokenizer"]["file"], "files": generated,
                "splits": split_specs,
                "split_policy": {"salt": SPLIT_SALT, "method": "sorted salted content SHA256, equal unique-article halves",
                                 "duplicate_articles_stay_together": True},
                "holdout_label": "third-round selection/confirmation views of prior pilot holdout; not official test",
                "source_verification": "all declared source assets SHA256 verified at preparation"}
    manifest["dataset_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True,
                                                       separators=(",", ":")).encode()).hexdigest()
    atomic_json(output / "manifest.json", manifest)
    return manifest


class DocumentStore3:
    """Article-bounded sampler; read-only views retain absolute source offsets."""

    def __init__(self, root: str | Path, *, allow_confirmation: bool = False):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema_version") != SCHEMA:
            raise ValueError("unsupported third-round dataset schema")
        self.source = Path(self.manifest["source_directory"]).resolve()
        if sha256(self.source / "manifest.json") != self.manifest["source_manifest_sha256"]:
            raise ValueError("source data manifest changed")
        self.allow_confirmation = allow_confirmation
        self.vocab_size = int(self.manifest["vocab_size"])
        self.tokens, self.documents, self._eligible = {}, {}, {}
        for filename, record in self.manifest["files"].items():
            path = _asset(self.root, filename)
            if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
                raise ValueError(f"third-round data index changed: {filename}")
        for filename, record in self.manifest["source_files"].items():
            if _asset(self.source, filename).stat().st_size != record["size_bytes"]:
                raise ValueError(f"source asset size changed: {filename}")
        for name, spec in self.manifest["splits"].items():
            self.tokens[name] = np.memmap(_asset(self.source, spec["tokens_source"]), mode="r",
                                         dtype=self.manifest["token_dtype"])
            path = (_asset(self.source, spec["documents_source"]) if "documents_source" in spec
                    else _asset(self.root, spec["documents"]))
            ranges = np.load(path, mmap_mode="r", allow_pickle=False)
            if (ranges.ndim != 2 or ranges.shape[1] != 2 or not len(ranges)
                    or np.any(ranges[:, 0] < 0) or np.any(ranges[:, 1] <= ranges[:, 0])
                    or np.any(ranges[:, 1] > len(self.tokens[name]))
                    or np.any(ranges[1:, 0] < ranges[:-1, 1])):
                raise ValueError(f"invalid {name} document ranges")
            self.documents[name] = ranges

    def sample_batch(self, split: str, batch_size: int, sequence_length: int,
                     rng: np.random.Generator) -> np.ndarray:
        if split in ("val", "validation"):
            split = "selection"
        if split == "confirmation" and not self.allow_confirmation:
            raise PermissionError("confirmation split requires frozen-list authorization")
        if split not in self.documents:
            raise ValueError(f"unknown split: {split}")
        if type(batch_size) is not int or batch_size < 1 or type(sequence_length) is not int or sequence_length < 1:
            raise ValueError("batch_size and sequence_length must be positive integers")
        needed = sequence_length + 1
        key = split, needed
        if key not in self._eligible:
            ranges = self.documents[split]
            self._eligible[key] = np.flatnonzero(ranges[:, 1] - ranges[:, 0] >= needed)
        eligible = self._eligible[key]
        if not len(eligible):
            raise ValueError(f"no {split} document has {needed} tokens")
        result = np.empty((batch_size, needed), dtype=np.int64)
        for row in result:
            index = int(eligible[int(rng.integers(len(eligible)))])
            start, stop = map(int, self.documents[split][index])
            offset = start + int(rng.integers(stop - start - needed + 1))
            row[:] = self.tokens[split][offset:offset + needed]
        return result


DocumentStore = DocumentStore3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.source, args.output)
    print(json.dumps({"dataset_id": result["dataset_id"], "splits": result["splits"]}, indent=2))


if __name__ == "__main__":
    main()
