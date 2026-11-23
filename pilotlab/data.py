"""Immutable WikiText pilot data and document-bounded next-token sampling.

The public source contains upstream *training* shards only.  ``validation`` in
this module is a deterministic article holdout, not the official WikiText split.
No source file is modified, and tokenizer training sees train articles only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Iterator

import numpy as np


SCHEMA = "tide-pilot-data-v1"
DEFAULT_SPLIT_SALT = "tide-pilot-document-split-v1"
SPECIAL_TOKENS = ("[UNK]", "[PAD]", "[BOS]", "[EOS]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_article_title(line: str) -> bool:
    """Recognize level-one WikiText titles, including spaced equals markers.

    Section headings such as ``== History ==`` and ``= = History = =`` stay
    within the article.  A row can contain more than one physical text line.
    """
    line = line.strip()
    if not (line.startswith("=") and line.endswith("=")):
        return False
    middle = line[1:-1].strip()
    return bool(middle) and not middle.startswith("=") and not middle.endswith("=")


def iter_articles(rows: Iterable[str | None]) -> Iterator[str]:
    """Join successive source rows into complete articles across shard edges."""
    current: list[str] = []
    nonempty = False
    for row in rows:
        if row is None:
            continue
        # Empty rows still separate paragraphs; preserve their newline.
        lines = row.splitlines(keepends=True) if row else ["\n"]
        for line in lines:
            if is_article_title(line) and nonempty:
                yield "".join(current)
                current, nonempty = [], False
            if line.strip() or nonempty:
                current.append(line if line.endswith(("\n", "\r")) else line + "\n")
                nonempty = nonempty or bool(line.strip())
    if nonempty:
        yield "".join(current)


def article_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def article_split(document_id: str, *, holdout_fraction: float = 0.02,
                  salt: str = DEFAULT_SPLIT_SALT) -> str:
    """Content identity keeps exact duplicate articles on the same side."""
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be strictly between zero and one")
    value = int.from_bytes(hashlib.sha256((salt + "\0" + document_id).encode()).digest()[:8], "big")
    return "validation" if value < int(holdout_fraction * (1 << 64)) else "train"


def source_files(source: Path) -> tuple[dict, list[Path]]:
    """Verify only the immutable manifest's declared Parquet shards."""
    manifest = json.loads((source / "manifest.json").read_text())
    files: list[Path] = []
    for record in manifest["files"]:
        if not record["path"].endswith(".parquet"):
            continue
        path = (source / record["path"]).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError("manifest path escapes source directory")
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"source identity mismatch: {record['path']}")
        files.append(path)
    if not files:
        raise ValueError("source manifest declares no Parquet shards")
    return manifest, files


def parquet_rows(files: Iterable[Path]) -> Iterator[str | None]:
    import pyarrow.parquet as pq

    for path in files:
        source = pq.ParquetFile(path)
        if "text" not in source.schema_arrow.names:
            raise ValueError(f"missing text column: {path}")
        for batch in source.iter_batches(batch_size=512, columns=["text"]):
            yield from batch.column(0).to_pylist()


def atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare(source: Path, output: Path, *, vocab_size: int = 8192,
            tokenizer_train_bytes: int = 16 * 1024 * 1024,
            holdout_fraction: float = 0.02,
            split_salt: str = DEFAULT_SPLIT_SALT) -> dict:
    if vocab_size < 260:
        raise ValueError("byte-level BPE needs at least 256 bytes and four special tokens")
    if tokenizer_train_bytes < 1:
        raise ValueError("tokenizer_train_bytes must be positive")
    article_split("validation", holdout_fraction=holdout_fraction, salt=split_salt)
    source = source.resolve()
    output = output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("generated data must not be written in the public source")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    upstream, files = source_files(source)

    # Bound native thread pools unless the launcher supplies a different budget.
    os.environ.setdefault("RAYON_NUM_THREADS", "8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=list(SPECIAL_TOKENS),
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    training = {"document_count": 0, "utf8_bytes": 0,
                "target_utf8_bytes": tokenizer_train_bytes,
                "cap_policy": "first train articles until target reached; final article remains whole",
                "split": "train"}
    training_ids: list[str] = []

    def training_text() -> Iterator[str]:
        for text in iter_articles(parquet_rows(files)):
            identity = article_id(text)
            if article_split(identity, holdout_fraction=holdout_fraction, salt=split_salt) != "train":
                continue
            training["document_count"] += 1
            training["utf8_bytes"] += len(text.encode("utf-8"))
            training_ids.append(identity)
            yield text
            if training["utf8_bytes"] >= tokenizer_train_bytes:
                break

    tokenizer.train_from_iterator(training_text(), trainer=trainer)
    if not training["document_count"]:
        raise ValueError("no training articles available for tokenizer")
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    (output / "tokenizer-training-document-ids.txt").write_text("".join(x + "\n" for x in training_ids))
    effective_vocab = tokenizer.get_vocab_size()
    dtype = np.dtype("<u2" if effective_vocab <= 65536 else "<u4")
    special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    eos = special_ids["[EOS]"]
    offsets = {name: [] for name in ("train", "validation")}
    counts = {name: Counter() for name in offsets}
    identities = {name: set() for name in offsets}
    token_files = {name: (output / f"{name}.tokens.bin").open("wb") for name in offsets}
    id_files = {name: (output / f"{name}.documents.jsonl").open("w") for name in offsets}

    def write_batch(batch: list[str], start_index: int) -> None:
        for source_index, text, encoded in zip(range(start_index, start_index + len(batch)),
                                               batch, tokenizer.encode_batch(batch, add_special_tokens=False)):
            identity = article_id(text)
            split = article_split(identity, holdout_fraction=holdout_fraction, salt=split_salt)
            ids = np.asarray(encoded.ids + [eos], dtype=dtype)
            start = counts[split]["token_count"]
            stop = start + len(ids)
            token_files[split].write(ids.tobytes())
            offsets[split].append((start, stop))
            counts[split].update(document_count=1, token_count=len(ids), utf8_bytes=len(text.encode("utf-8")))
            identities[split].add(identity)
            record = {"document_id": identity, "source_article_index": source_index,
                      "start": start, "stop": stop,
                      "title": text.splitlines()[0].strip() if text else ""}
            id_files[split].write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        batch, batch_bytes, start_index = [], 0, 0
        for text in iter_articles(parquet_rows(files)):
            batch.append(text)
            batch_bytes += len(text.encode("utf-8"))
            if len(batch) >= 64 or batch_bytes >= 8 * 1024 * 1024:
                write_batch(batch, start_index)
                start_index += len(batch)
                batch, batch_bytes = [], 0
        if batch:
            write_batch(batch, start_index)
    finally:
        for stream in (*token_files.values(), *id_files.values()):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
    if identities["train"] & identities["validation"]:
        raise AssertionError("document identity crosses split")
    if set(training_ids) - identities["train"]:
        raise AssertionError("tokenizer used documents outside train split")

    split_metadata = {}
    for split, ranges in offsets.items():
        if not ranges:
            raise ValueError(f"empty {split} split; select a different fixed split policy")
        array = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
        np.save(output / f"{split}.documents.npy", array, allow_pickle=False)
        lengths = array[:, 1] - array[:, 0]
        split_metadata[split] = dict(counts[split]) | {
            "tokens": f"{split}.tokens.bin", "documents": f"{split}.documents.npy",
            "document_ids": f"{split}.documents.jsonl", "unique_document_count": len(identities[split]),
            "duplicate_document_count": len(ranges) - len(identities[split]),
            "document_tokens_min": int(lengths.min()), "document_tokens_max": int(lengths.max()),
            "eligible_documents": {str(n): int((lengths >= n + 1).sum()) for n in (64, 128, 256, 512, 1024, 2048, 4096)},
        }
    generated = {}
    for path in sorted(output.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            generated[path.name] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema_version": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source), "source_repository": upstream.get("repository"),
        "source_revision": upstream.get("revision"), "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "source_files": upstream["files"], "licenses": upstream.get("licenses", []),
        "holdout_label": "pilot article holdout from upstream train; not official validation or test",
        "split_policy": {"unit": "complete level-one-title article", "hash": "sha256 of utf8 article content; salted hash threshold",
                         "holdout_fraction": holdout_fraction, "salt": split_salt,
                         "exact_duplicate_articles_stay_together": True,
                         "preamble_policy": "nonempty text preceding first title is one document"},
        "tokenizer": {"file": "tokenizer.json", "kind": "byte-level BPE", "requested_vocab_size": vocab_size,
                      "vocab_size": effective_vocab, "special_token_ids": special_ids,
                      "training": training, "eos_policy": "one EOS appended per complete document; no BOS prepended"},
        "vocab_size": effective_vocab, "token_dtype": dtype.str, "splits": split_metadata,
        "files": generated,
    }
    manifest["dataset_id"] = hashlib.sha256(json.dumps({"files": generated, "split_policy": manifest["split_policy"],
                                                       "source_revision": manifest["source_revision"]},
                                                      sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(output / "manifest.json", manifest)
    return manifest


class DocumentStore:
    """Read-only memory-mapped token arrays with deterministic safe sampling."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema_version"] != SCHEMA:
            raise ValueError("unsupported pilot dataset schema")
        self.vocab_size = int(self.manifest["vocab_size"])
        self.tokens = {}
        self.documents = {}
        self._eligible = {}
        for name, info in self.manifest["splits"].items():
            self.tokens[name] = np.memmap(self.root / info["tokens"], mode="r", dtype=self.manifest["token_dtype"])
            self.documents[name] = np.load(self.root / info["documents"], mmap_mode="r", allow_pickle=False)
            if self.documents[name].ndim != 2 or self.documents[name].shape[1] != 2:
                raise ValueError("document ranges must have shape [N,2]")
            ranges = self.documents[name]
            if len(ranges) == 0 or np.any(ranges[:, 1] <= ranges[:, 0]) or ranges[0, 0] != 0:
                raise ValueError("invalid document ranges")
            if np.any(ranges[1:, 0] != ranges[:-1, 1]) or ranges[-1, 1] != len(self.tokens[name]):
                raise ValueError("document ranges do not partition token array")

    def sample_batch(self, split: str, batch_size: int, sequence_length: int,
                     rng: np.random.Generator) -> np.ndarray:
        if split == "val":
            split = "validation"
        if split not in self.tokens:
            raise ValueError(f"unknown split: {split}")
        if type(batch_size) is not int or batch_size < 1 or type(sequence_length) is not int or sequence_length < 1:
            raise ValueError("batch_size and sequence_length must be positive integers")
        needed = sequence_length + 1
        key = (split, needed)
        if key not in self._eligible:
            ranges = self.documents[split]
            self._eligible[key] = np.flatnonzero(ranges[:, 1] - ranges[:, 0] >= needed)
        eligible = self._eligible[key]
        if not len(eligible):
            raise ValueError(f"no {split} document has {needed} tokens")
        result = np.empty((batch_size, needed), dtype=np.int64)
        for row in result:
            doc = int(eligible[int(rng.integers(len(eligible)))])
            start, stop = (int(x) for x in self.documents[split][doc])
            offset = start + int(rng.integers(stop - start - needed + 1))
            row[:] = self.tokens[split][offset:offset + needed]
        return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--vocab-size", type=int, default=8192)
    command.add_argument("--tokenizer-train-bytes", type=int, default=16 * 1024 * 1024)
    command.add_argument("--holdout-fraction", type=float, default=0.02)
    command.add_argument("--split-salt", default=DEFAULT_SPLIT_SALT)
    args = parser.parse_args(argv)
    result = prepare(args.source, args.output, vocab_size=args.vocab_size,
                     tokenizer_train_bytes=args.tokenizer_train_bytes,
                     holdout_fraction=args.holdout_fraction, split_salt=args.split_salt)
    print(json.dumps({"dataset_id": result["dataset_id"], "vocab_size": result["vocab_size"],
                      "splits": result["splits"], "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
