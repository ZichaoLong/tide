"""Read-only split views, duplicate isolation and confirmation access control."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pilotlab.data3 import DocumentStore3, prepare
from pilotlab.io import sha256


def write_source(path: Path):
    path.mkdir()
    splits = {}
    for name, count, base in (("train", 8, 0), ("validation", 10, 100)):
        rows = []
        arrays = []
        for index in range(count):
            # Two copies of the first holdout document test identity grouping.
            identity_index = base if name == "validation" and index == count - 1 else base + index
            arrays.append(((np.arange(512, dtype=np.uint16) * 7 + identity_index) % 31).astype("<u2"))
            rows.append({"document_id": hashlib.sha256(str(identity_index).encode()).hexdigest(),
                         "start": index * 512, "stop": (index + 1) * 512})
        np.concatenate(arrays).tofile(path / f"{name}.tokens.bin")
        np.save(path / f"{name}.documents.npy", np.array([[r["start"], r["stop"]] for r in rows], dtype=np.int64))
        (path / f"{name}.documents.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        splits[name] = {"tokens": f"{name}.tokens.bin", "documents": f"{name}.documents.npy",
                        "document_ids": f"{name}.documents.jsonl", "token_count": count * 512}
    (path / "tokenizer.json").write_text("{}\n")
    (path / "tokenizer-training-document-ids.txt").write_text(hashlib.sha256(b"0").hexdigest() + "\n")
    files = {file.name: {"size_bytes": file.stat().st_size, "sha256": sha256(file)}
             for file in path.iterdir() if file.is_file()}
    manifest = {"schema_version": "tide-pilot-data-v1", "files": files, "splits": splits,
                "vocab_size": 32, "token_dtype": "<u2", "dataset_id": "third-round-fixture",
                "tokenizer": {"file": "tokenizer.json"}}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def test_prepare_preserves_source_and_groups_duplicates(tmp_path):
    source = write_source(tmp_path / "source")
    original = {file.name: sha256(file) for file in source.iterdir()}
    output = tmp_path / "view"
    manifest = prepare(source, output)
    selected = [json.loads(line)["document_id"] for line in (output / "selection.documents.jsonl").read_text().splitlines()]
    confirmed = [json.loads(line)["document_id"] for line in (output / "confirmation.documents.jsonl").read_text().splitlines()]
    assert not set(selected) & set(confirmed)
    assert len(selected) + len(confirmed) == 10
    duplicate = hashlib.sha256(b"100").hexdigest()
    assert (selected.count(duplicate), confirmed.count(duplicate)) in ((2, 0), (0, 2))
    assert abs(len(set(selected)) - len(set(confirmed))) == 1
    assert {file.name: sha256(file) for file in source.iterdir()} == original
    assert not list(output.glob("*.tokens.bin"))
    assert manifest["source_manifest_sha256"] == sha256(source / "manifest.json")
    with pytest.raises(FileExistsError):
        prepare(source, output)
    with pytest.raises(ValueError, match="inside its source"):
        prepare(source, source / "new")


def test_sampler_is_reproducible_and_confirmation_is_closed(tmp_path):
    output = tmp_path / "view"
    prepare(write_source(tmp_path / "source"), output)
    store = DocumentStore3(output)
    for split in ("train", "selection"):
        first = store.sample_batch(split, 12, 64, np.random.default_rng(7))
        second = store.sample_batch(split, 12, 64, np.random.default_rng(7))
        assert np.array_equal(first, second)
        assert first.shape == (12, 65)
        assert np.all(np.diff(first, axis=1) % 31 == 7)
    with pytest.raises(PermissionError, match="frozen-list authorization"):
        store.sample_batch("confirmation", 1, 64, np.random.default_rng(0))
    opened = DocumentStore3(output, allow_confirmation=True)
    assert opened.sample_batch("confirmation", 1, 64, np.random.default_rng(0)).shape == (1, 65)
    with pytest.raises(ValueError, match="no selection document"):
        store.sample_batch("selection", 1, 512, np.random.default_rng(0))


def test_changed_source_or_index_is_rejected(tmp_path):
    source, output = write_source(tmp_path / "source"), tmp_path / "view"
    prepare(source, output)
    (output / "selection.documents.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="index changed"):
        DocumentStore3(output)
    with (source / "train.tokens.bin").open("ab") as stream:
        stream.write(b"\x00\x00")
    with pytest.raises(ValueError, match="source data asset changed"):
        prepare(source, tmp_path / "other")
