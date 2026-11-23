"""Behavioral checks for the new pilot's article split and sampler."""
import json

import numpy as np
import pytest

from pilotlab.data import DocumentStore, SCHEMA, article_id, article_split, is_article_title, iter_articles, prepare, sha256_file


def test_article_boundaries_preserve_sections_and_shard_continuation():
    rows = ["\n", " = First article = \n", "First paragraph.\n", " = = History = = \n",
            "Continued across an imaginary shard edge.\n", "== Nested ==\n", "More.\n",
            " = Second article = \nSecond body.\n", ""]
    articles = list(iter_articles(rows))
    assert len(articles) == 2
    assert "History" in articles[0] and "Nested" in articles[0]
    assert "imaginary shard edge" in articles[0]
    assert "Second body" in articles[1]
    assert "Second article" not in articles[0]
    assert is_article_title("= A =")
    assert not is_article_title("= = History = =")
    assert not is_article_title("== History ==")


def test_content_split_is_deterministic_disjoint_and_duplicate_safe():
    documents = [f"= Article {n} =\nDistinct body {n}.\n" for n in range(200)]
    split_ids = {"train": set(), "validation": set()}
    for text in documents + documents[:10]:
        identity = article_id(text)
        split = article_split(identity, holdout_fraction=0.2, salt="test")
        assert split == article_split(identity, holdout_fraction=0.2, salt="test")
        split_ids[split].add(identity)
    assert split_ids["train"] and split_ids["validation"]
    assert not split_ids["train"] & split_ids["validation"]
    assert len(split_ids["train"] | split_ids["validation"]) == len(documents)


def write_store(tmp_path):
    splits = {}
    for name, documents in {"train": [np.arange(10, 20), np.arange(100, 115), np.arange(200, 203)],
                            "validation": [np.arange(300, 312)]}.items():
        tokens = np.concatenate(documents).astype("<u2")
        tokens.tofile(tmp_path / f"{name}.tokens.bin")
        stops = np.cumsum([len(x) for x in documents])
        ranges = np.column_stack((np.concatenate(([0], stops[:-1])), stops)).astype(np.int64)
        np.save(tmp_path / f"{name}.documents.npy", ranges)
        splits[name] = {"tokens": f"{name}.tokens.bin", "documents": f"{name}.documents.npy"}
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": SCHEMA,
        "vocab_size": 512, "token_dtype": "<u2", "splits": splits}))
    return DocumentStore(tmp_path)


def test_sampler_is_deterministic_contiguous_and_document_bounded(tmp_path):
    store = write_store(tmp_path)
    a = store.sample_batch("train", 128, 7, np.random.default_rng(17))
    b = store.sample_batch("train", 128, 7, np.random.default_rng(17))
    assert a.dtype == np.int64 and a.shape == (128, 8)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a, axis=1) == 1)
    assert all((10 <= row[0] and row[-1] < 20) or (100 <= row[0] and row[-1] < 115) for row in a)
    assert np.all(store.sample_batch("val", 8, 10, np.random.default_rng(18)) >= 300)
    assert store.vocab_size == 512


def test_sampler_rejects_unsupported_length_and_bad_ranges(tmp_path):
    store = write_store(tmp_path)
    with pytest.raises(ValueError, match="no train document"):
        store.sample_batch("train", 2, 15, np.random.default_rng(0))
    with pytest.raises(ValueError, match="positive integers"):
        store.sample_batch("train", 0, 4, np.random.default_rng(0))
    np.save(tmp_path / "train.documents.npy", np.array([[0, 10], [11, 25], [25, 28]]))
    with pytest.raises(ValueError, match="do not partition"):
        DocumentStore(tmp_path)


def test_prepare_tokenizer_train_documents_are_disjoint_from_holdout(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("tokenizers")
    source, output = tmp_path / "source", tmp_path / "prepared"
    source.mkdir()
    rows = []
    for i in range(80):
        rows.extend([f" = Article {i} = \n", f"Body {i} has some repeated words and useful text. " * 8 + "\n",
                     " = = Section = = \n", f"More article {i} body content.\n"])
    shard = source / "train.parquet"
    pq.write_table(pa.table({"text": rows}), shard)
    (source / "manifest.json").write_text(json.dumps({"repository": "fixture", "revision": "fixed",
        "files": [{"path": shard.name, "size_bytes": shard.stat().st_size, "sha256": sha256_file(shard)}]}))
    manifest = prepare(source, output, vocab_size=320, tokenizer_train_bytes=1024,
                       holdout_fraction=0.25, split_salt="integration-test")
    train_ids = {json.loads(line)["document_id"] for line in (output / "train.documents.jsonl").read_text().splitlines()}
    val_ids = {json.loads(line)["document_id"] for line in (output / "validation.documents.jsonl").read_text().splitlines()}
    tokenizer_ids = set((output / "tokenizer-training-document-ids.txt").read_text().splitlines())
    assert not train_ids & val_ids
    assert tokenizer_ids <= train_ids and tokenizer_ids
    assert manifest["splits"]["train"]["document_count"] + manifest["splits"]["validation"]["document_count"] == 80
    store = DocumentStore(output)
    assert store.sample_batch("validation", 2, 16, np.random.default_rng(1)).shape == (2, 17)
    with pytest.raises(FileExistsError, match="nonempty output"):
        prepare(source, output)
