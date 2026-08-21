"""Deterministic FineWeb-Edu preparation and fixed-length token loading."""

from __future__ import annotations

import array
import hashlib
import json
import os
import pathlib
import shutil
import sys
from collections.abc import Iterator, Sequence
from typing import Any


FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


class UInt32TokenWriter:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.handle = path.open("wb")
        self.hasher = hashlib.sha256()
        self.buffer = array.array("I")
        self.count = 0

    def append(self, token: int) -> None:
        if token < 0 or token > 0xFFFFFFFF:
            raise ValueError(f"token id is outside uint32: {token}")
        self.buffer.append(token)
        self.count += 1
        if len(self.buffer) >= 262_144:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        if sys.byteorder != "little":
            self.buffer.byteswap()
        payload = self.buffer.tobytes()
        if sys.byteorder != "little":
            self.buffer.byteswap()
        self.handle.write(payload)
        self.hasher.update(payload)
        self.buffer = array.array("I")

    def close(self) -> dict[str, Any]:
        self.flush()
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        return {
            "path": self.path.name,
            "tokens": self.count,
            "bytes": self.count * 4,
            "sha256": self.hasher.hexdigest(),
        }


def prepare_fineweb(
    *,
    model_path: str,
    output_dir: str,
    validation_tokens: int,
    train_tokens: int,
    sequence_length: int,
) -> pathlib.Path:
    """Stream, tokenize, and atomically publish the pinned v0 token split."""

    if validation_tokens < sequence_length or train_tokens < sequence_length:
        raise ValueError("each split must contain at least one sequence")
    final_dir = pathlib.Path(output_dir).resolve()
    if final_dir.exists():
        raise FileExistsError(f"output directory already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = final_dir.with_name(f".{final_dir.name}.partial-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir()

    validation_writer = UInt32TokenWriter(staging / "validation.bin")
    train_writer = UInt32TokenWriter(staging / "train.bin")
    writers = [
        (validation_writer, validation_tokens),
        (train_writer, train_tokens),
    ]
    document_count = 0
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.eos_token_id is None:
            raise ValueError("the tokenizer has no EOS token")
        dataset = load_dataset(
            FINEWEB_DATASET,
            name=FINEWEB_CONFIG,
            split="train",
            revision=FINEWEB_REVISION,
            streaming=True,
        )
        writer_index = 0
        next_report = 250_000
        total_target = validation_tokens + train_tokens
        total_written = 0
        for document in dataset:
            text = document.get("text")
            if not isinstance(text, str) or not text:
                continue
            document_count += 1
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_ids.append(tokenizer.eos_token_id)
            for token_id in token_ids:
                while writer_index < len(writers) and writers[writer_index][0].count >= writers[writer_index][1]:
                    writer_index += 1
                if writer_index >= len(writers):
                    break
                writer, target = writers[writer_index]
                if writer.count < target:
                    writer.append(int(token_id))
                    total_written += 1
                    if total_written >= next_report:
                        print(
                            f"prepared {total_written:,}/{total_target:,} tokens "
                            f"from {document_count:,} documents",
                            flush=True,
                        )
                        next_report += 250_000
            if writer_index >= len(writers):
                break

        if validation_writer.count != validation_tokens or train_writer.count != train_tokens:
            raise RuntimeError(
                "dataset stream ended early: "
                f"validation={validation_writer.count}/{validation_tokens}, "
                f"train={train_writer.count}/{train_tokens}"
            )
        validation_info = validation_writer.close()
        train_info = train_writer.close()
        manifest = {
            "schema_version": 1,
            "dataset": {
                "name": FINEWEB_DATASET,
                "config": FINEWEB_CONFIG,
                "revision": FINEWEB_REVISION,
                "split": "train",
                "documents_consumed": document_count,
            },
            "tokenizer": {
                "model_path_hint": model_path,
                "class": type(tokenizer).__name__,
                "vocab_size": len(tokenizer),
                "eos_token_id": tokenizer.eos_token_id,
            },
            "packing": {
                "document_separator": "eos",
                "sequence_length": sequence_length,
                "dtype": "little-endian uint32",
                "validation_complete_sequences": validation_tokens // sequence_length,
                "train_complete_sequences": train_tokens // sequence_length,
            },
            "files": {
                "validation": validation_info,
                "train": train_info,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, final_dir)
    except Exception:
        for writer, _ in writers:
            if not writer.handle.closed:
                writer.handle.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"published token data at {final_dir}", flush=True)
    return final_dir


class TokenSequenceDataset:
    def __init__(self, path: str | pathlib.Path, sequence_length: int) -> None:
        import numpy as np

        self.path = pathlib.Path(path)
        self.sequence_length = sequence_length
        self.tokens = np.memmap(self.path, mode="r", dtype="<u4")
        self.sequence_count = len(self.tokens) // sequence_length
        if self.sequence_count == 0:
            raise ValueError(f"token file has no complete sequence: {self.path}")

    def __len__(self) -> int:
        return self.sequence_count

    def __getitem__(self, index: int):
        import numpy as np
        import torch

        if index < 0 or index >= self.sequence_count:
            raise IndexError(index)
        start = index * self.sequence_length
        values = np.array(self.tokens[start : start + self.sequence_length], dtype=np.int64)
        return torch.from_numpy(values)


def deterministic_batches(
    dataset: TokenSequenceDataset,
    *,
    batch_size: int,
    seed: int,
    consumed_sequences: int = 0,
) -> Iterator[tuple[Any, int]]:
    """Yield batches and the updated sequence cursor, reproducibly across resume."""

    import torch

    if batch_size < 1 or consumed_sequences < 0:
        raise ValueError("batch size must be positive and cursor nonnegative")
    cursor = consumed_sequences
    while True:
        epoch = cursor // len(dataset)
        offset = cursor % len(dataset)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        while offset < len(order):
            indices = order[offset : offset + batch_size]
            if len(indices) < batch_size:
                cursor += len(indices)
                break
            batch = torch.stack([dataset[index] for index in indices])
            cursor += batch_size
            offset += batch_size
            yield batch, cursor


def load_data_manifest(data_dir: str | pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(data_dir) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))
