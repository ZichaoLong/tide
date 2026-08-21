from __future__ import annotations

import pathlib

import torch

from tide.data import TokenSequenceDataset, UInt32TokenWriter, deterministic_batches


def test_uint32_dataset_and_resume_cursor(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "tokens.bin"
    writer = UInt32TokenWriter(path)
    for token in range(24):
        writer.append(token)
    info = writer.close()
    assert info["tokens"] == 24
    dataset = TokenSequenceDataset(path, sequence_length=4)
    assert len(dataset) == 6
    assert dataset[2].tolist() == [8, 9, 10, 11]

    uninterrupted = deterministic_batches(dataset, batch_size=2, seed=31)
    first, cursor = next(uninterrupted)
    expected_second, expected_cursor = next(uninterrupted)
    resumed = deterministic_batches(
        dataset,
        batch_size=2,
        seed=31,
        consumed_sequences=cursor,
    )
    actual_second, actual_cursor = next(resumed)
    assert cursor == 2
    assert actual_cursor == expected_cursor == 4
    assert torch.equal(actual_second, expected_second)
    assert not torch.equal(first, actual_second)
