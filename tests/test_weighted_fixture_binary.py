"""Binary material preserves actual aliases, exact bytes and publication."""
import hashlib
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from weighted_fixture_binary import TensorWriter


def test_binary_tensor_bytes_zero_codec_and_actual_aliases(tmp_path):
    path = tmp_path/"material.bin"
    w = TensorWriter(path)
    a = torch.arange(12., requires_grad=True).reshape(3,4)
    i = w.tensor(a)
    assert w.tensor(a) == i
    j = w.tensor(a.clone())
    assert j != i
    z = w.tensor(torch.zeros(32,32), zero=True)
    meta = w.finish()
    raw = path.read_bytes()
    assert meta["nbytes"] == len(raw) == 96
    assert hashlib.sha256(raw).hexdigest() == meta["sha256"]
    for row in w.rows[:2]:
        data = raw[row["offset"]:row["offset"]+row["nbytes"]]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        restored = torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(row["shape"])
        torch.testing.assert_close(restored, a, rtol=0, atol=0)
    assert w.rows[z]["encoding"] == "zeros"
    with pytest.raises(FileExistsError): TensorWriter(path)


def test_binary_refuses_mislabelled_zero_and_aborts_without_publication(tmp_path):
    path = tmp_path/"material.bin"
    w = TensorWriter(path)
    with pytest.raises(ValueError, match="frozen zeros"): w.tensor(torch.ones(1), zero=True)
    with pytest.raises(ValueError, match="frozen zeros"): w.tensor(torch.zeros(1, requires_grad=True), zero=True)
    with pytest.raises(ValueError, match="FP32"): w.tensor(torch.ones(1, dtype=torch.float64))
    w.abort()
    assert not path.exists() and not list(tmp_path.iterdir())
