"""Binary material preserves actual aliases, exact bytes and publication."""
import hashlib
import json
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


def test_legacy_decode_keeps_tensor_indices_exact_values_and_refusals(tmp_path):
    from decode_weighted_fixture import decode
    from durable_records import write_json
    w = TensorWriter(tmp_path/'material.bin')
    x = torch.tensor([0., -0., 1/3, 1e-40, -7.25])
    a, b = w.tensor(x), w.tensor(x.clone())
    z = w.tensor(torch.zeros(2), zero=True)
    data = dict(schema='tide-weighted-benchmark-fixture-v2', tensors=w.rows,
                binary=w.finish(), nodes=[{'first': a, 'shared': a, 'equal': b, 'zero': z}])
    source = tmp_path/'material.json'
    write_json(source, data)
    restored = decode(source, max_elements=12)
    assert restored['nodes'] == data['nodes'] and a != b
    assert len(restored['tensors']) == 3 and restored['tensors'][z]['values'] == [0., 0.]
    # Compare bytes: all FP32 values, subnormal and signed zero survive JSON.
    row = json.loads(json.dumps(restored))['tensors'][a]
    assert torch.tensor(row['values'], dtype=torch.float32).numpy().tobytes() == x.numpy().tobytes()
    with pytest.raises(ValueError, match='element ceiling'):
        decode(source, max_elements=11)
    data['tensors'][a]['offset'] = -1
    write_json(source, data)
    with pytest.raises(ValueError, match='byte range'):
        decode(source, max_elements=12)
    data['tensors'][a]['offset'] = 0
    write_json(source, data)
    (tmp_path/'material.bin').write_bytes(b'\0'*data['binary']['nbytes'])
    with pytest.raises(ValueError, match='hash mismatch'):
        decode(source, max_elements=12)
