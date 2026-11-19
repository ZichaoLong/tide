import gzip
import hashlib
import json

import pytest

from scripts.materialize_record import materialize


def fixture(tmp_path, payload=b'{\n  "state": "failed", "value": 0.125\n}\n'):
    root = tmp_path / 'repo'
    directory = root / 'artifacts' / 'run'
    directory.mkdir(parents=True)
    data = gzip.compress(payload, mtime=0)
    archive = directory / 'result.json.gz'
    archive.write_bytes(data)
    row = dict(path='artifacts/run/result.json', archive_path='artifacts/run/result.json.gz',
               original_bytes=len(payload), original_sha256=hashlib.sha256(payload).hexdigest(),
               archive_sha256=hashlib.sha256(data).hexdigest())
    index = root / 'index.json'
    def update():
        index.write_text(json.dumps(dict(schema='tide-json-archive-index-v1', records=[row])))
    update()
    return root, archive, index, row, payload, update


def test_exact_bytes_and_no_overwrite(tmp_path):
    root, archive, index, row, payload, _ = fixture(tmp_path)
    output = tmp_path / 'restored.json'
    result = materialize(index, row['path'], output, root=root)
    assert output.read_bytes() == payload
    assert result['sha256'] == row['original_sha256']
    with pytest.raises(FileExistsError):
        materialize(index, row['path'], output, root=root)
    assert output.read_bytes() == payload


@pytest.mark.parametrize('field', ['archive_sha256', 'original_sha256', 'original_bytes'])
def test_corruption_or_false_identity_leaves_no_output(tmp_path, field):
    root, archive, index, row, _, update = fixture(tmp_path)
    row[field] = 1 if field == 'original_bytes' else '0' * 64
    update()
    output = tmp_path / 'bad.json'
    with pytest.raises(ValueError):
        materialize(index, row['path'], output, root=root)
    assert not output.exists()
    assert not list(tmp_path.glob('.record-*'))


def test_truncated_archive_with_matching_outer_hash_is_rejected(tmp_path):
    root, archive, index, row, _, update = fixture(tmp_path)
    damaged = archive.read_bytes()[:-4]
    archive.write_bytes(damaged)
    row['archive_sha256'] = hashlib.sha256(damaged).hexdigest()
    update()
    output = tmp_path / 'bad.json'
    with pytest.raises(EOFError):
        materialize(index, row['path'], output, root=root)
    assert not output.exists()


@pytest.mark.parametrize('kind', ['traversal', 'symlink', 'duplicate'])
def test_ambiguous_or_escaped_paths_are_rejected(tmp_path, kind):
    root, archive, index, row, _, update = fixture(tmp_path)
    if kind == 'traversal':
        row['archive_path'] = 'artifacts/../outside.gz'
        update()
    elif kind == 'symlink':
        outside = tmp_path / 'outside.gz'
        archive.rename(outside)
        archive.symlink_to(outside)
    else:
        index.write_text(json.dumps(dict(schema='tide-json-archive-index-v1', records=[row, row])))
    output = tmp_path / 'bad.json'
    with pytest.raises(ValueError):
        materialize(index, row['path'], output, root=root)
    assert not output.exists()
