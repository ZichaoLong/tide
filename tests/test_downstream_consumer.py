"""Rejection/atomicity tests for the copyable experiment-owned boundary."""
from pathlib import Path
import json
import sys
import zipfile
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples/consumer_core5'))
from dependency import capture, environment, installed_files, native_identity, verify
from recording import publish, tensor_record
from checkpoint import save_application, restore_application
from task import Task
from tidegraph.planning import ExecutionSpec


def test_lock_checks_installed_bytes_runtime_and_native_identity(tmp_path):
    root, files = installed_files()
    wheel = tmp_path/'fixture.whl'
    with zipfile.ZipFile(wheel, 'w') as z:
        for name in files: z.write(root/name, 'tidegraph/'+name)
    r = capture(wheel, 'a'*40, 'b'*40)
    path = tmp_path/'lock.json';path.write_text(json.dumps(r))
    lock, actual = verify(path)
    assert lock == r and actual['installed_files_sha256'] == r['foundation']['installed_files_sha256']
    r['foundation']['installed_files_sha256'] = '0'*64
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='dependency does not match'): verify(path)
    r = capture(wheel, 'a'*40, 'b'*40);r['environment']['torch'] = 'different'
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='environment differs'): verify(path)
    with zipfile.ZipFile(wheel, 'a') as z: z.writestr('tidegraph/not_installed.py','')
    with pytest.raises(ValueError, match='differs from'): capture(wheel, 'a'*40, 'b'*40)
    with pytest.raises(ValueError, match='immutable'): capture(wheel, 'HEAD', 'b'*40)


def test_native_lock_checks_actual_bytes_and_build_environment(tmp_path):
    from recording import sha256
    binary=tmp_path/'_tide_native.so';binary.write_bytes(b'identity fixture, never loaded')
    manifest=dict(environment(), source='a'*40, cpp_source_sha256='b'*64, backend='cpu',
                  binary_sha256={binary.name:sha256(binary)})
    path=tmp_path/'build-manifest.json';path.write_text(json.dumps(manifest))
    assert native_identity(tmp_path)['binary_sha256'] == sha256(binary)
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError, match='binary/manifest'): native_identity(tmp_path)
    manifest['binary_sha256'][binary.name]=sha256(binary)
    manifest['torch']='different';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='matching CPU'): native_identity(tmp_path)


def test_atomic_application_record_preserves_existing_and_failed_publication(tmp_path):
    target = tmp_path/'checkpoint.pt'
    tensor_record(target, {'sentinel': torch.tensor([7], dtype=torch.int64)})
    before = target.read_bytes()
    with pytest.raises(FileExistsError): tensor_record(target, {'other': 8})
    assert target.read_bytes() == before
    def interrupted(stream):
        stream.write(b'partial')
        raise OSError('injected write failure')
    with pytest.raises(OSError, match='injected'): publish(tmp_path/'failed.pt', interrupted)
    assert not (tmp_path/'failed.pt').exists() and not list(tmp_path.glob('*.tmp'))


def task():
    return Task(dict(topology='fc', family='kv', width=8, region_width=2, depth=2),
                ExecutionSpec(dtype='float64', scoring_dtype='float64', packing='packed').record(),7)


def test_checkpoint_rejects_changed_workload_lock_and_progress_before_reuse(tmp_path):
    original=task();original.step()
    dependency={'lock_sha256':'c'*64}
    path=tmp_path/'application.pt';save_application(path,original,dependency)
    with pytest.raises(ValueError, match='identity'): restore_application(path,task(),{'lock_sha256':'d'*64})
    record=torch.load(path,weights_only=True)
    record['step']+=1
    bad=tmp_path/'bad.pt';torch.save(record,bad)
    with pytest.raises(ValueError, match='progress'): restore_application(bad,task(),dependency)
    record['position']+=2
    torch.save(record,bad)
    with pytest.raises(ValueError, match='clock'): restore_application(bad,task(),dependency)
    altered=task();altered.workload_id='e'*64
    with pytest.raises(ValueError, match='identity'): restore_application(path,altered,dependency)
