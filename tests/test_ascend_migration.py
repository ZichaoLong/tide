"""Reject old/private mixed baselines before installation or build reuse."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ascend_environment import validate
from build_environment import check_reuse
from libtorch_distribution import standalone


def environment():
    return dict(ASCEND_PYTHON_ENV='/public/cann92', ASCEND_CANN_VERSION='9.2.0',
                ASCEND_TORCH_VERSION='2.12.0', ASCEND_TORCH_NPU_VERSION='2.12.0.post3',
                ASCEND_CONSTRAINTS_FILE='/public/constraints.txt')


@pytest.mark.parametrize('base,prefix,versions,error', [
    ('/public/cann91', '/private/old', {'torch':'2.12.0+cpu','torch-npu':'2.12.0.post3'}, 'venv base'),
    ('/public/cann92', '/public/cann92', {'torch':'2.12.0+cpu','torch-npu':'2.12.0.post3'}, 'private venv'),
    ('/public/cann92', '/private/new', {'torch':'2.12.0+cpu','torch-npu':'2.12.0.post2'}, 'torch-npu'),
    ('/public/cann92', '/private/new', {'torch':'2.10.0','torch-npu':'2.12.0.post3'}, 'torch:'),
])
def test_mismatched_private_environment_rejected(base, prefix, versions, error):
    with pytest.raises(ValueError, match=error):
        validate(environment(), base, prefix, versions, require_venv=True)


def test_matching_private_environment_and_explicit_historical_module():
    validate(environment(), '/public/cann92', '/private/new',
             {'torch':'2.12.0+cpu','torch-npu':'2.12.0.post3'}, require_venv=True)
    old = environment() | dict(ASCEND_PYTHON_ENV='/public/cann91', ASCEND_CANN_VERSION='9.1.0',
                               ASCEND_TORCH_NPU_VERSION='2.12.0.post2')
    validate(old, '/public/cann91', '/private/old',
             {'torch':'2.12.0+cpu','torch-npu':'2.12.0.post2'}, require_venv=True)
    with pytest.raises(ValueError, match='load the matching'):
        validate({}, '/base', '/private', {})


def test_same_libtorch_version_cannot_reuse_other_cann_or_sdk(tmp_path, monkeypatch):
    monkeypatch.setenv('ASCEND_CANN_VERSION', '9.2.0')
    (tmp_path/'build-manifest.json').write_text(json.dumps(dict(torch='2.12.0', environment=dict(cann_version='9.1.0'))))
    with pytest.raises(ValueError, match='CANN differs'):
        check_reuse(tmp_path, 'npu', 'standalone', torch_dir='/torch', torch_version='2.12.0')
    monkeypatch.setenv('Torch_npu_DIR', '/sdk92')
    (tmp_path/'CMakeCache.txt').write_text('Torch_npu_DIR:PATH=/sdk91\n')
    with pytest.raises(ValueError, match='Torch_npu_DIR differs'):
        check_reuse(tmp_path, 'npu', 'standalone', torch_dir='/torch', torch_version='2.12.0')


def test_standalone_identity_hashes_actual_distribution_and_checks_abi(tmp_path, monkeypatch):
    import hashlib
    # The independent build driver also runs under the site's Python 3.9.9.
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    config = tmp_path/'share/cmake/Torch'
    config.mkdir(parents=True)
    (config/'TorchConfigVersion.cmake').write_text('set(PACKAGE_VERSION "2.12.0")\n')
    (config/'TorchConfig.cmake').write_text('set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")\n')
    (tmp_path/'lib').mkdir()
    for name in ('libtorch.so', 'libtorch_cpu.so', 'libc10.so'):
        (tmp_path/'lib'/name).write_bytes(name.encode())
    before = standalone(tmp_path, 1)
    (tmp_path/'lib/libtorch_cpu.so').write_bytes(b'changed')
    after = standalone(tmp_path, 1)
    assert before['torch'] == '2.12.0' and before['discovery'] == 'standalone-prefix'
    assert before['distribution_sha256'] != after['distribution_sha256']
    with pytest.raises(ValueError, match='ABI differs'):
        standalone(tmp_path, 0)


@pytest.mark.parametrize('mismatch', ['config', 'identity'])
def test_continuation_refuses_stale_configuration_or_installation(tmp_path, monkeypatch, mismatch):
    import check_ascend_migration as entry
    from tidegraph.experiment import identity
    root = Path(__file__).resolve().parents[1]
    build, previous = tmp_path/'build', tmp_path/'previous'
    build.mkdir()
    (build/'sdk-environment.json').write_text('{}')
    monkeypatch.setattr(identity, 'capture', lambda paths: {'current':'identity'})
    monkeypatch.setattr(entry.subprocess, 'run', lambda *a, **kw: pytest.fail('stale check executed a workload'))
    for client in ('pytorch', 'python-native', 'libtorch'):
        case = previous/client
        (case/'related/inference').mkdir(parents=True)
        config = entry.configuration(root, build, {}, 'cpu', client)
        if mismatch == 'config':
            config['task']['seed'] += 1
        (case/'config.json').write_text(json.dumps(config))
        (case/'related/inference/identity.json').write_text('{"old":"identity"}')
    out = tmp_path/'out'
    monkeypatch.setattr(sys, 'argv', ['gate', '--build-root', str(build), '--output-dir', str(out),
                                    '--device', 'cpu', '--continue-from', str(previous)])
    assert entry.main() == 1
    rows = json.loads((out/'result.json').read_text())['clients']
    assert len(rows) == 3 and all(r['state'] == 'failed' and not r['commands'] for r in rows)
