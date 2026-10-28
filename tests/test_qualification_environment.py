"""Fresh qualification children need both the fixed project and vendor paths."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from python_environment import with_pythonpath
from verify_resident_target import target_environment


@pytest.fixture
def inherited_vendor(tmp_path, monkeypatch):
    vendor = tmp_path/'vendor'
    vendor.mkdir()
    (vendor/'tide_vendor_env_probe.py').write_text('value = 37\n')
    # The fixed checkout must take precedence over an inherited project copy.
    (vendor/'tidegraph').mkdir()
    (vendor/'tidegraph/__init__.py').write_text('raise RuntimeError("stale project")\n')
    monkeypatch.setenv('PYTHONPATH', str(vendor))
    return vendor


def check_fresh_child(tmp_path, environment):
    code = ('import importlib.util,json,tide_vendor_env_probe;'
            'print(json.dumps([tide_vendor_env_probe.value,'
            'importlib.util.find_spec("tidegraph").origin]))')
    child = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                           env=environment, text=True, capture_output=True)
    assert child.returncode == 0, child.stderr
    value, location = json.loads(child.stdout)
    assert value == 37
    assert Path(location) == ROOT/'python/tidegraph/__init__.py'


@pytest.mark.parametrize('fails', [False, True])
@pytest.mark.parametrize('implementation', ['python', 'native'])
def test_accelerator_entry_preserves_vendor_path_for_fresh_resume(tmp_path, monkeypatch, inherited_vendor, fails, implementation):
    import qualify_accelerator as entry
    import accelerator_cases
    import tidegraph.qualification

    monkeypatch.setattr(entry.torch, 'set_num_interop_threads', lambda count: None)
    monkeypatch.setattr(accelerator_cases, 'cases', lambda *args: [('probe', object())])
    monkeypatch.setattr(accelerator_cases, 'fixture_options', lambda *args: {})
    monkeypatch.chdir(tmp_path)
    previous = Path.cwd()
    library = tmp_path/'core'
    library.mkdir()
    (library/'build-manifest.json').write_text('{}')
    def qualify(*args, **kwargs):
        assert Path.cwd() == tmp_path/'gate'
        assert kwargs['native_library'] == (str(library) if implementation == 'native' else None)
        check_fresh_child(Path.cwd(), dict(os.environ))
        (Path.cwd()/'kernel_meta').mkdir()
        if fails:
            raise RuntimeError('retained qualification failure')
        return dict(state='passed', coverage={}, candidate=dict(runtime={}),
                    checks=[], tolerances={}, elapsed_seconds=0.)
    monkeypatch.setattr(tidegraph.qualification, 'qualify', qualify)
    monkeypatch.setattr(sys, 'argv', ['qualify_accelerator.py', '--device', 'cpu',
        '--implementation', implementation, '--output-dir', 'gate', '--case', 'probe'] +
        (['--native-library', 'core'] if implementation == 'native' else []))
    if fails:
        with pytest.raises(RuntimeError, match='retained qualification failure'):
            entry.main()
    else:
        entry.main()
    assert Path.cwd() == previous
    assert (tmp_path/'gate/kernel_meta').is_dir()
    assert json.loads((tmp_path/'gate/result.json').read_text())['state'] == ('failed' if fails else 'passed')


def test_unsupported_csr_initialization_keeps_vendor_artifacts_outside_source(tmp_path, monkeypatch):
    import qualify_accelerator as entry
    import accelerator_cases
    import tidegraph

    class Config:
        def to_dict(self):
            return {}

    monkeypatch.setattr(entry.torch, 'set_num_interop_threads', lambda count: None)
    monkeypatch.setattr(accelerator_cases, 'cases', lambda *args: [('fiber-policy-fiber_pooling', Config())])
    previous = Path.cwd()
    def reject(*args, **kwargs):
        assert Path.cwd() == tmp_path/'gate'
        (Path.cwd()/'kernel_meta').mkdir()
        raise ValueError('NPU CSR pooling is unsupported')
    monkeypatch.setattr(tidegraph, 'GraphRuntime', reject)
    monkeypatch.setattr(sys, 'argv', ['qualify_accelerator.py', '--device', 'npu:0',
        '--implementation', 'python', '--output-dir', str(tmp_path/'gate')])
    entry.main()
    record = json.loads((tmp_path/'gate/result.json').read_text())
    assert record['state'] == 'passed' and record['unsupported'][0]['rejection_verified']
    assert (tmp_path/'gate/kernel_meta').is_dir()
    assert Path.cwd() == previous


@pytest.mark.parametrize('device', ['npu:0', 'cuda:0'])
def test_resident_children_preserve_vendor_path_and_fixed_build(tmp_path, inherited_vendor, device):
    core = tmp_path/'core'
    env = target_environment(ROOT, core, tmp_path/'resident', tmp_path/'consumer', device)
    check_fresh_child(tmp_path, env)
    assert env['PYTHONPATH'].split(os.pathsep) == [str(ROOT/'python'), str(core), str(inherited_vendor)]
    assert env['TIDE_RESIDENT_DEVICE'] == env['TIDE_ONLINE_DEVICE'] == device
    assert os.environ['PYTHONPATH'] == str(inherited_vendor)


@pytest.mark.parametrize('inherited', [None, ''])
def test_unset_vendor_path_does_not_add_current_directory(inherited, tmp_path):
    env = {} if inherited is None else {'PYTHONPATH': inherited}
    result = with_pythonpath(env, tmp_path/'fixed')
    assert result['PYTHONPATH'].split(os.pathsep) == [str(tmp_path/'fixed')]


@pytest.mark.parametrize('fails', [False, True])
def test_standalone_component_scratch_is_outside_frozen_source(tmp_path, monkeypatch, fails):
    import verify_device_control as entry
    from source_identity import digest

    build = tmp_path/'build'
    build.mkdir()
    binary = build/'probe'
    binary.write_text('#!/bin/sh\nmkdir kernel_meta\n' + ('exit 2\n' if fails else 'echo component-passed\n'))
    binary.chmod(0o700)
    (build/'control-build.json').write_text(json.dumps(dict(
        backend='npu', component_sha256='component', core=dict(cpp_source_sha256='core'),
        binary_sha256={'probe': digest(binary)})))
    monkeypatch.setattr(entry, 'component_hash', lambda _: 'component')
    monkeypatch.setattr(entry, 'source_hash', lambda _: 'core')
    monkeypatch.setattr(entry, 'source_state', lambda _: ('fixed', ''))
    monkeypatch.setattr(entry, 'CHECKS', {'probe': ('probe', ('float32',))})
    monkeypatch.setattr(entry, 'MARKERS', {'probe': 'component-passed'})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['verify_device_control.py', '--device', 'npu:0',
        '--build-dir', 'build', '--output-dir', 'gate', '--checks', 'probe'])
    if fails:
        with pytest.raises(RuntimeError, match='device gate failed'):
            entry.main()
    else:
        entry.main()
    assert (tmp_path/'gate/kernel_meta').is_dir()
    assert not (tmp_path/'kernel_meta').exists()
    assert Path.cwd() == tmp_path
    assert json.loads((tmp_path/'gate/result.json').read_text())['state'] == ('failed' if fails else 'passed')
