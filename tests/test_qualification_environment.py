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


def test_accelerator_entry_preserves_vendor_path_for_fresh_resume(tmp_path, monkeypatch, inherited_vendor):
    import qualify_accelerator as entry
    import accelerator_cases
    import tidegraph.qualification

    monkeypatch.setattr(entry.torch, 'set_num_interop_threads', lambda count: None)
    monkeypatch.setattr(accelerator_cases, 'cases', lambda *args: [('probe', object())])
    monkeypatch.setattr(accelerator_cases, 'fixture_options', lambda *args: {})
    def qualify(*args, **kwargs):
        check_fresh_child(tmp_path, dict(os.environ))
        return dict(state='passed', coverage={}, candidate=dict(runtime={}),
                    checks=[], tolerances={}, elapsed_seconds=0.)
    monkeypatch.setattr(tidegraph.qualification, 'qualify', qualify)
    monkeypatch.setattr(sys, 'argv', ['qualify_accelerator.py', '--device', 'cpu',
        '--implementation', 'python', '--output-dir', str(tmp_path/'gate'), '--case', 'probe'])
    entry.main()
    assert json.loads((tmp_path/'gate/result.json').read_text())['state'] == 'passed'


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
