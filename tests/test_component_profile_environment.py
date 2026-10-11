"""Synthetic lifecycle checks, without accelerator execution or benchmark claims."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import profile_batched_components as entry


@pytest.mark.parametrize('mode', ['passed', 'profiled', 'resolution-failed', 'execution-failed'])
def test_component_initialization_and_profile_keep_artifacts_outside_caller(tmp_path, monkeypatch, mode):
    import tidegraph.runtime as runtime

    caller = tmp_path/'caller';caller.mkdir()
    out = tmp_path/'result'
    monkeypatch.chdir(caller)
    monkeypatch.setattr(entry.torch, 'set_num_interop_threads', lambda count: None)
    monkeypatch.setattr(entry, 'source_state', lambda root: ('fixed', ''))
    def resolve(spec):
        assert spec == 'cpu' and Path.cwd() == out
        Path('kernel_meta').mkdir()
        if mode == 'resolution-failed':
            raise RuntimeError('synthetic unavailable device')
        return torch.device('cpu'), 'explicit:cpu'
    monkeypatch.setattr(runtime, 'resolve_device', resolve)
    def make_case(*args):
        assert Path.cwd() == out
        Path('kernel_meta/model-cache').write_text('synthetic vendor artifact')
        return object()
    monkeypatch.setattr(entry, 'make_case', make_case)
    calls = []
    def operation(case, kind, variant):
        assert Path.cwd() == out
        calls.append(variant)
        if mode == 'execution-failed':
            raise RuntimeError('synthetic operation failure')
        return [torch.ones(1)], [None]
    monkeypatch.setattr(entry, 'operation', operation)
    class Trace:
        def __enter__(self):
            assert Path.cwd() == out
            return self
        def __exit__(self, *args):
            return False
        def export_chrome_trace(self, path):
            assert Path.cwd() == out
            Path(path).write_text(json.dumps({'traceEvents': []}))
    monkeypatch.setattr(entry.torch.profiler, 'profile', lambda **kwargs: Trace())
    argv = ['profile_batched_components', '--device', 'cpu', '--kind', 'read',
            '--variant', 'batched', '--rows', '2', '--width', '4', '--output-dir', '../result']
    if mode == 'profiled':
        argv.append('--profile')
    monkeypatch.setattr(sys, 'argv', argv)
    if mode.endswith('failed'):
        with pytest.raises(RuntimeError, match='synthetic'):
            entry.main()
    else:
        entry.main()
    assert Path.cwd() == caller and not list(caller.iterdir())
    assert (out/'kernel_meta').is_dir()
    record = json.loads((out/'result.json').read_text())
    assert record['state'] == ('failed' if mode.endswith('failed') else 'passed')
    assert record['requested_device'] == 'cpu'
    if mode == 'resolution-failed':
        assert record['device'] is None and not calls
    elif mode == 'execution-failed':
        assert calls == ['replay'] and not (out/'measurement.json').exists()
    else:
        assert calls == ['replay', 'batched', 'batched', 'batched'] + (['batched'] if mode == 'profiled' else [])
        assert record['warmup'] == record['measured_steps'] == 1
        assert (out/'trace.json').exists() == (mode == 'profiled')
