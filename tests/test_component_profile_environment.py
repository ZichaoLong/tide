"""Synthetic lifecycle checks, without accelerator execution or benchmark claims."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import profile_batched_components as entry


@pytest.mark.parametrize('mode', ['passed', 'profiled', 'npu-profiled', 'npu-deferred', 'resolution-failed', 'execution-failed'])
def test_component_initialization_and_profile_keep_artifacts_outside_caller(tmp_path, monkeypatch, mode):
    import tidegraph.runtime as runtime

    caller = tmp_path/'caller';caller.mkdir()
    out = tmp_path/'result'
    deferred = mode == 'npu-deferred'
    profiled = mode in {'profiled', 'npu-profiled', 'npu-deferred'}
    npu = mode in {'npu-profiled', 'npu-deferred'}
    requested = 'npu:0' if npu else 'cpu'
    class SyntheticDevice:
        type = 'npu'
        def __str__(self):
            return 'npu:0'
    monkeypatch.chdir(caller)
    monkeypatch.setattr(entry.torch, 'set_num_interop_threads', lambda count: None)
    monkeypatch.setattr(entry, 'source_state', lambda root: ('fixed', ''))
    def resolve(spec):
        assert spec == requested and Path.cwd() == out
        Path('kernel_meta').mkdir()
        if mode == 'resolution-failed':
            raise RuntimeError('synthetic unavailable device')
        return (SyntheticDevice(), 'explicit:npu') if npu else (torch.device('cpu'), 'explicit:cpu')
    monkeypatch.setattr(runtime, 'resolve_device', resolve)
    monkeypatch.setattr(runtime, 'synchronize', lambda device: None)
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
            if npu:
                callback(self)
            return False
        def export_chrome_trace(self, path):
            assert not deferred, 'deferred capture must not fork the exporter'
            assert Path.cwd() == out
            events = [dict(ph='X',cat='kernel',name='synthetic-device-event',ts=0,dur=1)] if npu else []
            Path(path).write_text(json.dumps({'traceEvents': events}))
    monkeypatch.setattr(entry.torch.profiler, 'profile', lambda **kwargs: Trace())
    def callback(trace):
        assert (out/'raw/collection').is_file()
    def handler(directory, *, analyse_flag):
        assert Path(directory) == out/'raw' and analyse_flag is False
        return callback
    def npu_profile(**kwargs):
        # Model TorchNPU's export-only cleanup when on_trace_ready is omitted.
        assert kwargs['on_trace_ready'] is callback
        (out/'raw').mkdir();(out/'raw/collection').write_text('synthetic raw collection')
        return Trace()
    if npu:
        profiler = SimpleNamespace(profile=npu_profile, tensorboard_trace_handler=handler,
                                   ProfilerActivity=SimpleNamespace(CPU='CPU', NPU='NPU'))
        monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(profiler=profiler))
    argv = ['profile_batched_components', '--device', requested, '--kind', 'read',
            '--variant', 'batched', '--rows', '2', '--width', '4', '--output-dir', '../result']
    if profiled:
        argv.append('--profile')
    if deferred:
        argv.append('--defer-profile-export')
    monkeypatch.setattr(sys, 'argv', argv)
    if mode.endswith('failed'):
        with pytest.raises(RuntimeError, match='synthetic'):
            entry.main()
    else:
        entry.main()
    assert Path.cwd() == caller and not list(caller.iterdir())
    assert (out/'kernel_meta').is_dir()
    record = json.loads((out/'result.json').read_text())
    assert record['state'] == ('failed' if mode.endswith('failed') else 'collected' if deferred else 'passed')
    assert record['requested_device'] == requested
    if mode == 'resolution-failed':
        assert record['device'] is None and not calls
    elif mode == 'execution-failed':
        assert calls == ['replay'] and not (out/'measurement.json').exists()
    else:
        assert calls == ['replay', 'batched', 'batched', 'batched'] + (['batched'] if profiled else [])
        assert record['warmup'] == record['measured_steps'] == 1
        assert (out/'trace.json').exists() == (profiled and not deferred)
        if deferred:
            assert record['profile_state'] == 'collected' and 'kernel_events' not in record
            assert record['measurement_sha256'] == entry.digest(out/'measurement.json')
            assert record['raw_files_sha256'] == entry.raw_identity(out/'raw')
        if npu:
            assert (out/'raw/collection').read_text() == 'synthetic raw collection'


@pytest.mark.parametrize('profile', [False, True])
def test_deferred_export_rejects_invalid_request_before_model(tmp_path, monkeypatch, profile):
    import tidegraph.runtime as runtime
    monkeypatch.setattr(entry.torch, 'set_num_interop_threads', lambda count: None)
    monkeypatch.setattr(entry, 'source_state', lambda root: ('fixed', ''))
    monkeypatch.setattr(runtime, 'resolve_device', lambda spec: (torch.device('cpu'), 'explicit:cpu'))
    monkeypatch.setattr(entry, 'make_case', lambda *args: pytest.fail('invalid request executed model'))
    out = tmp_path/'result'
    argv = ['capture', '--device', 'cpu', '--kind', 'read', '--variant', 'batched',
            '--defer-profile-export', '--output-dir', str(out)]
    if profile:
        argv.append('--profile')
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ValueError if profile else SystemExit):
        entry.main()
    assert not (out/'measurement.json').exists()
    if profile:
        assert json.loads((out/'result.json').read_text())['state'] == 'failed'
