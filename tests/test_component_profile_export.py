"""Offline export must preserve measurement/capture identity and fail closed."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import export_component_profile as entry
from durable_records import write_json
from npu_profile_analysis import serial_parser


def capture_at(path):
    raw = path/'raw/session_ascend_pt'
    raw.mkdir(parents=True)
    write_json(raw/'profiler_info.json', {'synthetic': True})
    (raw/'collection').write_bytes(b'immutable raw')
    write_json(path/'measurement.json', dict(device='npu:0', source='capture-source', dirty='', seconds=.1))
    write_json(path/'result.json', dict(state='collected', profile_state='collected',
        raw_files_sha256=entry.raw_identity(path/'raw'),
        measurement_sha256=entry.digest(path/'measurement.json')))


@pytest.mark.parametrize('mode', ['passed', 'no-kernel', 'fallback', 'export-failed', 'changed-input', 'stale-trace', 'capture-running', 'deleted-original'])
def test_offline_export_lifecycle(tmp_path, monkeypatch, mode):
    capture, out, caller = (tmp_path/name for name in ('capture', 'export', 'caller'))
    caller.mkdir();capture_at(capture)
    monkeypatch.chdir(caller)
    monkeypatch.setattr(entry, 'source_state', lambda root: ('export-source', ''))
    if mode == 'changed-input':
        (capture/'raw/session_ascend_pt/collection').write_bytes(b'changed')
    if mode == 'capture-running':
        data = json.loads((capture/'result.json').read_text());data['state'] = 'running'
        write_json(capture/'result.json', data)
    if mode == 'stale-trace':
        write_json(capture/'raw/session_ascend_pt/trace_view.json', {'traceEvents': []})
        data = json.loads((capture/'result.json').read_text())
        data['raw_files_sha256'] = entry.raw_identity(capture/'raw');write_json(capture/'result.json', data)
    before = entry.raw_identity(capture)
    calls = []
    def analyse(raw, output):
        calls.append(raw)
        assert Path.cwd() == out and raw == out/'raw-copy' and output == out
        assert not list(raw.rglob('trace_view.json'))
        if mode == 'export-failed':
            raise RuntimeError('synthetic worker failure')
        if mode == 'deleted-original':
            (capture/'measurement.json').unlink()
        if mode == 'stale-trace':
            return {}
        kernels = [] if mode == 'no-kernel' else [dict(ph='X', cat='kernel', name='device-op', ts=1, dur=2)]
        if mode == 'fallback':
            kernels.append(dict(name='fallback_to_cpu'))
        write_json(raw/'session_ascend_pt/trace_view.json', dict(traceEvents=kernels))
        return dict(parser_workers=1)
    monkeypatch.setattr(entry, 'analyse_raw', analyse)
    monkeypatch.setattr(sys, 'argv', ['export_component_profile', '--capture-dir', '../capture', '--output-dir', '../export'])
    if mode == 'passed':
        entry.main()
    else:
        with pytest.raises((RuntimeError, ValueError)):
            entry.main()
    assert Path.cwd() == caller and not list(caller.iterdir())
    if mode != 'deleted-original':
        assert entry.raw_identity(capture) == before
    result = json.loads((out/'result.json').read_text())
    assert result['state'] == ('passed' if mode == 'passed' else 'failed')
    assert len(calls) == (0 if mode in {'changed-input', 'capture-running'} else 1)
    if mode == 'deleted-original':
        assert result['original_unchanged'] is False
    if mode == 'passed':
        assert result['seconds'] == .1 and result['capture_source'] == 'capture-source'
        assert result['source'] == 'export-source' and result['original_unchanged']
        assert result['kernel_events'] == 1
    with pytest.raises(FileExistsError):
        entry.main()  # Existing export can never be overwritten.


def test_export_cannot_write_inside_capture(tmp_path, monkeypatch):
    capture = tmp_path/'capture';capture_at(capture)
    monkeypatch.setattr(sys, 'argv', ['export', '--capture-dir', str(capture), '--output-dir', str(capture/'export')])
    with pytest.raises(SystemExit):
        entry.main()
    assert not (capture/'export').exists()


class SyntheticManager:
    def __init__(self, *, max_concurrent_num=16, progress_bar=None):
        self.max_concurrent_num = max_concurrent_num
        self.task_infos = {}

    def add_task(self, task):
        self.task_infos[task.name] = SimpleNamespace(task=task, status=SimpleNamespace(name='Init'))

    def run(self):
        assert self.max_concurrent_num == 1
        done = set()
        for name, info in self.task_infos.items():
            # Reproduce the real manager's early exit when its sole remaining
            # running task is marked NON_BLOCKING, leaving dependent work Init.
            if info.task.mode & 4:
                return
            assert set(info.task.deps) <= done
            info.status.name = info.task.outcome
            if info.status.name != 'Succeed':
                return
            done.add(name)


@pytest.mark.parametrize('outcome', ['Succeed', 'Failed'])
def test_serial_adapter_keeps_timeline_and_dependents_and_restores(tmp_path, outcome):
    parser = SimpleNamespace(ConcurrentTasksManager=SyntheticManager)
    task_type = type('CANNTimelineParser', (), {})
    timeline = task_type();timeline.name='timeline';timeline.mode=6;timeline.deps=[];timeline.outcome='Succeed'
    receipt = tmp_path/'tasks.json'
    def run():
        with serial_parser(parser, receipt):
            manager = parser.ConcurrentTasksManager()
            manager.add_task(SimpleNamespace(name='export', mode=1, deps=[], outcome=outcome))
            manager.add_task(timeline)
            manager.add_task(SimpleNamespace(name='view', mode=2, deps=['timeline'], outcome='Succeed'))
            assert manager.task_infos['timeline'].task.deps == ['export']
            manager.run()
    if outcome == 'Failed':
        with pytest.raises(RuntimeError, match='unexecuted'):
            run()
    else:
        run()
    assert parser.ConcurrentTasksManager is SyntheticManager
    assert timeline.deps == [] and timeline.mode == 6
    data = json.loads(receipt.read_text())
    assert data['state'] == ('passed' if outcome == 'Succeed' else 'failed')
    assert data['tasks']['view'] == ('Succeed' if outcome == 'Succeed' else 'Init')


@pytest.mark.parametrize('mode', [0, 6, 8])
def test_unknown_analysis_task_modes_fail_and_restore(tmp_path, mode):
    parser = SimpleNamespace(ConcurrentTasksManager=SyntheticManager)
    with pytest.raises(RuntimeError, match='unsupported'):
        with serial_parser(parser, tmp_path/'tasks.json'):
            parser.ConcurrentTasksManager().add_task(SimpleNamespace(name='unknown', mode=mode))
    assert parser.ConcurrentTasksManager is SyntheticManager


def test_unknown_manager_interface_fails(tmp_path):
    parser = SimpleNamespace(ConcurrentTasksManager=object)
    with pytest.raises(RuntimeError, match='interface'):
        with serial_parser(parser, tmp_path/'tasks.json'):
            pytest.fail('unsupported manager accepted')
    assert parser.ConcurrentTasksManager is object
