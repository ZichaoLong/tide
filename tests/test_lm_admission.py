"""Capacity exploration is explicit and does not turn execution failures into success."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from tidegraph.lm.admission import assess
from tidegraph.lm.config import resolve
from tidegraph.lm.retry import isolation_manifest
from tidegraph.lm.selection import single, validate

GIB = 1024**3


def request(policy=None):
    c = deepcopy(next(x['config'] for x in isolation_manifest()['cases']
                      if x['name'] == 'stage2-qwenmoe-npu-b512-l512'))
    if policy is not None:
        c['execution']['device_memory_admission'] = policy
    return c


class Backend:
    def mem_get_info(self, index):
        return 61 * GIB, 64 * GIB


def test_legacy_identity_and_default_rejection_are_unchanged():
    resolved = resolve(request())
    assert resolved['config_id'] == '9d3637ed49802f48d584ef426c428f0110ffb3d285c9bd01f0f66c8a13a41534'
    assert 'device_memory_admission' not in resolved['request']['execution']
    rows = []
    with pytest.raises(RuntimeError, match='capacity rejected before model allocation'):
        assess(resolved['request'], resolved['budget'], Backend(), observer=rows.extend)
    assert len(rows) == 2 and all(x['decision'] == 'rejected' for x in rows)
    assert rows[0]['reserve'] == 6.4 * GIB


def test_observe_records_original_budget_without_changing_workload(capsys):
    original = resolve(request())
    resolved = resolve(request('observe'))
    assert original['budget'] == resolved['budget']
    assert original['request']['task'] == resolved['request']['task']
    assert original['config_id'] != resolved['config_id']
    rows = assess(resolved['request'], resolved['budget'], Backend())
    assert len(rows) == 2 and all(x['decision'] == 'observe_exceeded' for x in rows)
    assert all(x['policy'] == 'observe' and not x['within_budget'] for x in rows)
    assert capsys.readouterr().err.count('allocation/runtime errors remain failures') == 2


def test_observe_still_propagates_device_query_errors():
    class Broken:
        def mem_get_info(self, index):
            raise RuntimeError('device query failed')
    r = resolve(request('observe'))
    with pytest.raises(RuntimeError, match='device query failed'):
        assess(r['request'], r['budget'], Broken())


@pytest.mark.parametrize('policy', ['skip-all', False, 1])
def test_policy_is_validated(policy):
    with pytest.raises(ValueError, match='execution.device_memory_admission'):
        resolve(request(policy))


def test_observe_does_not_disable_host_or_configuration_limits():
    c = request('observe'); c['resources']['host_gib'] = 0
    with pytest.raises(ValueError, match='resources.host_gib'):
        resolve(c)
    c = request('observe'); c['execution'].update(backend='cpu', devices=1, placement='single')
    with pytest.raises(ValueError, match='observe requires accelerator Transformers inference'):
        resolve(c)


def test_allocation_failure_keeps_admission_and_failure_result(monkeypatch, tmp_path):
    from tidegraph.lm import engine, resources
    from tidegraph.lm.report import load, render
    class AllocationFailure(RuntimeError):
        pass
    def allocate(resolved, assets, *, on_admission):
        assess(resolved['request'], resolved['budget'], Backend(), observer=on_admission)
        raise AllocationFailure('simulated device OOM')
    monkeypatch.setattr(engine, 'setup', allocate)
    monkeypatch.setattr(engine, 'package_identity', lambda: {'test': 'no model execution'})
    monkeypatch.setattr(resources, 'disk_admission', lambda *a, **k: None)
    with pytest.raises(AllocationFailure, match='simulated device OOM'):
        engine.run(request('observe'), tmp_path/'assets', tmp_path/'run')
    result = json.loads((tmp_path/'run/result.json').read_text())
    assert result['state'] == result['execution_status'] == 'failed'
    assert result['error_kind'] == 'AllocationFailure'
    assert result['exploratory_capacity'] is True
    assert len(result['device_admission']) == 2
    assert result['validation']['checkpoint'] == 'not_checked'
    rows = load(tmp_path)
    assert rows[0]['device_memory_admission'] == 'observe'
    assert 'observe' in render(rows, 'csv') and 'observe' in render(rows, 'html')


def test_single_configuration_queue_preserves_expanded_request(monkeypatch, tmp_path):
    from tidegraph.lm.cli import main
    from tidegraph.lm import queue
    p = tmp_path/'config.json'; p.write_text(json.dumps(request('observe')))
    captured = {}
    def execute(root, backend, **kwargs):
        captured.update(kwargs)
        return {'state': 'test_only'}
    monkeypatch.setattr(queue, 'execute', execute)
    assert main(['queue', '--root', str(tmp_path), '--backend', 'npu', '--config', str(p)]) == 0
    selected = validate(captured['selection'])
    assert len(selected['cases']) == 1
    assert selected == single(p)
    assert selected['cases'][0]['config'] == resolve(request('observe'))['request']


def test_tracked_trial_is_two_device_b512_with_same_math():
    path = Path(__file__).resolve().parents[1]/'examples/external_lm/qwenmoe-two-npu-b512-observe.json'
    trial = resolve(path)['request']; original = request()
    assert trial['task'] == original['task']
    assert trial['resources'] == original['resources']
    expected = {**original['execution'], 'device_memory_admission': 'observe'}
    assert trial['execution'] == expected
    assert trial['measurement'] == {**original['measurement'], 'timeout_seconds': 7200}
