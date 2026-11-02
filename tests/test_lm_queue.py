from pathlib import Path
import json
from tidegraph.lm.queue import placeholder, asset_status
from tidegraph.lm.matrix import case
from tidegraph.lm.report import load


def test_dependency_failure_is_not_model_failure_or_throughput(tmp_path):
    placeholder(case('qwen15','cpu'),tmp_path,'dependency_failed','network')
    row=load(tmp_path)[0]
    assert row['execution_status']=='dependency_failed'
    assert row['numerical']=='not_checked' and row['output_tokens_per_s'] is None


def test_missing_asset_never_means_ready(tmp_path):
    assert asset_status('qwen05',tmp_path)=='waiting'


def test_check_failure_has_its_own_schema(tmp_path):
    placeholder(case('qwen15','cpu',kind='check'),tmp_path,'dependency_failed','network')
    assert json.loads((tmp_path/'check.json').read_text())['schema']=='tide-lm-check-v1'
    assert not load(tmp_path)


def test_queue_reaches_worker_and_preserves_effective_manifest(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tidegraph.lm import queue
    from tidegraph.experiment.records import read_json,write_json
    item=case('qwen15','cpu',2,16,kind='check',output_tokens=4)
    monkeypatch.setattr(queue,'manifest',lambda:dict(cases=[item],identity='test'))
    monkeypatch.setattr(queue,'asset_status',lambda *_:'ready')
    def run(command,**kwargs):
        output=Path(command[command.index('--case-dir')+1])/'run';output.mkdir()
        write_json(output/'check.json',dict(state='completed'))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(queue.subprocess,'run',run)
    result=queue.execute(tmp_path,'cpu',hours=.01,attempt='test')
    assert result['state']=='completed' and result['results'][0]['exit_code']==0
    assert read_json(tmp_path/'runs/cpu-test/manifest.json')['cases']==[item]
    assert (tmp_path/'runs/cpu-test/report.html').exists()


def test_worker_failure_is_visible_even_when_outer_manifest_finishes(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tidegraph.lm import queue
    item=case('qwen15','cpu')
    monkeypatch.setattr(queue,'manifest',lambda:dict(cases=[item]))
    monkeypatch.setattr(queue,'asset_status',lambda *_:'ready')
    monkeypatch.setattr(queue.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=7))
    result=queue.execute(tmp_path,'cpu',hours=.01)
    assert result['results'][0]['exit_code']==7
    row=load(tmp_path)[0]
    assert row['execution_status']=='process_failed' and row['output_tokens_per_s'] is None


def test_compute_lock_timeout_is_bounded_and_does_not_steal_lock(tmp_path):
    import fcntl
    import pytest
    from tidegraph.lm.queue import compute_lock
    path=tmp_path/'compute.lock'
    with path.open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX)
        with pytest.raises(TimeoutError):
            with compute_lock(path,.05):
                pytest.fail('stole held lock')
    with compute_lock(path,.1):
        pass


def test_kernel_waiter_receives_released_lock(tmp_path):
    import fcntl
    import threading
    import time
    from tidegraph.lm.queue import compute_lock
    path=tmp_path/'compute.lock'
    with path.open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX)
        release=threading.Timer(.05,lambda:fcntl.flock(owner,fcntl.LOCK_UN));release.start()
        begin=time.monotonic()
        with compute_lock(path,1):
            assert time.monotonic()-begin>=.04
        release.join()
