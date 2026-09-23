"""Real short subprocess failures must leave no task descendants."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from foundation_lifecycle import run_child, group_pids
from foundation_resources import resources, balanced_affinity


def test_budget_uses_effective_constraints():
    r=resources();a=balanced_affinity(r)
    assert set(a)<=set(r['effective_cpu_ids'])
    assert len(a)==r['cpu_budget']<=max(1,r['effective_cpu_count']/2)
    assert r['memory_budget_bytes']*2<=r['host_available_bytes']


@pytest.mark.parametrize('failure',['timeout','memory','orphan'])
def test_reaps_direct_and_orphan_children(tmp_path,failure):
    script=tmp_path/'child.py'
    script.write_text('import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c","import time;time.sleep(20)"])\n'+('' if failure=='orphan' else 'time.sleep(20)\n'))
    audit={}
    with (tmp_path/'log').open('w') as log:
        kwargs=dict(cwd=tmp_path,env=os.environ,log=log,affinity=[min(os.sched_getaffinity(0))],
                    timeout=.5,memory_budget=1 if failure=='memory' else 2**40,audit=audit)
        if failure=='orphan':assert run_child([sys.executable,str(script)],**kwargs)==0
        else:
            with pytest.raises(MemoryError if failure=='memory' else TimeoutError):
                run_child([sys.executable,str(script)],**kwargs)
    assert audit['remaining_group_pids']==[] and not group_pids(audit['pid'])
    assert audit['worker_exit_code'] is not None


def test_signal_cancels_suite_before_another_workload(tmp_path):
    import time
    root=tmp_path/'fake-root';(root/'scripts').mkdir(parents=True)
    ready=tmp_path/'ready'
    (root/'scripts/foundation_worker.py').write_text(
        f'import os,pathlib,time\npathlib.Path({str(ready)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n')
    real=Path(__file__).resolve().parents[1]
    harness=tmp_path/'harness.py'
    harness.write_text(f'''import sys
from pathlib import Path
from types import SimpleNamespace
sys.path[:0]=[{str(real/'scripts')!r},{str(real/'python')!r}]
from foundation_control import run_variant
config=dict(id='P01',graph='pdg',topology='sparse-ring',width=8,batch=1,sequence=2,body_nodes=4,module='ema',dtype='float32',training_window=None)
args=SimpleNamespace(threads=1,workers=1,warmup=0,timeout_seconds=30,tracking='off')
run_variant(Path({str(root)!r}),Path('unused'),Path({str(tmp_path/'run')!r}),config,'native-stream-packed',['nograd-forward'],args,{{'native_binary_sha256':'unused'}},0)
Path({str(tmp_path/'continued')!r}).touch()
''')
    env=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (tmp_path/'harness.log').open('w') as log:
        p=subprocess.Popen([sys.executable,str(harness)],env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+20
            while not ready.exists() and p.poll() is None and time.monotonic()<deadline:time.sleep(.1)
            assert ready.exists(),(tmp_path/'harness.log').read_text()
            p.send_signal(signal.SIGTERM)
            assert p.wait(timeout=10)==143
        finally:
            if p.poll() is None:p.kill();p.wait()
    summary=json.loads((tmp_path/'run/summary.json').read_text())
    assert summary['status']=='cancelled' and summary['exit_code']==143
    assert summary['unreaped_child_pid'] is None
    assert not group_pids(int(ready.read_text())) and not (tmp_path/'continued').exists()


def test_export_identity_rejects_changed_sources(tmp_path):
    from source_identity import source_state,digest
    (tmp_path/'a.py').write_text('one')
    (tmp_path/'source-export.json').write_text(json.dumps(dict(schema='tide-foundation-source-v1',
        commit='frozen-test-source',files_sha256={'a.py':digest(tmp_path/'a.py')})))
    assert source_state(tmp_path)==('frozen-test-source','')
    (tmp_path/'a.py').write_text('two')
    with pytest.raises(ValueError,match='source changed'):source_state(tmp_path)
