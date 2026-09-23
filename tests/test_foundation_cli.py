"""End-to-end entry/record/rejection checks; tiny smoke is not performance evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from experiment_record import read_events
from foundation_report import summarize


def command(*args):
    return [sys.executable,str(ROOT/'scripts/benchmark_foundation.py'),'--device','cpu',*args]


def test_three_family_smoke_has_real_modes_and_terminal_records(tmp_path):
    build=Path(os.environ.get('TIDE_BUILD_DIR',ROOT/'build')).resolve()
    out=tmp_path/'smoke'
    args=command('--tier','smoke','--ids','TR01','--variants','native-stream-packed','native-frontier','native-settle',
                 '--build-dir',str(build),'--output-dir',str(out),'--tracking','off','--workers','2','--warmup','0','--allow-dirty-smoke')
    p=subprocess.run(args,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
    assert p.returncode==0,p.stdout
    suite=json.loads((out/'suite.json').read_text())
    assert suite['state']=='evaluated' and suite['failed_runs']==0
    assert {r['config']['graph'] for r in suite['runs']}=={'pdg','timed-dag','settle'}
    for run in suite['runs']:
        directory=out/run['directory'];record=json.loads((directory/'run.json').read_text())
        assert record['status']=='completed'
        event,=read_events(directory/'metrics.jsonl',record['run_id'])
        assert all(event['metrics'][f'perf/{mode}/seconds']>0 for mode in ('grad-forward','backward','optimizer','train-step'))
        measured=json.loads((directory/'worker/measured.json').read_text())
        assert measured['optimizer_updates']==2 and measured['observations']['outputs']==24
        audit=json.loads((directory/'process-audit.json').read_text())
        assert audit['worker_exit_code']==0 and audit['remaining_group_pids']==[]
    report=summarize(suite)
    assert all(r['completed_repeats']==1 and not r['dispersion_qualified'] for r in report['rows'])


@pytest.mark.parametrize('args',[['--tier','smoke','--timeout-seconds','0'],['--tier','medium','--repeats','1'],['--tier','smoke','--ids','P01','--variants','native-settle'],
                                ['--tier','large','--modes','backward'],['--tier','smoke','--ids','P01','--modes','train-step']])
def test_unsupported_requests_are_rejected(args):
    result=subprocess.run(command(*args,'--describe'),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    assert result.returncode!=0
