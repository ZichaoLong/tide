"""Both actual CLI clients independently train/infer with the declared owner map."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from test_online_consumer import packet,make_continuous_packet,ROOT
from test_online_consumer_npu import target
from online_consumer_support import observer,same
from test_consumer_sample_chunks import compare_records
from tools.online_bench.host import run

CASES=[('add','float32','pdg','streaming',True,2,5,2),
       ('attention','float16','timed-dag','prefill',True,2,2,0),
       ('attention','float32','settle','prefill',True,2,2,0),
       ('add','float32','timed-dag','prefill',True,1,2,0),
       ('attention','float16','timed-dag','streaming',False,2,2,0),
       ('add','float32','settle','prefill',False,2,2,0)]


@pytest.mark.parametrize('implementation',['native','libtorch'])
@pytest.mark.parametrize('case',CASES)
def test_fixed_owner_cli_complete_update(case,implementation,tmp_path):
    target()
    memory,dtype,family,schedule,training,devices,batch,rows=case
    p=packet(memory);c=p['workload']
    p=make_continuous_packet(graph=p['graph'],memory=memory,width=c['width'],batch=batch,tokens=c['tokens'],
        vocab=c['vocab'],budget=c['budget'],seed=c['seed'],clear=c['clear'])
    owners=[(i+1)%devices for i in range(p['graph']['nodes']+2)]
    path=tmp_path/'packet.json';path.write_text(json.dumps(p));out=tmp_path/'consumer'
    command=[sys.executable,str(ROOT/'scripts/run_execution_flow.py'),'--packet',str(path),'--output-dir',str(out),
        '--device','npu:0','--implementation',implementation,'--preset','resident','--family',family,'--schedule',schedule,
        '--dtype',dtype,'--steps','2','--warmup','0','--windows-per-step','2','--optimizer','adamw',
        '--devices',str(devices),'--owner-map',','.join(map(str,owners)),'--chunk-policy','aggressive','--diagnostics',
        '--resident-outputs','64','--resident-trace','4096','--resident-kv-trace-rows','8192']
    if training:command.append('--training')
    if rows:command.extend(['--sample-chunk-rows',str(rows),'--auto-sample-chunks'])
    if implementation=='libtorch':command.extend(['--native-binary',os.environ['TIDE_ONLINE_BINARY']])
    else:command.extend(['--native-library',os.environ['TIDE_BUILD_DIR'],'--resident-library',os.environ['TIDE_RESIDENT_LIBRARY']])
    done=subprocess.run(command,capture_output=True,text=True,timeout=120)
    assert done.returncode==0,done.stdout+done.stderr
    got=json.loads((out/'result.json').read_text())
    records=(out/'consumer' if implementation=='libtorch' else out)/'diagnostics.jsonl'
    actual=[json.loads(s) for s in records.read_text().splitlines()]
    expected=[];ref=run(p,family=family,implementation='python',device='cpu',schedule='streaming',training=training,
        optimizer='adamw',steps=2,warmup=0,windows_per_step=2,diagnostics=True,observer=observer(expected))
    half=dtype=='float16';tol=dict(atol=2e-3,rtol=2e-2) if half else dict(atol=1e-6,rtol=1e-5)
    if rows:compare_records(actual,expected,batch,**tol,tensor_norm=half)
    else:same(actual,expected,**tol,tensor_norm=half)
    torch.testing.assert_close(torch.tensor(got['losses']),torch.tensor(ref['losses']),**tol)
    for key in ('parameters','outputs','final_cut','input_tokens_per_step'):assert got[key]==ref[key]
    ad=got['memory_admission'];assert ad['full_owners']==ad['state_owners']==owners
    assert ad['owner_selection']=='explicit' and ad['owner_moves']==ad['owner_evaluations']==0
    assert ad['allocator_within_estimate']
    (tmp_path/'observed.json').write_text(json.dumps(dict(candidate=got,observations=actual)))


@pytest.mark.parametrize('implementation',['native','libtorch'])
def test_fixed_owner_cli_rejects_unused_device(implementation,tmp_path):
    target();p=packet();path=tmp_path/'packet.json';path.write_text(json.dumps(p));out=tmp_path/'consumer'
    command=[sys.executable,str(ROOT/'scripts/run_execution_flow.py'),'--packet',str(path),'--output-dir',str(out),
        '--device','npu:0','--implementation',implementation,'--preset','resident','--family','timed-dag','--schedule','prefill',
        '--steps','1','--warmup','0','--devices','2','--owner-map',','.join(['0']*(p['graph']['nodes']+2))]
    if implementation=='libtorch':command.extend(['--native-binary',os.environ['TIDE_ONLINE_BINARY']])
    else:command.extend(['--native-library',os.environ['TIDE_BUILD_DIR'],'--resident-library',os.environ['TIDE_RESIDENT_LIBRARY']])
    r=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert r.returncode!=0
    record=json.loads((out/'result.json').read_text())
    if implementation=='libtorch':
        assert record['state']=='failed'
        record=json.loads((out/'consumer/result.json').read_text())
    assert record['state']=='failed' and 'invalid consumer owner map' in record['error']
    assert not (out/'diagnostics.jsonl').exists()
