"""Finite consumer admission: language parity, refusal and real split updates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from flow_protocol import make_continuous_packet, native_text
from flow_topology import ranked_graph
from flow_failure import RecordedFailure
from tools.online_bench.eager_capacity import plan, MIB
from tools.online_bench.host import run
from online_consumer_support import observer
from test_consumer_sample_chunks import compare_records


def packet(memory='attention',batch=5,width=4,delayed=False):
    return make_continuous_packet(graph=ranked_graph(layers=3,region_width=2,fanout=2,local_span=2,delayed=delayed),
        memory=memory,width=width,batch=batch,tokens=2,vocab=7,clear=False)


def probe(p, options, budgets, tmp_path):
    binary=Path(os.environ['TIDE_ONLINE_BINARY']).with_name('tidegraph-eager-capacity-probe')
    source=tmp_path/'packet.txt';source.write_text(native_text(p))
    cmd=[str(binary),'--device=cpu','--packet='+str(source),'--output-dir='+str(tmp_path/'unused'),
         '--family=timed-dag','--preset=cpu','--schedule=prefill','--devices='+str(len(budgets))]
    rename=dict(dtype='dtype',optimizer='optimizer',steps='steps',warmup='warmup',windows='windows-per-step',
        workers='workers',sample_rows='sample-chunk-rows',policy='chunk-policy',
        owner_policy='owner-policy',head_workspace_bytes='head-workspace-bytes')
    for key,name in rename.items():
        if key in options:cmd.append('--'+name+'='+str(options[key]))
    for key,name in [('training','training'),('auto_sample_chunks','auto-sample-chunks')]:
        if options.get(key):cmd.append('--'+name)
    if options.get('owner_map'):cmd.append('--owner-map='+','.join(map(str,options['owner_map'])))
    done=subprocess.run(cmd,input=' '.join(map(str,budgets)),text=True,capture_output=True,timeout=30)
    assert done.returncode==0,done.stdout+done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize('i',range(12))
def test_static_language_parity(i,tmp_path):
    p=packet('attention' if i%2 else 'add',batch=17,width=[4,128,2048][i%3],delayed=bool(i%3))
    devices=1+i%3
    options=dict(dtype='float64' if i%4==0 else 'float32',training=bool(i%3),optimizer='adamw' if i%2 else 'sgd',
        steps=1+i%3,warmup=i%2,windows=1+i%3,workers=1+i%2,auto_sample_chunks=True,
        policy='aggressive' if i%2 else 'conservative',owner_policy='memory' if i%3 else 'locality')
    if i==11:options.update(dtype='float16',training=False)
    if i==7:options['owner_map']=[0,1,0,1,0,1,0,0]
    budgets=[(64-j)*1024**3 for j in range(devices)]
    full=plan(p,budgets=budgets,**options)
    assert probe(p,options,budgets,tmp_path)==full
    # Force the weakest card to reject even B1. Both paths retain every attempt.
    budgets[-1]=MIB
    refused=plan(p,budgets=budgets,**options)
    assert refused['state']=='refused' and refused['effective_sample_rows']==1
    assert probe(p,options,budgets,tmp_path)==refused


def test_logical_kv_survives_physical_split_and_grows_with_continuation():
    p=packet();kwargs=dict(budgets=[64*1024**3]*2,training=True,steps=2,warmup=1)
    full=plan(p,**kwargs);small=plan(p,sample_rows=1,**kwargs)
    for a,b in zip(full['devices'],small['devices']):
        assert a['components']['persistent_state_and_kv']==b['components']['persistent_state_and_kv']
        assert a['components']['learned']==b['components']['learned']
        assert a['components']['gradients_and_optimizer_slots']==b['components']['gradients_and_optimizer_slots']
        assert a['components']['vector_work']>b['components']['vector_work']
    longer=plan(p,**dict(kwargs,steps=3))
    assert all(a['components']['persistent_state_and_kv']>b['components']['persistent_state_and_kv']
               for a,b in zip(longer['devices'],full['devices']))


def head_budget(p,dtype,training):
    kwargs=dict(budgets=[64*1024**3],dtype=dtype,training=training,optimizer='adamw',steps=2,warmup=0,policy='aggressive')
    one=plan(p,sample_rows=1,**kwargs)['devices'][0]['components']['head_work']
    two=plan(p,sample_rows=2,**kwargs)['devices'][0]['components']['head_work']
    budget=((one+two)//2)*10//9
    automatic=plan(p,head_workspace_bytes=budget,auto_sample_chunks=True,**kwargs)
    assert automatic['state']=='admitted' and automatic['effective_sample_rows']==1
    return budget


@pytest.mark.parametrize('family,memory,schedule,training',[
    ('pdg','add','streaming',True),('timed-dag','attention','prefill',True),('settle','attention','streaming',True),
    ('timed-dag','add','prefill',False)])
@pytest.mark.parametrize('implementation',['python','native','libtorch'])
def test_forced_automatic_chunks_preserve_complete_records(family,memory,schedule,training,implementation,dtype,tmp_path):
    name=str(dtype).split('.')[-1];p=packet(memory,delayed=family=='timed-dag')
    # An explicitly selected unavailable backend fails; CPU is the independent
    # numerical reference in every run. Physical IDs belong to the launcher.
    device=os.environ.get('TIDE_EAGER_CAPACITY_DEVICE','cpu')
    count=1 if device=='cpu' else 2
    preset='cpu' if device=='cpu' else {'pdg':'mixed-a','timed-dag':'mixed-b','settle':'mixed-c'}[family]
    expected=[];actual=[];budget=head_budget(p,name,training)
    common=dict(family=family,dtype=name,training=training,optimizer='adamw',steps=2,warmup=0,windows_per_step=2,diagnostics=True)
    reference=run(p,implementation='python',device='cpu',schedule='streaming',observer=observer(expected),**common)
    options=dict(auto_sample_chunks=True,chunk_policy='aggressive',head_workspace_bytes=budget,devices=count,preset=preset)
    if implementation=='libtorch':
        path=tmp_path/'packet.txt';path.write_text(native_text(p));out=tmp_path/'run'
        cmd=[os.environ['TIDE_ONLINE_BINARY'],'--device='+device,'--dtype='+name,'--packet='+str(path),'--output-dir='+str(out),
            '--family='+family,'--preset='+preset,'--devices='+str(count),'--schedule='+schedule,'--steps=2','--warmup=0','--windows-per-step=2',
            '--optimizer=adamw','--diagnostics','--auto-sample-chunks','--chunk-policy=aggressive',
            '--head-workspace-bytes='+str(budget)]
        if training:cmd.append('--training')
        done=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        assert done.returncode==0,done.stdout+done.stderr
        result=json.loads((out/'result.json').read_text())
        actual=[json.loads(row) for row in (out/'diagnostics.jsonl').read_text().splitlines()]
    else:
        if implementation=='native':options['native_library']=os.environ['TIDE_BUILD_DIR']
        result=run(p,implementation=implementation,device=device,schedule=schedule,observer=observer(actual),**common,**options)
    compare_records(actual,expected,5)
    assert result['batch_execution']['effective_sample_chunk_rows']==1
    assert result['memory_admission']['sample_reductions']==2
    assert result['memory_admission']['allocator_within_estimate']
    assert len(result['memory_admission']['devices'])==count
    for key in ('parameters','outputs','final_cut','input_tokens_per_step'):assert result[key]==reference[key]
    torch.testing.assert_close(torch.tensor(result['losses']),torch.tensor(reference['losses']),atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('automatic',[False,True])
def test_refusal_precedes_model_allocation(automatic,monkeypatch):
    import tools.online_bench.host as host
    def forbidden(*args,**kwargs):raise AssertionError('model allocation entered')
    monkeypatch.setattr(host,'build_model',forbidden)
    with pytest.raises(RecordedFailure) as error:
        run(packet(),family='timed-dag',implementation='python',device='cpu',device_memory_bytes=1,auto_sample_chunks=automatic)
    r=error.value.record
    assert r['failure_phase']=='preallocation_memory_admission' and r['memory_admission']['state']=='refused'
    assert r['memory_admission']['effective_sample_rows']==(1 if automatic else 5)


@pytest.mark.parametrize('implementation',['python','native','libtorch'])
def test_unified_cli_retains_refusal(implementation,tmp_path):
    p=packet();path=tmp_path/'packet.json';path.write_text(json.dumps(p));out=tmp_path/'run'
    cmd=[sys.executable,str(ROOT/'scripts/run_execution_flow.py'),'--device','cpu','--packet',str(path),'--output-dir',str(out),
        '--family','timed-dag','--preset','cpu','--schedule','prefill','--implementation',implementation,
        '--device-memory-bytes','1','--auto-sample-chunks','--chunk-policy','aggressive']
    if implementation=='libtorch':cmd+=['--native-binary',os.environ['TIDE_ONLINE_BINARY']]
    elif implementation=='native':cmd+=['--native-library',os.environ['TIDE_BUILD_DIR']]
    done=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    assert done.returncode!=0
    r=json.loads((out/'result.json').read_text())
    assert r['state']=='failed' and r['failure_phase']=='preallocation_memory_admission'
    assert r['memory_admission']['state']=='refused' and r['memory_admission']['effective_sample_rows']==1


def test_underestimate_preserves_full_failed_measurement(monkeypatch):
    from tools.online_bench.memory import MemoryRecord
    original=MemoryRecord.record
    def excessive(self):
        r=original(self)
        if r['phases'][-1]['phase']=='measured':r['phases'][-1]['cpu_peak_rss_bytes']=2**50
        return r
    monkeypatch.setattr(MemoryRecord,'record',excessive)
    with pytest.raises(RecordedFailure) as error:
        run(packet('add',batch=1),family='pdg',implementation='python',device='cpu',steps=1,warmup=0,training=True)
    r=error.value.record
    assert r['failure_phase']=='post_run_memory_calibration'
    assert r['outputs']==[4] and len(r['seconds'])==1 and r['final_cut']>0
    assert not r['memory_admission']['allocator_within_estimate']


def test_offline_eager_cli_needs_no_torch_or_hardware(tmp_path):
    # -S removes site-packages entirely: shape planning must not initialize Torch.
    p=packet();path=tmp_path/'packet.json';path.write_text(json.dumps(p))
    cmd=[sys.executable,'-S',str(ROOT/'scripts/plan_execution_flow.py'),'--packet',str(path),
        '--preset','mixed-c','--devices','2','--device-memory-bytes',str(64*1024**3),
        '--auto-sample-chunks','--training','--chunk-policy','aggressive','--steps','2','--warmup','0']
    expected=plan(p,budgets=[64*1024**3]*2,training=True,policy='aggressive',steps=2,warmup=0,auto_sample_chunks=True)
    done=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
    assert done.returncode==0,done.stderr
    assert json.loads(done.stdout)['memory_admission']==expected
    refused=subprocess.run(cmd+['--device-memory-bytes','1'],text=True,capture_output=True,timeout=30)
    assert refused.returncode==2
    assert json.loads(refused.stdout)['memory_admission']['state']=='refused'
    bad=subprocess.run(cmd+['--resident-queue','1'],text=True,capture_output=True,timeout=30)
    assert bad.returncode==2 and 'resident capacities' in bad.stderr
