"""FP16 consumers against independent CPU schedules and master/slot trajectories."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from tools.online_bench.host import run
from tidegraph.precision import FP32MasterOptimizer
from flow_protocol import native_text
from online_consumer_support import observer,same
from test_online_consumer import packet


def target():
    # Plugin registration precedes the first independent CPU backward.
    from tidegraph.runtime import resolve_device
    device,_=resolve_device(os.environ.get('TIDE_EAGER_PRECISION_DEVICE','cpu'))
    if device.type!='cpu':
        assert device.index is not None and getattr(torch,device.type).device_count()>=device.index+2
    return device


def tensor(value):
    return None if value is None else dict(shape=list(value.shape),values=value.detach().cpu().double().flatten().tolist())


def standalone_binary():
    value=os.environ.get('TIDE_ONLINE_BINARY')
    if not value:pytest.skip('standalone consumer not explicitly selected')
    assert Path(value).is_file(),'explicit standalone consumer unavailable'
    return value


@pytest.mark.parametrize('kind',['sgd','adamw'])
def test_master_alias_none_zero_slots_and_nonfinite(kind):
    device=target()
    binary=Path(standalone_binary()).with_name('tidegraph-eager-precision-probe')
    done=subprocess.run([str(binary),str(device),kind],capture_output=True,text=True,timeout=120)
    assert done.returncode==0,done.stdout+done.stderr
    rows=[json.loads(s) for s in done.stdout.splitlines() if s.startswith('{')]
    assert rows[-1]==dict(nonfinite_rejected=True,aliases_preserved=True)
    names=sorted(['active','zero','unused','intermittent'])
    payload={name:torch.tensor([1.,-.5],dtype=torch.float16,requires_grad=True) for name in names}
    opt=FP32MasterOptimizer(payload.values(),optimizer=kind,loss_scale=128,lr=.01,weight_decay=.1,foreach=False,
        **(dict(momentum=.25) if kind=='sgd' else dict(eps=1e-6)))
    expected=[]
    for step in range(3):
        opt.zero_grad()
        for _ in range(2):
            loss=payload['active'].float().sum()*((step+1)*.03125)+payload['zero'].float().sum()*0
            if step!=1:loss=loss+payload['intermittent'].float().sum()*.0625
            opt.backward(loss*.5)
        opt.step();owners={}
        for name,master in zip(names,opt.masters):
            state=opt.state.get(master,{})
            slot=None if not state else dict(step=int(state.get('step',0)),
                first=tensor(state.get('momentum_buffer',state.get('exp_avg'))),second=tensor(state.get('exp_avg_sq')))
            owners[name]=dict(payload=tensor(payload[name]),gradient=tensor(payload[name].grad),master=tensor(master),slot=slot)
        expected.append(dict(step=step,owners=owners))
    same(rows[:-1],expected,atol=1e-7,rtol=1e-6)
    assert all(row['owners']['unused']['gradient'] is None and row['owners']['unused']['slot'] is None for row in rows[:-1])
    assert rows[1]['owners']['intermittent']['gradient'] is None
    assert rows[0]['owners']['zero']['slot'] is not None


@pytest.mark.parametrize('memory',['add','attention'])
@pytest.mark.parametrize('family',['pdg','timed-dag','settle'])
@pytest.mark.parametrize('schedule',['streaming','prefill'])
@pytest.mark.parametrize('implementation',['python','native','libtorch'])
def test_complete_half_consumer(memory,family,schedule,implementation,tmp_path):
    device=target();accelerated=device.type!='cpu';p=packet(memory,delayed=family=='timed-dag')
    expected=[];actual=[];kind='sgd' if schedule=='streaming' else 'adamw'
    common=dict(family=family,dtype='float16',training=True,optimizer=kind,steps=2,warmup=0,
        windows_per_step=2,sample_chunk_rows=1,loss_scale=128,diagnostics=True)
    reference=run(p,implementation='python',device='cpu',schedule='streaming',observer=observer(expected),**common)
    preset={'pdg':'mixed-a','timed-dag':'mixed-b','settle':'mixed-c'}[family] if accelerated else 'cpu'
    count=2 if accelerated else 1
    if implementation=='libtorch':
        path=tmp_path/'packet.txt';path.write_text(native_text(p));out=tmp_path/'consumer'
        command=[standalone_binary(),'--device='+str(device),'--dtype=float16',
            '--packet='+str(path),'--output-dir='+str(out),'--family='+family,'--preset='+preset,
            '--schedule='+schedule,'--training','--optimizer='+kind,'--steps=2','--warmup=0',
            '--windows-per-step=2','--sample-chunk-rows=1','--loss-scale=128','--diagnostics',
            '--devices='+str(count),'--workers=2','--packed-sources','--batch-next']
        done=subprocess.run(command,capture_output=True,text=True,timeout=120)
        assert done.returncode==0,done.stdout+done.stderr
        actual=[json.loads(s) for s in (out/'diagnostics.jsonl').read_text().splitlines()]
        measured=json.loads((out/'result.json').read_text())
    else:
        measured=run(p,implementation=implementation,device=device,schedule=schedule,preset=preset,
            devices=count,observer=observer(actual),workers=2 if implementation=='native' else 1,
            packed_sources=implementation=='native',batch_next=implementation=='native',
            native_library=os.environ['TIDE_BUILD_DIR'] if implementation=='native' else None,**common)
    same(actual,expected,atol=2e-3,rtol=2e-2)
    torch.testing.assert_close(torch.tensor(measured['losses']),torch.tensor(reference['losses']),atol=2e-3,rtol=2e-2)
    for field in ['parameters','outputs','final_cut','batch_execution','precision']:
        assert measured[field]==reference[field]
    assert measured['precision']==dict(payload='float16',loss='float32',optimizer_masters='float32',
        gradient_accumulation='payload',loss_scale=128.)
    assert measured['memory_admission']['allocator_within_estimate']
    assert all(d['components']['master_parameters']>0 for d in measured['memory_admission']['devices'])


@pytest.mark.parametrize('implementation',['python','native','libtorch'])
def test_half_scale_unified_cli(implementation,tmp_path):
    device=target();accelerated=device.type!='cpu';path=tmp_path/'packet.json';path.write_text(json.dumps(packet()))
    out=tmp_path/'consumer'
    command=[sys.executable,str(ROOT/'scripts/run_execution_flow.py'),'--packet',str(path),'--output-dir',str(out),
        '--device',str(device),'--dtype','float16','--family','timed-dag','--preset','mixed-c' if accelerated else 'cpu',
        '--schedule','prefill','--implementation',implementation,'--training','--loss-scale','32',
        '--steps','1','--warmup','0','--devices','2' if accelerated else '1']
    if implementation=='libtorch':command+=['--native-binary',standalone_binary()]
    elif implementation=='native':command+=['--native-library',os.environ['TIDE_BUILD_DIR']]
    done=subprocess.run(command,capture_output=True,text=True,timeout=120)
    assert done.returncode==0,done.stdout+done.stderr
    result=json.loads((out/'result.json').read_text());assert result['precision']['loss_scale']==32
    assert result['precision']['optimizer_masters']=='float32'


@pytest.mark.parametrize('scale',[0,-1,float('nan'),float('inf'),True])
def test_invalid_scale_precedes_model_construction(scale,monkeypatch):
    import tools.online_bench.host as host
    def forbidden(*args,**kwargs):raise AssertionError('model construction entered')
    monkeypatch.setattr(host,'build_model',forbidden)
    with pytest.raises(ValueError,match='loss-scale'):
        run(packet(),family='timed-dag',implementation='python',device='cpu',dtype='float16',training=True,loss_scale=scale)


@pytest.mark.parametrize('dtype_name,training,preset',[('float32',True,'cpu'),('float16',False,'cpu'),('float16',True,'resident')])
def test_nonunit_scale_requires_eager_half_training(dtype_name,training,preset):
    with pytest.raises(ValueError,match='loss-scale'):
        run(packet(),family='timed-dag',implementation='python',device='cpu',dtype=dtype_name,training=training,preset=preset,loss_scale=2)
