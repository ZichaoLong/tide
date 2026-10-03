"""Capacity-driven physical splitting must preserve complete continued updates."""
import json
import os
from dataclasses import asdict
import pytest
import torch
from test_online_consumer import packet
from test_online_consumer_npu import target
from test_online_resident_consumer import standalone
from online_consumer_support import observer,same
from tidegraph import ResidentLimits,ResidentPlacement
from tools.online_bench.host import run
from tools.online_bench.head_budget import head_budget
from tools.online_bench.capacity import Capacities,Chunks,MIB,packet_geometry,plan


@pytest.mark.parametrize('capacity_pressure',['operators','owners'])
@pytest.mark.parametrize('implementation',['native','libtorch'])
@pytest.mark.parametrize('memory',['add','attention'])
@pytest.mark.parametrize('payload_dtype',['float32','float16'])
def test_complete_training_with_automatic_splitting(implementation,memory,payload_dtype,capacity_pressure,tmp_path):
    d=target();p=packet(memory);half=payload_dtype=='float16';opt='adamw' if memory=='attention' else 'sgd'
    schedule='prefill' if memory=='attention' else 'streaming'
    caps=Capacities(outputs=16,trace=512,kv_trace=512)
    head=head_budget(16,4,7,2 if half else 4,True,4*1024**3,True)
    chunks=Chunks(head=head.rows)
    g=packet_geometry(p,windows=2,payload=2 if half else 4,training=True,adamw=opt=='adamw',diagnostics=True,devices=2)
    original=plan(g,caps,chunks,[64*1024**3]*2,True)
    minimum=plan(g,caps,Chunks(1,1,1,1,1,1,1),[64*1024**3]*2,True)
    target_peak=(max(c['estimated_peak_bytes'] for c in original['devices'])+max(c['estimated_peak_bytes'] for c in minimum['devices']))//2
    if capacity_pressure=='owners':
        target_peak=max(c['estimated_peak_bytes'] for c in minimum['devices'])-128
    budget=(target_peak+128*MIB)*10//9
    expected_plan=plan(g,caps,chunks,[budget]*2,True)
    if capacity_pressure=='owners':
        assert expected_plan['owner_moves']>0 and expected_plan['full_owners']!=minimum['full_owners']
    else:
        assert expected_plan['physical_reductions']>0 and expected_plan['owner_moves']==0
    expected=[];actual=[]
    kw=dict(family='timed-dag',training=True,optimizer=opt,steps=2,warmup=0,windows_per_step=2,diagnostics=True)
    reference=run(p,implementation='python',device='cpu',schedule='streaming',observer=observer(expected),**kw)
    if implementation=='libtorch':
        candidate,actual=standalone(p,d,'timed-dag',schedule,True,opt,tmp_path,devices=2,dtype_name=payload_dtype,
            extra=['--chunk-policy=aggressive','--resident-outputs=16','--resident-trace=512',
                   '--resident-kv-trace-rows=512','--device-memory-bytes='+str(budget)])
    else:
        candidate=run(p,implementation='native',device=d,schedule=schedule,preset='resident',dtype=payload_dtype,
            observer=observer(actual),native_library=os.environ['TIDE_BUILD_DIR'],resident_library=os.environ['TIDE_RESIDENT_LIBRARY'],
            resident_limits=ResidentLimits(outputs=16,trace=512,kv_trace_rows=512,workspace_bytes=512*MIB,chunk_policy='aggressive'),
            resident_placement=ResidentPlacement(devices=(str(d),f'npu:{d.index+1}')),device_memory_bytes=budget,**kw)
    tol=dict(atol=2e-3,rtol=2e-2) if half else dict(atol=1e-6,rtol=1e-5)
    same(actual,expected,**tol)
    torch.testing.assert_close(torch.tensor(candidate['losses']),torch.tensor(reference['losses']),**tol)
    assert candidate['outputs']==reference['outputs'] and candidate['final_cut']==reference['final_cut']
    capacity=candidate['memory_admission']
    for key in expected_plan:
        assert capacity[key]==expected_plan[key]
    assert capacity['allocator_within_estimate']
    assert candidate['head_memory']['rows']==capacity['effective_chunks']['head']
    (tmp_path/'capacity-result.json').write_text(json.dumps(dict(candidate=candidate,observations=actual)))


def test_refusal_precedes_parameter_construction(monkeypatch):
    import tools.online_bench.resident as resident
    def forbidden(*args,**kwargs):
        raise AssertionError('model construction ran after impossible admission')
    monkeypatch.setattr(resident,'runtime_for',forbidden)
    with pytest.raises(ValueError,match='minimum physical rows'):
        run(packet(),family='timed-dag',implementation='native',device=target(),preset='resident',device_memory_bytes=1,
            native_library=os.environ['TIDE_BUILD_DIR'],resident_library=os.environ['TIDE_RESIDENT_LIBRARY'])
