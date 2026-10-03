"""Replay a static placement across batch geometries without weakening admission."""
import argparse
import json
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import pytest
from test_consumer_capacity import geometry,cpp_plan,capacity_probe
from tools.online_bench.capacity import Capacities,Chunks,MIB,plan,plan_samples,MemoryRefusal
from flow_resident_options import add_arguments,native_arguments,python_arguments,owner_map
ROOT=Path(__file__).resolve().parents[1]


def test_explicit_map_replay_has_language_parity_and_never_rebalances(capacity_probe):
    _,g=geometry(width=16);g=replace(g,batch=17,context_bytes=64*MIB)
    caps=Capacities(kv=8192,trace=512,kv_trace=512);one=Chunks(1,1,1,1,1,1,1)
    first=plan(g,caps,one,[64*1024**3]*3,True)
    small=plan(replace(g,batch=9,sample_chunks=2),caps,one,[64*1024**3]*3,True)
    target=(first['devices'][1]['estimated_peak_bytes']+small['devices'][1]['estimated_peak_bytes'])//2
    budgets=[64*1024**3]*3;budgets[1]=(target+128*MIB)*10//9
    chosen=plan(g,caps,one,budgets,True);assert chosen['owner_moves']>0
    for batch in (2,9,17):
        current=replace(g,batch=batch,sample_chunks=(17-1)//batch+1)
        owners=chosen['full_owners']
        result=plan_samples(current,caps,one,budgets,True,17,True,tuple(owners),tuple(owners))
        assert cpp_plan(capacity_probe,current,caps,one,budgets,True,17,True,owners)==result
        assert result['full_owners']==result['state_owners']==owners
        assert result['owner_selection']=='explicit' and result['owner_moves']==result['owner_evaluations']==0
        assert all(d['estimated_peak_bytes']<=d['usable_bytes'] for d in result['devices'])
    owners=first['full_owners']
    with pytest.raises(MemoryRefusal):plan(g,caps,one,budgets,True,owners,owners)
    assert 'minimum physical rows' in cpp_plan(capacity_probe,g,caps,one,budgets,True,owners=owners)


@pytest.mark.parametrize('variant',['short','unused','outside','negative'])
def test_bad_joint_maps_fail_in_both_planners(variant,capacity_probe):
    _,g=geometry();owners=[i%g.devices for i in range(len(g.sources)+2)]
    if variant=='short':owners.pop()
    elif variant=='unused':owners=[0]*len(owners)
    else:owners[0]=g.devices if variant=='outside' else -1
    with pytest.raises(ValueError,match='invalid consumer owner map'):
        plan(g,Capacities(),Chunks(),[64*1024**3]*g.devices,True,owners,owners)
    assert 'invalid consumer owner map' in cpp_plan(capacity_probe,g,Capacities(),Chunks(),[64*1024**3]*g.devices,True,owners=owners)


def test_python_owner_indices_require_exact_integers():
    _,g=geometry();owners=[i%g.devices for i in range(len(g.sources)+2)]
    for value in (False,0.0):
        owners[0]=value
        with pytest.raises(ValueError,match='invalid consumer owner map'):
            plan(g,Capacities(),Chunks(),[64*1024**3]*g.devices,True,owners,owners)


@pytest.mark.parametrize('value',['','0,','0,,1','-1,0','16,0','0.0,1','True,0','000,1','0, 1'])
def test_owner_map_csv_rejects_invalid_indices(value):
    with pytest.raises(argparse.ArgumentTypeError):owner_map(value)


def test_offline_and_cli_adapters_replay_joint_map(tmp_path):
    packet,g=geometry(devices=2)
    owners=[(i+1)%2 for i in range(len(g.sources)+2)]
    spelling=','.join(map(str,owners))
    path=tmp_path/'packet.json';path.write_text(json.dumps(packet))
    command=[sys.executable,str(ROOT/'scripts/plan_execution_flow.py'),'--packet',str(path),
        '--devices','2','--device-memory-bytes',str(64*1024**3),'--owner-map',spelling]
    record=json.loads(subprocess.check_output(command,text=True))['memory_admission']
    assert record['full_owners']==record['state_owners']==owners
    assert record['owner_selection']=='explicit' and record['owner_moves']==record['owner_evaluations']==0
    parser=argparse.ArgumentParser();add_arguments(parser)
    args=parser.parse_args(['--devices','2','--owner-map',spelling]);args.preset='resident';args.windows_per_step=2
    assert '--owner-map='+spelling in native_arguments(args)
    # Only the already-resolved logical index is consumed here. Keep this
    # metadata test independent of importing a vendor backend on CPU.
    requested=python_arguments(args,argparse.Namespace(index=0))['resident_placement']
    assert list(requested.full_owners)==list(requested.state_owners)==owners
    assert requested.devices==('npu:0','npu:1')


def test_standalone_owner_csv_parser_is_available_without_device_work():
    binary=os.environ.get('TIDE_ONLINE_BINARY')
    if not binary:pytest.skip('standalone parser not explicitly selected')
    for value in ('','0,','0,,1','-1,0','16,0','0.0,1','True,0','000,1','0, 1'):
        r=subprocess.run([binary,'--help','--owner-map='+value],capture_output=True,text=True,timeout=15)
        assert r.returncode!=0 and 'invalid consumer owner map' in r.stderr
