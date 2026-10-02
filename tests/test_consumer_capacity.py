"""Static liveness, heterogeneous-card admission and C++/Python contract parity."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import subprocess
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from flow_topology import ranked_graph
from flow_protocol import make_continuous_packet
from tools.online_bench.capacity import Capacities, Chunks, MIB, packet_geometry, plan



@pytest.fixture(scope='module')
def capacity_probe(tmp_path_factory):
    binary = tmp_path_factory.mktemp('capacity-probe')/'probe'
    subprocess.run(['c++','-std=c++17','-O1',str(ROOT/'tests/consumer_capacity_probe.cpp'),'-o',str(binary)],check=True)
    return binary


def geometry(memory='attention',width=64,devices=3,training=True):
    p = make_continuous_packet(graph=ranked_graph(layers=4,region_width=4,fanout=3,local_span=4),
                              memory=memory,width=width,batch=2,tokens=2,vocab=257,clear=False)
    return p,packet_geometry(p,windows=2,payload=4,training=training,adamw=True,diagnostics=True,devices=devices)


def cpp_plan(binary,g,c,chunks,budgets,aggressive,logical_batch=None,automatic=False):
    values = [g.width,g.batch,g.vocab,g.windows,g.payload,int(g.attention),int(g.training),int(g.adamw),int(g.diagnostics),
              g.regions,g.devices,int(g.locality),g.sample_chunks,g.context_bytes,len(g.sources),len(g.edges),int(aggressive),*g.sources,*g.slots,
              *(x for e in g.edges for x in e),*asdict(c).values(),*asdict(chunks).values(),*budgets]
    if logical_batch is not None:
        values.extend((logical_batch,int(automatic)))
    done = subprocess.run([str(binary)],input=' '.join(map(str,values)),text=True,capture_output=True,timeout=10)
    return json.loads(done.stdout) if done.returncode==0 else done.stderr


def test_cross_language_shapes_and_parameter_inventory(capacity_probe):
    rng = random.Random(42)
    for i in range(24):
        p,g = geometry('attention' if i%2 else 'add',rng.choice([4,16,128,2048]),rng.choice([1,2,3,8]),bool(i%3))
        g = replace(g,payload=2 if i%4 else 4,windows=1+i%3,locality=bool(i%2),diagnostics=g.training or i%3==0,
                    sample_chunks=3 if i%5 else 1,context_bytes=64*MIB if i%3 else 0)
        caps = Capacities(trace=512,kv_trace=1024)
        chunks = Chunks(head=256)
        budgets = [(64-j)*1024**3 for j in range(g.devices)]
        python = plan(g,caps,chunks,budgets,bool(i%2))
        assert cpp_plan(capacity_probe,g,caps,chunks,budgets,bool(i%2)) == python
        assert sum(python['canonical_elements'])+2*g.vocab*g.width == p['counts']['parameters']
        assert all((c['components']['retained_pack_workspace']>0)==g.training for c in python['devices'])


def test_weakest_card_and_fixed_state_refusal(capacity_probe):
    _,g = geometry(width=512)
    caps,chunks = Capacities(trace=128,kv_trace=256),Chunks(head=512)
    full = plan(g,caps,chunks,[64*1024**3]*g.devices,True)
    minimum = plan(g,caps,Chunks(1,1,1,1,1,1,1),[64*1024**3]*g.devices,True)
    index = 1
    target = (minimum['devices'][index]['estimated_peak_bytes']+full['devices'][index]['estimated_peak_bytes'])//2
    budgets = [64*1024**3]*g.devices;budgets[index] = (target+128*MIB)*10//9
    reduced = plan(g,caps,chunks,budgets,True)
    assert reduced['physical_reductions']>0
    assert cpp_plan(capacity_probe,g,caps,chunks,budgets,True) == reduced
    assert reduced['full_owners']==full['full_owners']
    assert all(reduced['effective_chunks'][k]<=v for k,v in full['effective_chunks'].items())
    budgets[index] = 1
    with pytest.raises(ValueError,match='minimum physical rows'):
        plan(g,caps,chunks,budgets,True)
    assert 'minimum physical rows' in cpp_plan(capacity_probe,g,caps,chunks,budgets,True)


def test_continuation_retention_and_precision_costs():
    _,g = geometry()
    caps,chunks = Capacities(),Chunks(head=1024)
    one = plan(replace(g,windows=1),caps,chunks,[64*1024**3]*3,True)
    two = plan(g,caps,chunks,[64*1024**3]*3,True)
    half = plan(replace(g,payload=2),caps,chunks,[64*1024**3]*3,True)
    for a,b,h in zip(one['devices'],two['devices'],half['devices']):
        assert b['components']['retained'] > a['components']['retained']
        assert h['components']['graph_optimizer'] == b['components']['graph_optimizer']
        assert h['components']['parameters'] < b['components']['parameters']
    assert one['canonical_elements']==two['canonical_elements']==half['canonical_elements']
    # Metadata plans do not initialize Torch or consume any reference trajectory.
    assert plan(replace(g,attention=False),caps,chunks,[64*1024**3]*3,True)['schema']=='tide-consumer-capacity-v1'


def test_overflow_refusal(capacity_probe):
    _,g = geometry(width=4)
    g = replace(g,width=2**62)
    with pytest.raises(ValueError,match='overflow'):
        plan(g,Capacities(),Chunks(),[64*1024**3]*3)
    assert 'overflow' in cpp_plan(capacity_probe,g,Capacities(),Chunks(),[64*1024**3]*3,False)


def test_compact_pool_charges_declared_cap_and_pack_workspace(capacity_probe):
    _,g = geometry(width=128)
    g = replace(g,sample_chunks=64)
    caps,chunks = Capacities(trace=512,kv_trace=4096),Chunks()
    dense = plan(g,caps,chunks,[64*1024**3]*3,True)
    bounded = plan(replace(g,context_bytes=32*MIB),caps,chunks,[64*1024**3]*3,True)
    assert cpp_plan(capacity_probe,replace(g,context_bytes=32*MIB),caps,chunks,[64*1024**3]*3,True)==bounded
    for a,b in zip(dense['devices'],bounded['devices']):
        assert b['components']['saved_contexts']==32*MIB < a['components']['saved_contexts']
        assert b['components']['context_pack_workspace']>0
        assert b['estimated_peak_bytes']<a['estimated_peak_bytes']


def test_offline_sample_pool_options(tmp_path):
    packet,_ = geometry()
    path = tmp_path/'packet.json';path.write_text(json.dumps(packet))
    command = [sys.executable,str(ROOT/'scripts/plan_execution_flow.py'),'--packet',str(path),
               '--devices','3','--device-memory-bytes',str(64*1024**3),
               '--sample-chunk-rows','1','--resident-context-bytes',str(MIB)]
    result = json.loads(subprocess.check_output(command,text=True))
    assert result['state']=='planned'
    for card in result['memory_admission']['devices']:
        assert 0 < card['components']['saved_contexts'] <= MIB
        assert card['components']['context_pack_workspace'] > 0
    bad = subprocess.run(command+['--resident-context-bytes','-1'],text=True,capture_output=True)
    assert bad.returncode==2 and 'nonnegative int64' in bad.stderr


def test_automatic_samples_preserve_logical_storage_and_refuse_at_one(capacity_probe):
    from tools.online_bench.capacity import plan_samples, MemoryRefusal
    _,g=geometry(width=16)
    g=replace(g,batch=17,context_bytes=64*MIB)
    caps=Capacities(kv=8192,trace=512,kv_trace=512)
    one=Chunks(1,1,1,1,1,1,1)
    larger=plan(g,caps,one,[64*1024**3]*3,True)
    smaller=plan(replace(g,batch=9,sample_chunks=2),caps,one,[64*1024**3]*3,True)
    weak=1
    a,b=(x['devices'][weak]['estimated_peak_bytes'] for x in (larger,smaller))
    assert b<a
    budgets=[64*1024**3]*3;budgets[weak]=((a+b)//2+128*MIB)*10//9
    with pytest.raises(MemoryRefusal):
        plan_samples(g,caps,one,budgets,True,17,False)
    result=plan_samples(g,caps,Chunks(),budgets,True,17,True)
    assert cpp_plan(capacity_probe,g,caps,Chunks(),budgets,True,17,True)==result
    assert result['sample_admission']==dict(logical_batch=17,effective_sample_rows=9,
        physical_chunks=2,attempted_sample_rows=[17,9],policy='halve_on_memory_refusal')
    assert result['full_owners']==larger['full_owners']
    assert result['canonical_elements']==larger['canonical_elements']
    assert all(d['components']['saved_contexts']>0 and d['components']['gradient_accumulation']>0 for d in result['devices'])
    with pytest.raises(MemoryRefusal,match='minimum physical rows'):
        plan_samples(g,caps,one,[1]*3,True,17,True)
    assert 'minimum physical rows' in cpp_plan(capacity_probe,g,caps,one,[1]*3,True,17,True)
    with pytest.raises(ValueError,match='overflow'):
        plan_samples(replace(g,width=2**62),caps,one,budgets,True,17,True)


def test_automatic_sample_cli_and_boolean_contract(tmp_path):
    from tools.online_bench.capacity import plan_samples
    packet,g=geometry()
    p=tmp_path/'packet.json';p.write_text(json.dumps(packet))
    result=json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/plan_execution_flow.py'),
        '--packet',str(p),'--devices','3','--device-memory-bytes',str(64*1024**3),
        '--auto-sample-chunks','--sample-chunk-rows','1'],text=True))
    assert result['memory_admission']['sample_admission']['attempted_sample_rows']==[1]
    for value in (1,None,'true'):
        with pytest.raises(ValueError,match='boolean'):
            plan_samples(g,Capacities(),Chunks(),[64*1024**3]*3,True,g.batch,value)
