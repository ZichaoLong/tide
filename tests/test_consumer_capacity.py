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
from tools.online_bench.capacity import Capacities, Chunks, MIB, packet_geometry, plan, envelope



@pytest.fixture(scope='module')
def capacity_probe(tmp_path_factory):
    binary = tmp_path_factory.mktemp('capacity-probe')/'probe'
    subprocess.run(['c++','-std=c++17','-O1',str(ROOT/'tests/consumer_capacity_probe.cpp'),'-o',str(binary)],check=True)
    return binary


def geometry(memory='attention',width=64,devices=3,training=True):
    p = make_continuous_packet(graph=ranked_graph(layers=4,region_width=4,fanout=3,local_span=4),
                              memory=memory,width=width,batch=2,tokens=2,vocab=257,clear=False)
    return p,packet_geometry(p,windows=2,payload=4,training=training,adamw=True,diagnostics=True,devices=devices)


def cpp_plan(binary,g,c,chunks,budgets,aggressive,logical_batch=None,automatic=False,owners=None):
    values = [g.width,g.batch,g.vocab,g.windows,g.payload,int(g.attention),int(g.training),int(g.adamw),int(g.diagnostics),
              g.regions,g.devices,int(g.locality),g.sample_chunks,g.context_bytes,len(g.sources),len(g.edges),int(aggressive),*g.sources,*g.slots,
              *(x for e in g.edges for x in e),*asdict(c).values(),*asdict(chunks).values(),*budgets]
    if logical_batch is not None or owners is not None:
        logical_batch=g.batch if logical_batch is None else logical_batch
        values.extend((logical_batch,int(automatic)))
        if owners is not None:
            values.extend((len(owners),*owners))
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


def test_aggressive_preserves_nonlimiting_operator_batches(capacity_probe):
    _,g = geometry(memory='add',width=512)
    caps = Capacities(trace=128,kv_trace=256)
    requested = Chunks(head=4)
    full = plan(g,caps,requested,[64*1024**3]*g.devices,True)
    small = plan(g,caps,replace(requested,reverse=2),[64*1024**3]*g.devices,True)
    weak = 1
    target = (full['devices'][weak]['estimated_peak_bytes']+small['devices'][weak]['estimated_peak_bytes'])//2
    budgets = [64*1024**3]*g.devices;budgets[weak]=(target+128*MIB)*10//9+1
    got = plan(g,caps,requested,budgets,True)
    assert cpp_plan(capacity_probe,g,caps,requested,budgets,True)==got
    assert got['row_selection']=='greedy_peak_excess' and got['effective_chunks']['reverse']<requested.reverse
    # Tiny Full/aggregate work and inactive Attention do not need to shrink
    # when the reverse scratch alone can resolve this card's memory excess.
    for key in ('full','aggregate','attention','keys','head'):
        assert got['effective_chunks'][key]==asdict(requested)[key]
    assert all(d['estimated_peak_bytes']<=d['usable_bytes'] for d in got['devices'])


def test_constrained_plans_keep_original_envelope_and_language_parity(capacity_probe):
    rng=random.Random(109)
    for i in range(16):
        _,g=geometry('add' if i%2 else 'attention',rng.choice([16,128,512]),rng.choice([2,3]))
        g=replace(g,payload=2 if i%3 else 4,windows=1+i%3,sample_chunks=1+i%4)
        caps=Capacities(trace=128,kv_trace=256)
        requested=Chunks(full=11,emission=9,aggregate=7,attention=5,keys=33,reverse=13,head=17)
        weak=i%g.devices
        for aggressive in (False,True):
            full=plan(g,caps,requested,[64*1024**3]*g.devices,aggressive)
            minimum=plan(g,caps,Chunks(1,1,1,1,1,1,1),[64*1024**3]*g.devices,aggressive)
            target=(minimum['devices'][weak]['estimated_peak_bytes']+full['devices'][weak]['estimated_peak_bytes'])//2
            denominator=10 if aggressive else 4
            budgets=[64*1024**3]*g.devices
            budgets[weak]=(target+128*MIB)*denominator//(denominator-1)+1
            got=plan(g,caps,requested,budgets,aggressive)
            assert cpp_plan(capacity_probe,g,caps,requested,budgets,aggressive)==got
            assert got['physical_reductions']>0
            assert got['full_owners']==full['full_owners'] and got['canonical_elements']==full['canonical_elements']
            assert all(1<=value<=asdict(requested)[key] for key,value in got['effective_chunks'].items())
            exact=envelope(g,caps,Chunks(**got['effective_chunks']),got['full_owners'],got['canonical_elements'],reuse_parameter_gradients=aggressive)
            for card,original in zip(got['devices'],exact):
                assert card['estimated_peak_bytes']==original['estimated_peak_bytes']<=card['usable_bytes']
                assert card['phases']==original['phases'] and card['components']==original['components']
            if not aggressive:
                assert got['row_selection']=='joint_halving'
                assert got['effective_chunks']=={k:max(1,v//2**got['physical_reductions']) for k,v in asdict(requested).items()}


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


@pytest.mark.parametrize('memory',['add','attention'])
@pytest.mark.parametrize('aggressive',[False,True])
@pytest.mark.parametrize('devices',[1,3])
def test_attention_gradient_charge_matches_materialized_model(memory,aggressive,devices,capacity_probe):
    from tools.online_bench.fixture import build_model
    packet,g=geometry(memory,width=16,devices=devices)
    _,model,_,_=build_model(packet)
    result=plan(g,Capacities(),Chunks(),[64*1024**3]*g.devices,aggressive)
    assert cpp_plan(capacity_probe,g,Capacities(),Chunks(),[64*1024**3]*g.devices,aggressive)==result
    copies=1 if aggressive and devices>1 else g.windows
    actual=[0]*g.devices
    projections=[0]*g.devices
    for i,node in enumerate(model.nodes):
        for name,value in node.extra.items():
            if name in ('fiber_qkv','fiber_out'):
                assert value.requires_grad
                actual[result['state_owners'][i]]+=copies*value.numel()*4  # FP32 adjoints.
            if name.startswith(('emit_w_','emit_b_')):
                projections[result['full_owners'][i]]+=copies*value.numel()*4
    charged=[d['components']['attention_parameter_gradients'] for d in result['devices']]
    assert charged==actual
    assert all(d['components']['projection_parameter_gradients']>=v for d,v in zip(result['devices'],projections))
    assert (sum(charged)>0)==(memory=='attention')
    assert all(d['components']['physical_and_canonical_gradients']>v for d,v in zip(result['devices'],charged))


@pytest.mark.parametrize('memory',['add','attention'])
@pytest.mark.parametrize('payload',[2,4])
def test_shared_training_storage_covers_materialized_inventory(memory,payload,capacity_probe):
    from tools.online_bench.fixture import build_model
    import torch
    packet,g=geometry(memory,width=16)
    g=replace(g,payload=payload,sample_chunks=3)
    _,model,_,_=build_model(packet,dtype=torch.float16 if payload==2 else torch.float32)
    shared_names={'fiber_qkv','fiber_out','fiber_qkv_bias','fiber_out_bias','fiber_decay','fiber_pool'}
    inventory=[0]*g.devices
    previous=None
    for windows in (1,2,3):
        current=plan(replace(g,windows=windows),Capacities(),Chunks(),[64*1024**3]*g.devices,True)
        assert cpp_plan(capacity_probe,replace(g,windows=windows),Capacities(),Chunks(),[64*1024**3]*g.devices,True)==current
        if windows==1:
            for i,node in enumerate(model.nodes):
                for name,value in node.extra.items():
                    if name in shared_names:
                        # Fiber pooling banks retain FP32 even for FP16 payload.
                        inventory[current['state_owners'][i]]+=value.numel()*(4 if name=='fiber_pool' else payload)
        for card,actual in zip(current['devices'],inventory):
            c=card['components']
            assert c['retained_attention_parameters']>=actual
            assert (c['retained_attention_parameters']>0)==(memory=='attention')
            assert 0<c['gradient_accumulation_live']<c['gradient_accumulation']
            if previous:
                old=previous['devices'][card['index']]['components']
                for name in ('retained_attention_parameters','gradient_accumulation','gradient_accumulation_live',
                             'attention_parameter_gradients','projection_parameter_gradients'):
                    assert c[name]==old[name]
                assert c['retained']>old['retained']  # State/KV/journals still grow.
        previous=current
    for variant in (replace(g,training=False),replace(g,sample_chunks=1)):
        result=plan(variant,Capacities(),Chunks(),[64*1024**3]*g.devices,True)
        assert all(d['components']['gradient_accumulation_live']==0 for d in result['devices'])
        if not variant.training:
            assert all(d['components']['retained_attention_parameters']==0 for d in result['devices'])


@pytest.mark.parametrize('memory',['add','attention'])
@pytest.mark.parametrize('payload',[2,4])
def test_private_bank_liveness_covers_model_and_keeps_legacy_copies(memory,payload,capacity_probe):
    from tools.online_bench.fixture import build_model
    import torch
    packet,g=geometry(memory,width=16)
    _,model,_,_=build_model(packet,dtype=torch.float16 if payload==2 else torch.float32)
    # Independent inventory: actual forward-bank parameter fields, before padding.
    total=0
    for node in model.nodes:
        for name,value in node.extra.items():
            if name.startswith(('emit_w_','emit_b_','fiber_')):
                total+=value.numel()*(4 if name=='fiber_pool' else payload)
    for devices in (1,3):
        current=replace(g,payload=payload,devices=devices,windows=1)
        records=[]
        for aggressive in (False,True):
            r=plan(current,Capacities(),Chunks(1,1,1,1,1,1,1),[64*1024**3]*devices,aggressive)
            assert cpp_plan(capacity_probe,current,Capacities(),Chunks(1,1,1,1,1,1,1),[64*1024**3]*devices,aggressive)==r
            copies=sum(d['components']['retained_parameter_copies'] for d in r['devices'])
            borrowed=sum(d['components']['borrowed_parameter_banks'] for d in r['devices'])
            assert copies+borrowed>=total>0
            assert (copies==0)==(aggressive and devices>1)
            assert (borrowed>0)==(aggressive and devices>1)
            records.append(r)
        for old,new in zip(records[0]['devices'],records[1]['devices']):
            saving=new['components']['borrowed_parameter_banks']
            assert old['phases']['forward_loss']-new['phases']['forward_loss']==saving
            assert old['phases']['construction']==new['phases']['construction']
            assert old['phases']['optimizer']==new['phases']['optimizer']
            for name in ('state_and_kv','journals','roots_and_consumer_gradients','physical_and_canonical_gradients','reverse_workspace'):
                assert old['components'][name]==new['components'][name]


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
    # Every card is constrained: moving owners cannot create the missing total
    # KV/state capacity. A separately tested weak-card-only case can rebalance.
    budgets=[((x['estimated_peak_bytes']+y['estimated_peak_bytes'])//2+128*MIB)*10//9
             for x,y in zip(larger['devices'],smaller['devices'])]
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


def test_bounded_owner_moves_resolve_weak_card_and_preserve_explicit_maps(capacity_probe):
    from tools.online_bench.capacity import MemoryRefusal
    _,g=geometry(width=16)
    g=replace(g,batch=17,context_bytes=64*MIB)
    caps=Capacities(kv=8192,trace=512,kv_trace=512);one=Chunks(1,1,1,1,1,1,1)
    initial=plan(g,caps,one,[64*1024**3]*3,True)
    smaller=plan(replace(g,batch=9,sample_chunks=2),caps,one,[64*1024**3]*3,True)
    a,b=(p['devices'][1]['estimated_peak_bytes'] for p in (initial,smaller))
    budgets=[64*1024**3]*3;budgets[1]=((a+b)//2+128*MIB)*10//9
    with pytest.raises(MemoryRefusal):
        plan(g,caps,one,budgets,True,initial['full_owners'],initial['state_owners'])
    result=plan(g,caps,one,budgets,True)
    assert cpp_plan(capacity_probe,g,caps,one,budgets,True)==result
    assert plan(g,caps,one,budgets,True,(),())==result
    assert plan(g,caps,one,budgets,True,[],[])==result
    assert 0<result['owner_moves']<=2*len(result['full_owners'])
    assert 0<result['owner_evaluations']<=4096
    assert result['full_owners']==result['state_owners']!=initial['full_owners']
    assert set(result['full_owners'])==set(range(g.devices))
    assert result['canonical_elements']==initial['canonical_elements']
    assert all(d['estimated_peak_bytes']<=d['usable_bytes'] for d in result['devices'])
    assert result['effective_chunks']==asdict(one) and result['physical_reductions']==0
    # Explicitly providing the accepted map remains a valid non-searching call.
    fixed=plan(g,caps,one,budgets,True,result['full_owners'],result['state_owners'])
    assert fixed['devices']==result['devices'] and fixed['owner_moves']==fixed['owner_evaluations']==0


def test_owner_balance_language_parity_across_shapes(capacity_probe):
    rng=random.Random(20261003)
    for i in range(16):
        _,g=geometry('attention' if i%2 else 'add',rng.choice([4,16,128]),rng.choice([2,3]))
        g=replace(g,payload=2 if i%3 else 4,windows=1+i%3,locality=bool(i%2))
        caps=Capacities(queue=128,arrivals=128,outputs=64,trace=128,kv_trace=256)
        one=Chunks(1,1,1,1,1,1,1)
        initial=plan(g,caps,one,[64*1024**3]*g.devices,True)
        target=max(d['estimated_peak_bytes'] for d in initial['devices'])-128
        budgets=[(target+128*MIB)*10//9]*g.devices
        current=plan(g,caps,one,budgets,True)
        assert cpp_plan(capacity_probe,g,caps,one,budgets,True)==current
        assert 0<current['owner_moves']<=2*len(g.sources)+4
        assert 0<current['owner_evaluations']<=4096
        assert all(d['estimated_peak_bytes']<=d['usable_bytes'] for d in current['devices'])
