"""One NPU parameter owner, independent sample continuations and one update."""
import json
import os
import pytest
import torch
from test_consumer_sample_chunks import CASES, compare_records
from test_online_consumer import make_continuous_packet, ranked_graph
from test_online_consumer_npu import target
from test_online_resident_consumer import standalone
from online_consumer_support import observer, same
from tidegraph import ResidentLimits, ResidentPlacement
from tools.online_bench.host import run


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("implementation", ["native", "libtorch"])
@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_resident_sample_chunks(case, implementation, dtype_name, tmp_path):
    family, memory, schedule, optimizer, clear, delayed, training = case
    device = target()
    pool = 64*1024**2 if memory=="attention" else 0
    policy = 'aggressive' if memory=='attention' else 'conservative'
    forward = ResidentLimits(workspace_bytes=512*1024**2,chunk_policy=policy)
    p = make_continuous_packet(graph=ranked_graph(layers=3,region_width=2,fanout=2,local_span=2,delayed=delayed),
        memory=memory,width=4,batch=5,tokens=2,vocab=7,clear=clear)
    expected, actual = [], []
    kw = dict(family=family,training=training,optimizer=optimizer,steps=2,warmup=0,windows_per_step=2,diagnostics=True)
    ref = run(p,implementation="python",device="cpu",schedule="streaming",observer=observer(expected),**kw)
    cards = 2 if schedule == "prefill" else 1
    if implementation == "libtorch":
        got, actual = standalone(p,device,family,schedule,training,optimizer,tmp_path,devices=cards,
                                dtype_name=dtype_name,extra=("--sample-chunk-rows=2", "--resident-context-bytes="+str(pool), "--chunk-policy="+policy))
    else:
        owners = ResidentPlacement(devices=(str(device),f"npu:{device.index+1}")) if cards==2 else None
        got = run(p,implementation="native",device=device,dtype=dtype_name,schedule=schedule,preset="resident",
            sample_chunk_rows=2,context_memory_bytes=pool,observer=observer(actual),native_library=os.environ["TIDE_BUILD_DIR"],
            resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],resident_limits=forward,resident_placement=owners,**kw)
    tol = dict(atol=2e-3,rtol=2e-2) if dtype_name=="float16" else dict(atol=1e-6,rtol=1e-5)
    compare_records(actual,expected,5,**tol,tensor_norm=dtype_name=="float16")
    if dtype_name=="float16":
        # Independently execute the whole batch at the same payload precision:
        # cross-dtype rounding and physical sample slicing are separate checks.
        whole=[]
        owners = ResidentPlacement(devices=(str(device),f"npu:{device.index+1}")) if cards==2 else None
        run(p,implementation="native",device=device,dtype=dtype_name,schedule=schedule,preset="resident",
            observer=observer(whole),native_library=os.environ["TIDE_BUILD_DIR"],
            resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],resident_limits=forward,resident_placement=owners,**kw)
        compare_records(actual,whole,5,**tol)
    for key in ("outputs","parameters","final_cut","input_tokens_per_step"):
        assert got[key] == ref[key]
    torch.testing.assert_close(torch.tensor(got["losses"]),torch.tensor(ref["losses"]),**tol)
    assert got['batch_execution'] == dict(logical_batch=5,requested_sample_chunk_rows=2,
                                         effective_sample_chunk_rows=2,physical_chunks=3)
    assert got['memory_admission']['allocator_within_estimate']
    for d in got['memory_admission']['devices']:
        c = d['components']
        assert c['saved_contexts'] == (min(pool,3*c['continuation_snapshot_bytes']) if pool else 3*c['continuation_snapshot_bytes']) > 0
        assert (c['context_pack_workspace']>0)==bool(pool)
        assert (c['gradient_accumulation'] > 0) == training
        assert (c['gradient_accumulation_live'] > 0) == training
        if training:
            assert c['gradient_accumulation_live']<c['gradient_accumulation']
    assert got['context_storage']['policy']==('compact' if pool else 'dense')
    assert got['context_storage']['requested_bytes_per_device']==pool
    assert all(x['peak_saved_bytes']<=x['budget_bytes'] for x in got['context_storage']['devices'])
    for step, stats in enumerate(got['statistics']):
        if training:
            # Measured retained parameter copies are shared by windows; dynamic
            # state/cache storage still has its separate per-window envelope.
            shared=sum(d['components']['retained_attention_parameters'] for d in got['memory_admission']['devices'])
            assert 0<=stats['retained_attention_bytes']<=shared
            streamed = policy == 'aggressive' and cards > 1
            assert stats['streamed_parameter_windows'] == (stats['retained_windows'] if streamed else 0)
            assert (stats['reused_projection_gradient_bytes'] > 0) == streamed
            assert (stats['reused_attention_gradient_bytes'] > 0) == (streamed and memory == 'attention')
            assert stats['borrowed_projection_bytes'] == (stats['retained_projection_bytes'] if streamed else 0)
        windows = [r for r in actual if r['kind']=='window' and r['step']==step]
        event_sum = sum(len(r['events']) for r in windows)
        event_max = max(len(r['events']) for r in windows)
        if family=='settle':
            # Settle diagnostics project away the adapter's identity nodes;
            # capacity counters cover the actually executed encoded graph.
            assert stats['events']>=event_sum and stats['window_events_max']>=event_max
        else:
            assert stats['events']==event_sum and stats['window_events_max']==event_max
        assert stats['window_outputs_max']==max(len(r['outputs']) for r in windows)
        assert max(len(r['pending']) for r in windows)<=stats['pending_peak']<=forward.queue
        assert 0<stats['window_stages_max']<=stats['stages']
    (tmp_path/'observed.json').write_text(json.dumps(dict(candidate=got,observations=actual)))


def test_resident_whole_batch_option_and_continued_warmup():
    from test_online_consumer import packet
    device=target();p=packet('attention')
    kw=dict(family='settle',implementation='native',device=device,schedule='prefill',preset='resident',
            training=True,optimizer='adamw',steps=1,warmup=1,windows_per_step=2,diagnostics=True,
            native_library=os.environ['TIDE_BUILD_DIR'],resident_library=os.environ['TIDE_RESIDENT_LIBRARY'])
    rows,other,split=[],[],[]
    a=run(p,observer=observer(rows),**kw)
    b=run(p,sample_chunk_rows=2**63-1,observer=observer(other),**kw)
    c=run(p,sample_chunk_rows=1,observer=observer(split),**kw)
    same(other,rows);compare_records(split,rows,p['workload']['batch'])
    assert a['final_cut']==b['final_cut']==c['final_cut']
    torch.testing.assert_close(torch.tensor(c['losses']),torch.tensor(a['losses']))
    assert b['batch_execution']['physical_chunks']==1
    assert all(d['components']['saved_contexts']==0 for d in b['memory_admission']['devices'])


def test_resident_saved_pool_refusal():
    from test_online_consumer import packet
    device=target()
    kw=dict(family='settle',implementation='native',device=device,preset='resident',steps=1,warmup=0,
            native_library=os.environ['TIDE_BUILD_DIR'],resident_library=os.environ['TIDE_RESIDENT_LIBRARY'])
    with pytest.raises(ValueError,match='budget'):
        run(packet('attention'),sample_chunk_rows=1,context_memory_bytes=1,**kw)
    for value in [-1,True,2**63]:
        with pytest.raises(ValueError,match='resident-context-bytes'):
            run(packet('attention'),context_memory_bytes=value,**kw)
