"""Static sample admission preserves independent complete logical-batch updates."""
from dataclasses import replace
import json
import os
import pytest
import torch
from test_consumer_sample_chunks import compare_records
from test_online_consumer import make_continuous_packet, ranked_graph, packet
from test_online_consumer_npu import target
from test_online_resident_consumer import standalone
from online_consumer_support import observer, same
from tidegraph import ResidentLimits, ResidentPlacement, ResidentTrainingLimits
from tools.online_bench.host import run
from tools.online_bench.capacity import Capacities, Chunks, MIB, packet_geometry, plan, plan_samples
from tools.online_bench.head_budget import head_budget


@pytest.mark.parametrize('implementation',['native','libtorch'])
@pytest.mark.parametrize('dtype_name',['float32','float16'])
@pytest.mark.parametrize('training',[False,True])
def test_automatic_samples_keep_complete_update(implementation,dtype_name,training,tmp_path):
    automatic_case(implementation,dtype_name,training,tmp_path)


@pytest.mark.parametrize('implementation',['native','libtorch'])
def test_automatic_samples_preserve_kv_journal_refusal(implementation,tmp_path):
    automatic_case(implementation,'float32',True,tmp_path,underprovisioned=True)


def automatic_case(implementation,dtype_name,training,tmp_path,underprovisioned=False):
    d=target();half=dtype_name=='float16'
    p=make_continuous_packet(graph=ranked_graph(layers=3,region_width=2,fanout=2,local_span=2),
        memory='attention',width=4,batch=17,tokens=2,vocab=7,clear=False)
    caps=Capacities(queue=512,arrivals=512,outputs=128,trace=2048,kv=8192,kv_trace=2048)
    g=replace(packet_geometry(p,windows=2,payload=2 if half else 4,training=training,adamw=True,
        diagnostics=True,devices=2),context_bytes=64*MIB)
    # Rank-aligned fixture: at most one complete event per node/token, with at
    # most source_count new KV rows. Last window is positions [6,8), so old+new
    # prefixes total at most 8**2-6**2 rows per source/sample across that window.
    # Charge every body node to one shard to keep this independent of placement
    # and actual selection. The old 2048-row failure remains a separate test.
    if not underprovisioned:
        caps.kv_trace=g.batch*sum(g.sources)*(8**2-6**2)
    minimum=Chunks(1,1,1,1,1,1,1)
    def peak(rows):
        q=replace(g,batch=rows,sample_chunks=(17-1)//rows+1)
        return max(d['estimated_peak_bytes'] for d in plan(q,caps,minimum,[64*1024**3]*2,True)['devices'])
    budget=((peak(17)+peak(9))//2+128*MIB)*10//9
    head=head_budget(128,4,7,g.payload,training,4*1024**3,True)
    expected_plan=plan_samples(g,caps,Chunks(head=head.rows),[budget]*2,True,17,True)
    selected=expected_plan['sample_admission']['effective_sample_rows']
    if training:
        # Full samples are preferable when a bounded owner move resolves the
        # initial coordinator bottleneck; otherwise retain the halved sample path.
        assert selected in (9,17)
        assert expected_plan['sample_admission']['attempted_sample_rows']==([17] if selected==17 else [17,9])
        if selected==17:
            assert expected_plan['owner_moves']>0
    expected=[];actual=[]
    kw=dict(family='timed-dag',training=training,optimizer='adamw',steps=2,warmup=0,
            windows_per_step=2,diagnostics=True)
    if not underprovisioned:
        ref=run(p,implementation='python',device='cpu',schedule='streaming',observer=observer(expected),**kw)
    def candidate():
        return execute_candidate(p,d,implementation,dtype_name,training,tmp_path,budget,caps,actual,kw)
    if underprovisioned:
        assert selected==17
        with pytest.raises((RuntimeError,AssertionError),match='content flow device refusal code=1.*kv_journal'):
            candidate()
        return
    got,actual=candidate()
    tol=dict(atol=2e-3,rtol=2e-2) if half else dict(atol=1e-6,rtol=1e-5)
    if selected<17:
        compare_records(actual,expected,17,**tol,tensor_norm=half)
    else:
        same(actual,expected,**tol,tensor_norm=half)
    for key in ('outputs','parameters','final_cut','input_tokens_per_step'):
        assert got[key]==ref[key]
    torch.testing.assert_close(torch.tensor(got['losses']),torch.tensor(ref['losses']),**tol)
    assert got['batch_execution']==dict(logical_batch=17,requested_sample_chunk_rows=0,
        effective_sample_chunk_rows=selected,physical_chunks=(17-1)//selected+1)
    for key,value in expected_plan.items():
        assert got['memory_admission'][key]==value
    assert got['memory_admission']['allocator_within_estimate']
    (tmp_path/'observed.json').write_text(json.dumps(dict(candidate=got,observations=actual)))


def execute_candidate(p,d,implementation,dtype_name,training,tmp_path,budget,caps,actual,kw):
    if implementation=='libtorch':
        got,actual=standalone(p,d,'timed-dag','prefill',training,'adamw',tmp_path,devices=2,dtype_name=dtype_name,
            extra=['--auto-sample-chunks','--chunk-policy=aggressive','--resident-queue=512','--resident-arrivals=512',
                   '--resident-outputs=128','--resident-trace=2048','--resident-kv-rows=8192','--resident-kv-trace-rows='+str(caps.kv_trace),
                   '--resident-retained-bytes='+str(2*1024**3),'--resident-backward-bytes='+str(32*1024**3),
                   '--resident-context-bytes='+str(64*MIB),'--device-memory-bytes='+str(budget)])
    else:
        got=run(p,implementation='native',device=d,dtype=dtype_name,preset='resident',schedule='prefill',
            auto_sample_chunks=True,device_memory_bytes=budget,context_memory_bytes=64*MIB,
            observer=observer(actual),native_library=os.environ['TIDE_BUILD_DIR'],
            resident_library=os.environ['TIDE_RESIDENT_LIBRARY'],
            resident_limits=ResidentLimits(queue=512,arrivals=512,outputs=128,trace=2048,kv_rows=8192,
                kv_trace_rows=caps.kv_trace,workspace_bytes=512*MIB,chunk_policy='aggressive'),
            training_limits=ResidentTrainingLimits(windows=2,retained_bytes=2*1024**3,backward_bytes=32*1024**3),
            resident_placement=ResidentPlacement(devices=(str(d),f'npu:{d.index+1}')),**kw)
    return got,actual


def test_automatic_sample_option_rejects_nonboolean():
    for value in (1,None,'true'):
        with pytest.raises(ValueError,match='boolean'):
            run(packet(),family='timed-dag',implementation='python',device='cpu',auto_sample_chunks=value)
