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
from tidegraph import ResidentPlacement
from tools.online_bench.host import run


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("implementation", ["native", "libtorch"])
@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_resident_sample_chunks(case, implementation, dtype_name, tmp_path):
    family, memory, schedule, optimizer, clear, delayed, training = case
    device = target()
    p = make_continuous_packet(graph=ranked_graph(layers=3,region_width=2,fanout=2,local_span=2,delayed=delayed),
        memory=memory,width=4,batch=5,tokens=2,vocab=7,clear=clear)
    expected, actual = [], []
    kw = dict(family=family,training=training,optimizer=optimizer,steps=2,warmup=0,windows_per_step=2,diagnostics=True)
    ref = run(p,implementation="python",device="cpu",schedule="streaming",observer=observer(expected),**kw)
    cards = 2 if schedule == "prefill" else 1
    if implementation == "libtorch":
        got, actual = standalone(p,device,family,schedule,training,optimizer,tmp_path,devices=cards,
                                dtype_name=dtype_name,extra=("--sample-chunk-rows=2",))
    else:
        owners = ResidentPlacement(devices=(str(device),f"npu:{device.index+1}")) if cards==2 else None
        got = run(p,implementation="native",device=device,dtype=dtype_name,schedule=schedule,preset="resident",
            sample_chunk_rows=2,observer=observer(actual),native_library=os.environ["TIDE_BUILD_DIR"],
            resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],resident_placement=owners,**kw)
    tol = dict(atol=2e-3,rtol=2e-2) if dtype_name=="float16" else dict(atol=1e-6,rtol=1e-5)
    compare_records(actual,expected,5,**tol,tensor_norm=dtype_name=="float16")
    if dtype_name=="float16":
        # Independently execute the whole batch at the same payload precision:
        # cross-dtype rounding and physical sample slicing are separate checks.
        whole=[]
        owners = ResidentPlacement(devices=(str(device),f"npu:{device.index+1}")) if cards==2 else None
        run(p,implementation="native",device=device,dtype=dtype_name,schedule=schedule,preset="resident",
            observer=observer(whole),native_library=os.environ["TIDE_BUILD_DIR"],
            resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],resident_placement=owners,**kw)
        compare_records(actual,whole,5,**tol)
    for key in ("outputs","parameters","final_cut","input_tokens_per_step"):
        assert got[key] == ref[key]
    torch.testing.assert_close(torch.tensor(got["losses"]),torch.tensor(ref["losses"]),**tol)
    assert got['batch_execution'] == dict(logical_batch=5,requested_sample_chunk_rows=2,
                                         effective_sample_chunk_rows=2,physical_chunks=3)
    assert got['memory_admission']['allocator_within_estimate']
    for d in got['memory_admission']['devices']:
        c = d['components']
        assert c['saved_contexts'] == 3*c['continuation_snapshot_bytes'] > 0
        assert (c['gradient_accumulation'] > 0) == training
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
