"""Bank visibility/relative positions/transpose; integer clocks exceed 2**54."""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch
from test_resident_kv_bank import Bank, bank_kernel
from test_resident_block_sequence import Fixture


@pytest.fixture(scope="module")
def read_kernel(tmp_path_factory):
    root=Path(__file__).resolve().parents[1]/"tools/device_online";work=tmp_path_factory.mktemp("bank_read")
    source,library=work/"adapter.cpp",work/"adapter.so"
    source.write_text('#define TIDE_KERNEL_CPU\n'+''.join(f'#include "tide_block_kv_{name}.cpp"\n'
        for name in ("read_plan","read_roots","read_weight_vjp","match_phase","match"))+
        'extern "C" void reset(){tide_cuda::worker=0;tide_cuda::workers=1;tide_cuda::reset_arena();}\n')
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-ffp-contract=off",
        "-I"+str(root/"cuda"),"-I"+str(root/"ascendc"),"-I"+str(Path(torch.__file__).parent/"include"),str(source),"-o",str(library)],check=True)
    dll=ctypes.CDLL(str(library))
    dll.tide_block_kv_read_plan.argtypes=[ctypes.c_void_p]*18+[ctypes.c_int64]*10+[ctypes.c_float]
    dll.tide_block_kv_read_roots.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int64]*6
    dll.tide_block_kv_read_weight_vjp.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int64]*3
    dll.tide_block_kv_match_phase.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int64]*2+[ctypes.c_float]*2
    dll.tide_block_kv_match.argtypes=[ctypes.c_void_p]*10+[ctypes.c_int64]*6+[ctypes.c_float]
    return dll


def plan(kernel,bank,proposal=True,beta=.3,full=False):
    f=bank.f;o,l,k,d=f.owners,f.length,bank.entries,2
    inputs=torch.randn(f.rows,d);owners=torch.arange(f.rows)%f.nodes;present=torch.ones(f.rows,dtype=torch.bool);present[5]=False
    weights=torch.empty(2,o,l,k);decay=torch.empty_like(weights);visible=torch.empty(o,l,k,dtype=torch.bool)
    ages=torch.empty(o,l,k,dtype=torch.int64);selected=torch.empty(o,l,dtype=torch.bool)
    qowners=torch.empty_like(owners);eowners=torch.empty(o*k,dtype=torch.int64);fowners=torch.empty_like(owners);clean=torch.empty_like(inputs)
    arrays=(bank.proposal_map if proposal else bank.old_map,f.proposal_i if proposal else f.old_i,
        bank.weights,bank.stamps,f.times,inputs,owners,present,weights,decay,visible,ages,selected,qowners,eowners,fowners,clean,f.error)
    kernel.reset()
    kernel.tide_block_kv_read_plan(*(x.data_ptr() for x in arrays),f.batch,l,f.nodes,f.capacity,k,d,f.nodes,f.horizon,1,full,beta)
    assert not f.error.item()
    return dict(weights=weights,decay=decay,visible=visible,ages=ages,present=selected,qowners=qowners,eowners=eowners,
                fowners=fowners,clean=clean,input=inputs,owners=owners,selected=present)


@pytest.mark.parametrize("snapshot",["old","proposal","bound"])
def test_bank_read_actual_membership_and_projection_use(read_kernel,bank_kernel,snapshot):
    f=Fixture();f.initial[0,18]=0;f.initial_fields[0][-1].data[0]=0
    bank=Bank(f,bound=snapshot=="bound");bank.run(bank_kernel);out=plan(read_kernel,bank,snapshot!="old",full=True)
    refs=f.reference()[snapshot=="proposal"] if snapshot!="bound" else None
    expected_entry=torch.full_like(out["eowners"],-1)
    for row in range(f.rows):
        b,step,node=row//(f.length*f.nodes),row//f.nodes%f.length,row%f.nodes;owner=b*f.nodes+node
        if refs is None:
            count=int(f.initial_i[owner,2]);stamps=f.initial_i[owner,4:4+count].tolist();weights=f.initial_fields[owner][-1]
        else:stamps,_,_,fields=refs[row];weights=fields[-1]
        visible=[]
        for stamp,weight in zip(stamps,weights.flatten()):
            age=int(f.times.flatten()[row])-stamp
            if out["selected"][row] and 0<=age<f.horizon:
                entry=(bank.stamps[owner]==stamp).nonzero().item();visible.append(entry)
                assert out["ages"][owner,step,entry]==age
                torch.testing.assert_close(out["weights"][1,owner,step,entry],weight.detach())
                torch.testing.assert_close(out["weights"][0,owner,step,entry],weight.detach()*torch.exp(torch.tensor(-age*.3)),rtol=1e-6,atol=1e-7)
                if weight>0:expected_entry[owner*bank.entries+entry]=node
        assert out["visible"][owner,step].nonzero().flatten().tolist()==sorted(visible)
        positive=bool(out["weights"][1,owner,step].gt(0).any())
        assert int(out["qowners"][row])==(node if positive else -1)
        assert int(out["fowners"][row])==(node if out["selected"][row] else -1)
        torch.testing.assert_close(out["clean"][row],out["input"][row] if positive else torch.zeros(2))
    assert torch.equal(out["eowners"],expected_entry)


@pytest.mark.parametrize("kind,channels",[(0,2),(1,2),(2,3)])
def test_bank_roots_and_passive_weight_transpose(read_kernel,bank_kernel,kind,channels):
    f=Fixture();bank=Bank(f);bank.run(bank_kernel);t=plan(read_kernel,bank,beta=1e30)
    torch.manual_seed(13);flags=torch.rand(channels,f.owners,f.length,3)>.4
    flags[:,~t["present"]]=False
    if not kind:flags[:,:,:,2]=False
    flags[:,:,:,1:]&=t["weights"][1].gt(0).any(-1)[None,:,:,None] if kind else True
    raw=torch.empty(f.rows,5,dtype=torch.bool);query=torch.empty(f.rows,dtype=torch.bool)
    keys=torch.empty(f.owners*bank.entries,dtype=torch.bool);values=torch.empty_like(keys)
    arrays=(flags,t["weights"],t["present"],raw,query,keys,values,f.error)
    read_kernel.tide_block_kv_read_roots(*(x.data_ptr() for x in arrays),f.batch,f.length,f.nodes,bank.entries,channels,kind)
    expect_k=torch.zeros(f.owners,bank.entries,dtype=torch.bool);expect_v=torch.zeros_like(expect_k)
    for owner in range(f.owners):
        b,node=divmod(owner,f.nodes)
        for step in range(f.length):
            row=(b*f.length+step)*f.nodes+node;weight,value,key=flags[:,owner,step].any(0).tolist()
            assert raw[row].tolist()==[False,bool(kind and key),bool(kind and value),bool(not kind and value),weight]
            assert bool(query[row])==bool(kind and key)
            if kind:
                expect_k[owner]|=t["weights"][1,owner,step].gt(0)&key
                expect_v[owner]|=t["weights"][1,owner,step].gt(0)&value
    assert torch.equal(keys.reshape_as(expect_k),expect_k) and torch.equal(values.reshape_as(expect_v),expect_v)
    bars=torch.randn_like(t["weights"]);bars[:,~t["visible"]]=float("nan")
    output=torch.empty_like(bank.weights)
    read_kernel.tide_block_kv_read_weight_vjp(*(x.data_ptr() for x in (bars,t["decay"],t["visible"],output,f.error)),f.owners,f.length,bank.entries)
    expected=torch.where(t["visible"][None],bars*t["decay"],0).sum((0,2))
    assert not f.error.item();torch.testing.assert_close(output,expected)


@pytest.mark.parametrize("pairs",[0,1,2])
def test_relative_match_forward_and_transpose_without_key_expansion(read_kernel,pairs):
    torch.manual_seed(150);o,q,k,d=2,3,4,4
    query=torch.randn(o,q,d);keys=torch.randn(o,k,d);visible=torch.rand(o,q,k)>.25
    ages=torch.arange(o*q*k,dtype=torch.int64).reshape(o,q,k);ages[0,0,1]=2**24+1
    freq=torch.tensor([.017,.031][:pairs] or [0.]);phase=torch.zeros(o,q,k,max(1,pairs));bias=torch.zeros(o,q,k)
    error=torch.zeros(1,dtype=torch.int32);position_scale=7.;slope=.03
    read_kernel.tide_block_kv_match_phase(*(x.data_ptr() for x in (ages,visible,freq,phase,bias,error)),o*q*k,pairs,position_scale,slope)
    cosine,sine=phase.cos(),phase.sin();logits=torch.zeros(o,q,k);bars=torch.randn_like(logits);bars[~visible]=0
    dq=torch.empty_like(query);dk=torch.empty_like(keys);scale=d**-.5
    arrays=(query,keys,visible,cosine,sine,logits,bars,dq,dk,error)
    read_kernel.tide_block_kv_match(*(x.data_ptr() for x in arrays),o,q,k,d,pairs,0,scale)
    read_kernel.tide_block_kv_match(*(x.data_ptr() for x in arrays),o,q,k,d,pairs,1,scale)
    qd=query.double().requires_grad_();kd=keys.double().requires_grad_();expected=torch.zeros_like(logits,dtype=torch.float64)
    for i in range(o):
        for a in range(q):
            for j in range(k):
                if not visible[i,a,j]:continue
                key=kd[i,j];pieces=[key[:d-2*pairs]]
                for pair in range(pairs):
                    # This validates the declared relative FP32 phase, while
                    # independent FP64 arithmetic validates the smooth map.
                    angle=phase[i,a,j,pair].double();x,y=key[d-2*pairs+2*pair:d-2*pairs+2*pair+2]
                    pieces.append(torch.stack((x*angle.cos()-y*angle.sin(),x*angle.sin()+y*angle.cos())))
                expected[i,a,j]=(qd[i,a]*torch.cat(pieces)).sum()*scale
    grad=torch.autograd.grad((expected*bars.double()).sum(),(qd,kd))
    assert not error.item();torch.testing.assert_close(logits.double(),expected,rtol=2e-6,atol=4e-7)
    for actual,want in zip((dq,dk),grad):torch.testing.assert_close(actual.double(),want,rtol=2e-6,atol=5e-7)
    torch.testing.assert_close(bias,torch.where(visible,-ages.float()/position_scale*slope,0),rtol=0,atol=0)
