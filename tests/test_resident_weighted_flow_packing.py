"""Core-5 complete fibers/regions: int64, physical atoms and transactional owners.

These host semantic-source tests check only the device metadata kernels. Full
graph values/records have an independent C++ schedule in weighted_flow_check.
"""
import ctypes
from pathlib import Path
import subprocess

import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root=Path(__file__).resolve().parents[1]/"tools/device_online"
    work=tmp_path_factory.mktemp("weighted_flow_pack")
    source=work/"adapter.cpp";binary=work/"adapter.so"
    names=("node_pack","region_pack","owner_adopt")
    source.write_text('#define TIDE_KERNEL_CPU\n'+''.join(
        f'#include "tide_weighted_{name}.cpp"\n' for name in names))
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-I"+str(root/"cuda"),
        "-I"+str(root/"ascendc"),"-I"+str(Path(torch.__file__).parent/"include"),str(source),"-o",str(binary)],check=True)
    lib=ctypes.CDLL(str(binary))
    lib.tide_weighted_node_pack.argtypes=[ctypes.c_void_p]*12+[ctypes.c_int64]*6
    lib.tide_weighted_region_pack.argtypes=[ctypes.c_void_p]*9+[ctypes.c_int64]*2
    lib.tide_weighted_owner_adopt.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int64]*2
    return lib


def node_fixture():
    time=2**54+17
    coords=torch.tensor([[0,2,time,0,0,0],[0,2,time,1,0,time-1],[1,1,time,0,1,0],[0,0,0,0,0,0]])
    fibers=torch.tensor([[0,2,time,0],[1,1,time,1],[0,0,0,0],[0,0,0,0]])
    offsets=torch.tensor([0,2,3,3,3]);counts=torch.tensor([3,2,2])
    # Two independent physical atoms use the same parameter class.
    classes=torch.tensor([[7,0,0,0,1],[-1,-1,-1,-1,-1],[7,1,1,0,1]])
    indices=torch.empty(2,4,dtype=torch.int64);owners=torch.empty_like(indices)
    lengths=torch.empty(2,dtype=torch.int64);times=torch.empty_like(lengths)
    present=torch.empty(2,dtype=torch.bool);bank=torch.empty_like(lengths);error=torch.zeros(1,dtype=torch.int32)
    return [fibers,offsets,coords,counts,classes,indices,owners,lengths,times,present,bank,error]


def call_node(library,fields):
    library.tide_weighted_node_pack(*(x.data_ptr() for x in fields),2,2,4,2,3,fields[5].shape[1])


def test_compact_slots_preserve_global_indices_and_padding(library):
    f=node_fixture()
    # The actual fiber begins at global index 1; local slots are only two.
    f[2][1:3]=f[2][:2].clone();f[1]=torch.tensor([1,3,4,4,4])
    f[5]=torch.empty(2,2,dtype=torch.int64);f[6]=torch.empty_like(f[5])
    call_node(library,f)
    assert f[-1].item()==0
    assert f[5].tolist()==[[1,2],[4,4]]  # Global sentinel stays 4, never 2.
    assert f[6].tolist()==[[7,7],[-1,-1]] and f[7].tolist()==[2,0]
    assert f[8].tolist()==[2**54+17,0]
    f[4][2,1]=0;call_node(library,f)
    assert f[-1].item()==2  # SrcClass multiplicity did not relax SourceDomain.


def test_compact_overflow_refused_before_any_slot_write(library):
    f=node_fixture()
    index_storage=torch.full((4,),-997,dtype=torch.int64)
    owner_storage=torch.full((4,),-998,dtype=torch.int64)
    f[5]=index_storage[1:3].reshape(2,1);f[6]=owner_storage[1:3].reshape(2,1)
    call_node(library,f)
    assert f[-1].item()==2 and f[9].tolist()==[False,False]
    assert index_storage.tolist()==[-997,4,4,-997]
    assert owner_storage.tolist()==[-998,-1,-1,-998]


@pytest.mark.parametrize("bad",["negative_offset","excess_fibers"])
def test_malformed_ready_bounds_refused(library,bad):
    f=node_fixture()
    if bad=="negative_offset":f[1][0]=-1
    else:f[3][1]=5
    call_node(library,f)
    assert f[-1].item()==2 and not f[9].any()


def test_physical_atoms_int64_absence_and_source_domain(library):
    f=node_fixture();call_node(library,f)
    assert not f[-1].item()
    assert f[5].tolist()==[[0,1,4,4],[4,4,4,4]]
    assert f[6].tolist()==[[7,7,-1,-1],[-1]*4]
    assert f[7].tolist()==[2,0] and f[8].tolist()==[2**54+17,0]
    assert f[9].tolist()==[True,False] and f[10].tolist()==[0,-1]
    f[4][2,1]=0  # Reusing a logical SourceDomain slot is a different error.
    call_node(library,f);assert f[-1].item()==2


def test_origin_order_and_two_times_of_one_private_owner_rejected(library):
    f=node_fixture();f[4][0,3]=1;f[4][2,2:]=torch.tensor([0,0,1])
    call_node(library,f);assert not f[-1].item() and f[5][0,:2].tolist()==[1,0]
    f=node_fixture();f[0][1,:3]=torch.tensor([0,2,2**54+19])
    call_node(library,f);assert f[-1].item()==2


def test_complete_region_ids_and_missing_candidates(library):
    present=torch.tensor([[False,True,True],[True,False,False]])
    time=torch.tensor([[0,2**54+21,2**54+21],[7,0,0]])
    nodes=torch.tensor([2,8,19]);order=torch.empty(6,dtype=torch.int64)
    ids=torch.empty(2,3,dtype=torch.int64);times=torch.empty(2,dtype=torch.int64)
    lengths=torch.empty_like(times);inverse=torch.empty_like(ids);error=torch.zeros(1,dtype=torch.int32)
    fields=[present,time,nodes,order,ids,times,lengths,inverse,error]
    library.tide_weighted_region_pack(*(x.data_ptr() for x in fields),2,3)
    assert not error.item() and lengths.tolist()==[2,1] and times.tolist()==[2**54+21,7]
    assert ids.tolist()==[[8,19,0],[2,0,0]] and order.tolist()==[1,2,6,3,6,6]
    assert inverse.tolist()==[[6,0,1],[3,6,6]]
    time[0,2]+=1
    library.tide_weighted_region_pack(*(x.data_ptr() for x in fields),2,3)
    assert error.item()==2


@pytest.mark.parametrize("history",[False,True])
def test_commit_preserves_absence_and_obeys_shared_error(library,history):
    target=torch.tensor([[-1,0],[7,1]]) if history else torch.tensor([0,1])
    present=torch.tensor([True,False]);time=torch.tensor([2**54+5,0]);error=torch.ones(1,dtype=torch.int32)
    before=target.clone()
    for blocked in (True,False):
        error.fill_(int(blocked))
        library.tide_weighted_owner_adopt(*(x.data_ptr() for x in (target,present,time,error)),2,int(history))
        if blocked:assert torch.equal(target,before)
    assert target.tolist()==([[2**54+5,1],[7,1]] if history else [1,1])
