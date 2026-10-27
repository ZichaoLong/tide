"""Shared parameter roles: exact static alias map, dynamic None/zero roots.

CPU adapter tests the segmented reduction, not device optimizer execution.
"""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    device=Path(__file__).resolve().parents[1]/"tools/device_online";directory=tmp_path_factory.mktemp("parameter-reduce")
    source=directory/"reduce.cpp";path=directory/"reduce.so"
    source.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_parameter_reduce.h"
extern "C" void run(void** t,int64_t count,int64_t tasks,int64_t blocks){
 tide_cuda::workers=blocks;
 for(tide_cuda::worker=0;tide_cuda::worker<blocks;++tide_cuda::worker){tide_cuda::reset_arena();
 tide_weighted_parameter_device::reduce((int64_t*)t[0],(int64_t*)t[1],(int64_t*)t[2],(float*)t[3],
   (uint8_t*)t[4],(int32_t*)t[5],(int32_t*)t[6],count,tasks);}
}
''')
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-ffp-contract=off",
                    "-I"+str(device/"cuda"),"-I"+str(device/"ascendc"),"-I"+str(Path(torch.__file__).parent/"include"),
                    str(source),"-o",str(path)],check=True)
    lib=ctypes.CDLL(str(path));lib.run.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_int64]*3
    return lib


def run(lib,table,groups,blocks=32):
    refs=torch.tensor([[x.data_ptr(),root.data_ptr()] for group in groups for x,root in group],dtype=torch.int64).reshape(-1,2)
    offsets=[0]
    for row in table:offsets.append(offsets[-1]+((row[3]+255)//256 if row[2]>=0 else 0))
    count=max((row[2]+row[3] for row in table if row[2]>=0),default=1)
    tensors=[torch.tensor(table,dtype=torch.int64),refs,torch.tensor(offsets),torch.full((count,),42.),
             torch.zeros(len(table),dtype=torch.bool),torch.zeros(16*max(1,offsets[-1]),dtype=torch.int32),torch.zeros(1,dtype=torch.int32)]
    def invoke():lib.run((ctypes.c_void_p*7)(*(t.data_ptr() for t in tensors)),len(table),offsets[-1],blocks)
    invoke();return tensors,invoke


@pytest.mark.parametrize("blocks",[1,32])
def test_all_alias_uses_none_zero_and_replay(library,blocks):
    yes=torch.tensor(True);no=torch.tensor(False);zero=torch.zeros(513);poison=torch.full((513,),float("nan"))
    first=torch.arange(513).float()*.01;second=-torch.arange(513).float()*.003
    groups=[[],[(first,yes),(poison,no),(second,yes)],[(torch.zeros(1),yes)],[(zero,no)],[]]
    table=[[0,0,-1,17],[0,3,0,513],[3,4,513,1],[4,5,514,513],[5,5,-1,2]]
    t,replay=run(library,table,groups,blocks)
    assert t[5].eq(0).all() and t[4].tolist()==[False,True,True,False,False]
    torch.testing.assert_close(t[3][:513],first+second);assert t[3][513:].eq(0).all()
    # The same program can change connection decisions without stale accumulation.
    yes.fill_(False);no.fill_(True);poison.zero_();zero.fill_(.25);replay()
    assert t[4].tolist()==[False,True,False,True,False]
    assert t[3][:514].eq(0).all() and t[3][514:].eq(.25).all()


def test_overflow_and_sticky_failure(library):
    root=torch.tensor(True);x=torch.full((257,),3e38)
    t,replay=run(library,[[0,2,0,257]],[[(x,root),(x,root)]])
    assert t[5][::16].tolist()==[23,23]
    t[6].fill_(23);t[3].fill_(17);replay();assert t[3].eq(17).all()
