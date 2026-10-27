"""CPU adapter of region composition math; not a runtime or device proof."""
import ctypes
from pathlib import Path
import subprocess
import torch


def build(directory):
    device=Path(__file__).resolve().parents[1]/"tools/device_online"
    wrapper=directory/"region.cpp"
    wrapper.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_region_reverse.h"
extern "C" int region(int stage,void** t,int64_t* m,float* f){
  tide_cuda::reset_arena();tide_weighted_device::Math math;math.init();
  using namespace tide_weighted_region_device;
  auto p=[&](int i){return static_cast<float*>(t[i]);};
  auto b=[&](int i){return static_cast<uint8_t*>(t[i]);};
  if(stage==0)return prepare(math,p(0),p(1),p(2),p(3),p(4),m[0],m[1],m[2],m[4],m[5],m[6],p(7),p(8),p(6));
  if(stage==1)return finish(math,p(2),p(5),m[0],m[2],m[3],m[4],m[5],m[7],f[0],f[1],f[2],f[3],f[4],p(6),p(9),p(10),p(11),p(12),p(13));
  if(stage==2)return reverse_finish(m[0],m[0],m[2],m[3],m[4],m[5],m[6],m[7],f[0],f[2],f[3],f[4],p(6),p(9),p(10),p(5),p(14),p(15),b(16),p(17),p(18),p(19),p(20),p(21),p(22),b(23),b(24),b(25),math);
  return reverse_prepare(m[0],m[1],m[2],b(24)[0],b(25)[0],m[6],p(0),p(1),p(3),p(4),p(6),p(7),p(8),p(26),p(27),p(17),p(18),p(28),p(29));
}
''')
    path=directory/"region.so"
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-ffp-contract=off",
                    "-I"+str(device/"cuda"),"-I"+str(device/"ascendc"),
                    "-I"+str(Path(torch.__file__).parent/"include"),str(wrapper),"-o",str(path)],check=True)
    lib=ctypes.CDLL(str(path))
    lib.region.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_void_p]
    return lib


class Frame:
    def __init__(self,lib,fields,mode,context,attention,received=True):
        self.lib=lib;n,e,q=fields[0].numel(),fields[3].shape[1],fields[4].shape[1]
        self.meta=torch.tensor([n,e,q,mode,context,attention,received,1],dtype=torch.int64)
        self.config=torch.tensor([.35 if mode<=1 else .12,.8,.7,.5,q**-.5])
        self.t=[x.detach().clone() for x in fields]+[torch.zeros(3),torch.zeros(e+2),torch.zeros(q),
            torch.zeros(1),torch.zeros(q),torch.zeros(n),torch.zeros(4),torch.zeros(2),
            torch.zeros(1),torch.zeros(1),torch.zeros(1,dtype=torch.bool),
            torch.zeros(n),torch.zeros(n),torch.zeros(n),torch.zeros(n,q),torch.zeros(1),torch.zeros(q),
            torch.zeros(6,dtype=torch.bool),torch.zeros(1,dtype=torch.bool),torch.zeros(1,dtype=torch.bool),
            torch.zeros(e+2),torch.zeros(q),torch.zeros(n,e),torch.zeros(n,q)]

    def stage(self,number):
        pointers=(ctypes.c_void_p*len(self.t))(*(x.data_ptr() for x in self.t))
        assert self.lib.region(number,pointers,self.meta.data_ptr(),self.config.data_ptr())==0
