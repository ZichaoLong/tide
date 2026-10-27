"""CPU arithmetic adapter, independent of resident queues/vendor execution."""
import ctypes
from pathlib import Path
import subprocess
import torch


def build(directory):
    device=Path(__file__).resolve().parents[1]/"tools/device_online"
    path=directory/"aggregate.cpp"
    path.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_aggregate_math.h"
extern "C" int aggregate(int stage,void** t,int64_t* m,float alpha){
 using namespace tide_weighted_aggregate_device;
 auto f=[&](int i){return static_cast<float*>(t[i]);};
 auto l=[&](int i){return static_cast<int64_t*>(t[i]);};
 auto b=[&](int i){return static_cast<uint8_t*>(t[i]);};
 if(stage==0)return prepare(f(0),l(1),m[0],m[1],m[2],alpha,f(2),l(3),f(4));
 if(stage==1)return reduce(f(0),f(5),m[0],m[1],m[2],alpha,f(4)[0],f(6),f(7));
 if(stage==2)return reverse_mean(f(0),f(6),f(7),f(8),m[0],m[1],m[2],alpha,m[3],f(9),f(10),b(11),b(12));
 return reverse_norm(f(13),f(14),f(15),b(12),m[0],m[2],f(9));
}
''')
    shared=directory/"aggregate.so"
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-ffp-contract=off",
                    "-I"+str(device/"cuda"),"-I"+str(device/"ascendc"),
                    "-I"+str(Path(torch.__file__).parent/"include"),str(path),"-o",str(shared)],check=True)
    lib=ctypes.CDLL(str(shared));lib.aggregate.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_float]
    return lib


def call(lib,stage,tensors,meta,alpha):
    return lib.aggregate(stage,(ctypes.c_void_p*len(tensors))(*(x.data_ptr() for x in tensors)),meta.data_ptr(),alpha)
