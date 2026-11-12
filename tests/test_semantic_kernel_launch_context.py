"""CPU semantic probes must not depend on a prior generated launch's grid."""
from pathlib import Path
import subprocess
import sys

import torch


def test_generated_launch_restores_context_after_success_and_exception(tmp_path):
    device = Path(__file__).resolve().parents[1]/"tools/device_online"
    source, output = tmp_path/"source", tmp_path/"generated"
    source.mkdir()
    (source/"CMakeLists.txt").write_text("ascendc_library(probe STATIC tide_probe.cpp)\n")
    (source/"tide_probe.cpp").write_text('''#include "kernel_operator.h"
extern "C" __global__ __aicore__ void tide_probe(GM_ADDR result,int64_t fail_at){
  if(int64_t(AscendC::GetBlockIdx())==fail_at)throw std::runtime_error("injected");
  auto* out=reinterpret_cast<int64_t*>(result);
  out[AscendC::GetBlockIdx()]=AscendC::GetBlockNum();
}
''')
    subprocess.run([sys.executable, str(device/"cuda/generate_kernels.py"),
                    "--source", str(source), "--output", str(output), "--backend", "cpu"], check=True)
    main = tmp_path/"main.cpp"
    main.write_text('''#include "tide_probe.cpp"
#include <cassert>
int main(){
  int64_t output[12]={};tide_cuda::worker=5;tide_cuda::workers=11;
  assert(tide_cuda_launch_tide_probe(4,nullptr,output,-1)==0);
  for(int i=0;i<4;++i)assert(output[i]==4);
  assert(tide_cuda::worker==5&&tide_cuda::workers==11);
  tide_probe(reinterpret_cast<GM_ADDR>(output),-1);assert(output[5]==11);
  bool failed=false;try{tide_cuda_launch_tide_probe(4,nullptr,output,2);}
  catch(const std::runtime_error&){failed=true;}
  assert(failed&&tide_cuda::worker==5&&tide_cuda::workers==11);
  assert(tide_cuda_launch_tide_probe(0,nullptr,output,-1)==1);
  assert(tide_cuda::worker==5&&tide_cuda::workers==11);
}
''')
    binary = tmp_path/"context-check"
    subprocess.run(["g++", "-std=c++17", "-O2", "-I"+str(output), "-I"+str(device/"cuda"),
                    "-I"+str(Path(torch.__file__).parent/"include"), str(main), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
