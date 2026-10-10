#include "device_backend.h"
#include "packed_full.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace tide::device_online;
using I=int64_t;
void require(bool yes,const char* reason){if(!yes)throw std::runtime_error(reason);}
void check(at::Device device,at::ScalarType dtype) {
  at::NoGradGuard guard;const auto opts=at::TensorOptions().device(device).dtype(dtype);I cases=0;
  const bool half=dtype==at::kHalf;
  for(I width:{1,7,33,257})for(I chunk:{1,4}) {
    constexpr I nodes=3,rows=11;
    auto weight=(at::arange(nodes*width*width,at::kFloat).reshape({nodes,width,width}).remainder(17)/128).to(dtype);
    auto bias=(at::arange(nodes*width,at::kFloat).reshape({nodes,width})/256).to(dtype);
    // This tanh owner's weights must never participate when it is inactive.
    weight[2].fill_(std::numeric_limits<float>::quiet_NaN());bias[2].fill_(std::numeric_limits<float>::quiet_NaN());
    PackedFull full({1,0,1},weight,bias,device,chunk,8*1024*1024);
    require(full.weights().scalar_type()==dtype&&full.biases().scalar_type()==dtype,"Full changed parameter precision");
    const auto smaller=PackedFull::minimum_bytes({1,0,1},width,at::kHalf);
    require(smaller<PackedFull::minimum_bytes({1,0,1},width,at::kFloat),"Full budget ignored actual element size");
    ActionBatch input{at::zeros({rows,4},opts.dtype(at::kLong)),at::zeros({rows,width},opts),at::zeros({rows},opts.dtype(at::kBool))};
    auto comparison=at::zeros_like(input.values),error=at::zeros({1},opts.dtype(at::kInt));
    DeviceProgram program(device);auto result=full.append_stage(program,input,comparison,error);program.finish();
    auto coords=at::zeros({rows,4},at::kLong);for(I i=0;i<rows;++i){coords[i][1].fill_(i%nodes);coords[i][2].fill_((I(1)<<55)+i);}
    for(I round=0;round<4;++round) {
      auto mask=at::zeros({rows},at::kBool),h=(at::arange(rows*width,at::kFloat).reshape({rows,width})/128).to(dtype);
      auto state=at::cos(h.to(at::kFloat)).to(dtype),expected=h.clone();
      auto precise=h.to(at::kDouble);I actions=0;
      for(I i=0;i<rows;++i) {
        bool active=round!=0&&i%nodes!=2&&(round!=2||i%nodes==1)&&(round!=3||i<4);mask[i].fill_(active);
        if(active&&i%nodes==0){
          // Scalar CPU reference preserves the declared storage-rounding points.
          // FP64 independently checks the mathematical expression on those inputs.
          expected[i].copy_(h[i]+at::tanh(at::matmul(state[i],weight[0])+bias[0]));
          precise[i].copy_(h[i].to(at::kDouble)+at::tanh(at::matmul(state[i].to(at::kDouble),weight[0].to(at::kDouble))+bias[0].to(at::kDouble)));
          ++actions;
        }
        if(!active){state[i].fill_(std::numeric_limits<float>::quiet_NaN());h[i].fill_(std::numeric_limits<float>::quiet_NaN());}
      }
      input.coordinates.copy_(coords);input.values.copy_(h);input.valid.copy_(mask);comparison.copy_(state);full.chunks().zero_();
      portable_torch::synchronize(device);program.run();
      require(error.cpu().item<int>()==0,"Full packing refused legal actions");
      require(full.chunks().cpu().item<I>()==(actions+chunk-1)/chunk,"device Full chunk count differs");
      require(result.values.scalar_type()==dtype,"Full changed output precision");
      const auto actual=result.values.cpu().index({mask});
      require(at::allclose(actual,expected.index({mask}),half?3e-3:1e-5,half?2e-3:1e-6),"selected Full/identity/tail differs or inactive poison leaked");
      require(at::allclose(actual.to(at::kDouble),precise.index({mask}),half?3e-3:1e-5,half?2e-3:1e-6),"Full differs from independent CPU FP64 formula");++cases;
    }
    program.close();
  }
  std::cout<<"packed-full: passed cases="<<cases<<" device_chunking=true inactive_poison_isolated=true keep_dtype=true CPU=storage_dtype_FP64\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("packed Full check requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;
    auto device=portable_torch::resolve_device(args);if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("packed Full requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);check(device,args.dtype);runtime.close();return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
