#include "tide/parameter_bank.h"
#include <ATen/core/grad_mode.h>
#include <limits>

void check_parameter_bank(){
  at::NoGradGuard guard;
  auto require=[](bool ok){if(!ok)throw std::runtime_error("parameter bank admission differs");};
  auto refused=[&](const at::Tensor& x,const std::vector<int64_t>& shape,int64_t offset,int64_t bytes){
    const auto saved=bytes;bool rejected=false;
    // Invalid CUDA index is deliberate: budget/shape refusal must precede any
    // target allocator/runtime access, including in a CPU-only LibTorch build.
    try{tide::copy_parameter_bank(x,shape,offset,at::Device(at::kCUDA,127),bytes);}
    catch(const std::invalid_argument&){rejected=true;}
    require(rejected&&bytes==saved);
  };
  auto huge=at::ones({1},at::kFloat).expand({int64_t{1}<<40});
  refused(huge,{1,int64_t{1}<<40},0,1024);
  refused(huge,{std::numeric_limits<int64_t>::max(),2},0,1024);
  for(auto dtype:{at::kFloat,at::kHalf}){
    auto parameter=at::arange(6,at::kFloat).reshape({2,3}).t().to(dtype);
    refused(parameter,{1,6},0,23);
    refused(parameter,{1,6},1,24);
    int64_t bytes=48;
    auto first=tide::copy_parameter_bank(parameter,{1,6},0,at::kCPU,bytes);
    auto second=tide::copy_parameter_bank(parameter,{2,3},0,at::kCPU,bytes);
    require(bytes==0&&first.scalar_type()==at::kFloat&&second.scalar_type()==at::kFloat);
    require(at::equal(first.reshape({-1}),parameter.reshape({-1}).to(at::kFloat)));
    require(first.data_ptr()!=second.data_ptr()&&first.data_ptr()!=parameter.data_ptr());
    first.zero_();require(second.sum().item<float>()==15&&parameter.sum().item<float>()==15);
    refused(parameter,{1},0,bytes);
    bytes=8;auto slice=tide::copy_parameter_bank(parameter,{2},2,at::kCPU,bytes);
    require(bytes==0&&at::equal(slice,parameter.reshape({-1}).narrow(0,2,2).to(at::kFloat)));
  }
}
