// Literal detached-copy witnesses; independent of benchmark/model arithmetic.
#include "return_packing.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace {
void same(const at::Tensor& original,const at::Tensor& returned){
  if(!returned.device().is_cpu()||returned.requires_grad()||original.sizes()!=returned.sizes()
      ||original.scalar_type()!=returned.scalar_type())throw std::runtime_error("return metadata differs");
  const auto bytes=[](const at::Tensor& x){return x.detach().cpu().contiguous().reshape({-1}).view(at::kByte);};
  if(!at::equal(bytes(original),bytes(returned)))throw std::runtime_error("return bytes differ");
}
}
int main(int argc,char** argv){
  portable_torch::RuntimeSession runtime;
  try{
    const auto args=portable_torch::parse_cli(argc,argv);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    const auto device=portable_torch::resolve_device(args);
    at::set_num_threads(1);
    for(const auto dtype:{args.dtype,at::kLong,at::kBool}){
      const auto o=at::TensorOptions().device(device).dtype(dtype);
      auto a=at::arange(4,at::TensorOptions().dtype(at::kLong)).to(o),b=at::ones({2,2},o);
      if(dtype==at::kLong)a=at::tensor({(int64_t(1)<<54)+7,int64_t(-1),int64_t(0),(int64_t(1)<<54)+8},o);
      if(dtype==at::kFloat||dtype==at::kHalf||dtype==at::kDouble){
        // ATen's literal-array constructor has no CPU Half dispatch. Build the
        // declared literal values in double, then explicitly select storage.
        a=at::tensor({0.,-0.,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()},
                     at::TensorOptions().dtype(at::kDouble)).to(o);
        a.set_requires_grad(true);
      }
      at::Tensor x,y,z;
      weighted_benchmark::return_copy::group({{a,{&x,&z}},{b,{&y}}});
      same(a,x);same(b,y);same(a,z);
      if(x.data_ptr()==z.data_ptr()||x.data_ptr()==y.data_ptr())throw std::runtime_error("returned fields share storage");
      x.fill_(1);same(a,z);
      std::vector<at::Tensor> copies{a,a,b,at::Tensor{},at::empty({0,2},o)};
      std::vector<at::Tensor*> fields;for(auto& value:copies)fields.push_back(&value);
      weighted_benchmark::return_copy::tensors(fields,true);
      same(a,copies[0]);same(a,copies[1]);same(b,copies[2]);
      if(copies[3].defined()||copies[4].numel()!=0)throw std::runtime_error("None/empty return changed");
      copies[0].fill_(2);same(a,copies[1]);
    }
    auto original=at::arange(12,at::TensorOptions().device(device).dtype(at::kFloat)).reshape({3,4}).t();
    auto copied=original;
    auto large=at::zeros({weighted_benchmark::return_copy::kPackBytes/4+1},original.options()),large_copy=large;
    weighted_benchmark::return_copy::tensors({&copied,&large_copy},true);
    same(original,copied);same(large,large_copy);
    if(copied.strides()!=original.cpu().clone().strides())throw std::runtime_error("fallback strides differ");
    portable_torch::synchronize(device);
    std::cout<<"complete-return-packing: passed\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
