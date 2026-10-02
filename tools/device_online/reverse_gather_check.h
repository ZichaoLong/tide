#pragma once
#include "reverse_gather.h"
#include <portable_torch/runtime.hpp>
#include <iostream>

namespace tide::device_online::test {
inline void reverse_gather_check(at::Device device,at::ScalarType dtype) {
  at::NoGradGuard no_grad;
  for(auto type:{dtype,at::kBool}) {
    auto input=at::zeros({3,7},at::TensorOptions().device(device).dtype(type));
    auto left=at::tensor({2,0,3,2},at::kLong).to(device),right=at::tensor({1,3,0,1},at::kLong).to(device);
    auto a=at::empty({4,7},input.options()),b=at::empty_like(a);
    CannProgram p(device),foreign(device);
    {
      ReverseGatherInput source(p,input);
      if(p.retained_tensor_bytes())throw std::runtime_error("unused reverse gather allocated storage");
      bool refused=false;try{source.select(foreign,left,a);}catch(const std::invalid_argument&){refused=true;}
      if(!refused)throw std::runtime_error("reverse gather accepted another program");
      source.select(p,left,a);source.select(p,right,b);
      const int64_t expected=input.nbytes()+4*7*input.element_size()+left.nbytes()+right.nbytes()+a.nbytes()+b.nbytes();
      if(p.retained_tensor_bytes()!=expected)throw std::runtime_error("reverse readers duplicated padded source storage");
    } // The compiled program must own the shared storage after this handle dies.
    p.finish();
    for(int replay=0;replay<3;++replay) {
      auto cpu=(at::arange(21,at::kFloat).reshape({3,7})+replay)*.25;
      if(type==at::kBool)cpu=cpu.remainder(2).eq(0);
      else cpu=cpu.to(type);
      input.copy_(cpu);a.fill_(1);b.fill_(1);portable_torch::synchronize(device);p.run();
      auto padded=at::cat({cpu,at::zeros({1,7},cpu.options())});
      if(!at::equal(a.cpu(),padded.index_select(0,left.cpu()))||!at::equal(b.cpu(),padded.index_select(0,right.cpu())))
        throw std::runtime_error("shared reverse gather replay/order/sentinel differs");
    }
    p.close();if(p.retained_tensor_bytes())throw std::runtime_error("reverse gather retained storage after close");
    foreign.close();
  }
  std::cout<<"shared-reverse-gather: passed types=2 replays=6 independent_destinations=true\n";
}
} // namespace tide::device_online::test
