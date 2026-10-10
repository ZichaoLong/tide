#include "../device_program.h"
#include "portable_torch/runtime.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto options=portable_torch::parse_cli(argc,argv,true);if(options.help){std::cout<<"CUDA resident failure contracts\n";return 0;}
    auto d=portable_torch::resolve_device(options);
    tide::device_online::validate_kernel_device(d);
    at::NoGradGuard no_grad;
    auto test=[&](auto action){try{action();}catch(const std::invalid_argument&){return;}throw std::runtime_error("invalid CUDA program accepted");};
    {tide::device_online::DeviceProgram p(d);p.label();test([&]{p.finish();});}
    {tide::device_online::DeviceProgram p(d);auto l=p.label();p.mark(l);p.mark(l);test([&]{p.finish();});}
    {tide::device_online::DeviceProgram p(d);p.limit_workspace(1);
      auto a=at::ones({1,2,2},at::TensorOptions().device(d));test([&]{p.batch_matmul(a,a,a);});}
    // A dynamic invalid branch is diagnosed by the device router at the one
    // window boundary; no host decision or out-of-bounds SWITCH is executed.
    {tide::device_online::DeviceProgram p(d);auto l=p.label();auto x=at::full({1},3,at::TensorOptions().device(d).dtype(at::kInt));
      p.branch(x,{l,l});p.mark(l);p.finish();bool refused=false;
      try{p.run();}catch(const std::runtime_error&){refused=true;}if(!refused)throw std::runtime_error("invalid CUDA branch accepted");}
    std::cout<<"device-failure: passed backend=cuda\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
