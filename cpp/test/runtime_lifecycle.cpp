#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <iostream>

int main(int argc,char** argv) {
  try {
    portable_torch::RuntimeSession runtime;
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    auto device=portable_torch::resolve_device(args);
    at::set_num_threads(1);at::set_num_interop_threads(1);
    {
      portable_torch::RuntimeSession nested;
      auto value=at::arange(64,at::TensorOptions().device(device).dtype(args.dtype));
      for(int i=0;i<16;++i)value=value*0.5+1;
      portable_torch::synchronize(device);
      if(!value.cpu().isfinite().all().item<bool>())throw std::runtime_error("runtime arithmetic failed");
    }
    // Releasing a nested owner must not finalize the still-owned runtime.
    {
      auto value=at::ones({8},at::TensorOptions().device(device).dtype(args.dtype));
      portable_torch::synchronize(device);
      if(value.sum().cpu().item<double>()!=8)throw std::runtime_error("nested session ended runtime");
    }
    runtime.close();runtime.close();
    if(device.type()==c10::DeviceType::PrivateUse1) {
      bool rejected=false;
      try {portable_torch::resolve_device(args);}catch(const std::logic_error&){rejected=true;}
      if(!rejected)throw std::runtime_error("finalized NPU runtime reopened");
    }
    std::cout<<"runtime-lifecycle: passed nested=true idempotent_close=true\n";
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
