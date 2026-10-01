#include "lh_component_check.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>

int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))
      throw std::invalid_argument("packed LH component requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;
    const auto device=portable_torch::resolve_device(args);
    if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("packed LH component requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    tide::device_online::test::LhPrecision precision;
    const auto cases=tide::device_online::test::lh_component(device,precision,args.dtype);
    std::cout<<"packed-lh: passed cases="<<cases<<" profiles=9 scope=component_only keep_dtype=true"
      <<" norm_rows="<<precision.normalized_rows<<" strict_component_misses="<<precision.strict_misses
      <<" cpu_fp64_max_abs="<<precision.max_cpu_error<<" device_fp64_max_abs="<<precision.max_device_error
      <<" max_condition_budget_fraction="<<precision.max_budget_fraction<<'\n';
    runtime.close();return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
