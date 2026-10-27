#include "weighted_training_check.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <iostream>
using namespace tide;using namespace tide::device_online;
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted training check requires explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    using namespace weighted_training_check;
    trajectory(device,"fc","kv",ResidentOptimizerKind::sgd,"none");
    trajectory(device,"hb","heterogeneous",ResidentOptimizerKind::adamw,"none");
    trajectory(device,"nonaligned","delta",ResidentOptimizerKind::sgd,"partial");
    trajectory(device,"feedback","heterogeneous",ResidentOptimizerKind::adamw,"partial");
    trajectory(device,"fc","affine",ResidentOptimizerKind::sgd,"all");
    failures(device);accumulation(device);
    std::cout<<"weighted-training: passed configurations=5 updates=20 windows=3 empty_window=true both_optimizers=true resume=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
