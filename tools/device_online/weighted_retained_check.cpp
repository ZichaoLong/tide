#include "weighted_window_check.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted retained check requires explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);int count=0;
    for(const std::string topology:{"fc","hb","nonaligned","feedback"})for(const std::string read:{"old","proposal"}){tide::device_online::window_check::run_retained(device,topology,read);++count;}
    std::cout<<"weighted-retained: passed configurations="<<count<<" windows=3 empty_bridge=true independent_roots=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
