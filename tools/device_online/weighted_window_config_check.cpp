// Whole-window configuration composition; local formula anchors are linked from
// docs/resident-weighted-window-vjp.md and the stage/region contracts.
#include "weighted_window_check.h"
#include "weighted_flow_fixture.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted window configurations require explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    using namespace tide;using namespace tide::device_online;int count=0;
    for(const auto& c:weighted_flow_check::configurations()){
      auto [g,m]=weighted_flow_check::configured(c,true,count%2==0,count%3==0);
      auto xs=weighted_check::inputs(g,at::kCPU,at::kFloat);for(auto& x:xs)x.value=x.value*at::tensor({.3f,1.f,1.f});
      xs[0].value=at::zeros_like(xs[0].value);
      const auto label="config-"+std::to_string(count)+"/"+c.selection_mode;
      window_check::run_window(device,std::move(g),std::move(m),std::move(xs),label,count%2==0,true);++count;
    }
    std::cout<<"weighted-window-config: passed configurations="<<count<<" independent_roots=true qry=true exact_raw_boundaries=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
