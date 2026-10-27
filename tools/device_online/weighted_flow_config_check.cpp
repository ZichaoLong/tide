// Allocation §2–5 and KV qry/position: whole graph composition and named roles.
#include "weighted_flow.h"
#include "weighted_flow_fixture.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include "../../cpp/bench/streaming.h"
#include "tide/greedy.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("weighted configurations require explicit accelerator FP32/FP16");
    args.allow_npu_float16=true;
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    using namespace tide;using namespace tide::device_online;int count=0;at::NoGradGuard no_grad;
    for(const auto& c:weighted_flow_check::configurations()){
      auto [g,m]=weighted_flow_check::configured(c,true,count%2==0,count%3==0,args.dtype);
      auto xs=weighted_check::inputs(g,at::kCPU,args.dtype);for(auto& x:xs)x.value=x.value*at::tensor({.3f,1.f,1.f}).to(args.dtype);
      xs[0].value=at::zeros_like(xs[0].value);Continuation q;q.identity=g.identity;q.batch_size=2;
      WeightedFlowLimits limits;limits.cache=8;limits.stages=32;WeightedFlow owner(g,m,q,device,limits);
      auto actual=owner.advance(xs,10);Greedy reference(g,m,Options{});auto expected=reference.run(q,xs,10,10);
      tide_bench::compare(actual,expected,true,args.dtype,std::nullopt,args.dtype==at::kHalf?2e-2:-1.,args.dtype==at::kHalf?1e-3:-1.);owner.close();++count;
      std::cout<<"weighted-flow-config case="<<count<<" mode="<<c.selection_mode<<" score="<<c.local_score<<" key="<<c.region_key.value_or("none")<<" passed\n"<<std::flush;
    }
    std::cout<<"weighted-flow-config: passed configurations="<<count<<" full_records=true qry=true source_padding=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
