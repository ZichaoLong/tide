// Public installed headers only. A one-node identity Full is a hand oracle:
// Agg=(m,z), regional_total with gamma=1 returns c=m, and G=z, never m*z.
#include <tide/resident_weighted.h>
#include "fixture.h"
#include <portable_torch/runtime.hpp>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted consumer requires explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    using namespace tide;auto fixture=weighted_consumer_fixture();auto g=fixture.graph;auto m=fixture.model;auto q=fixture.initial;
    device_online::WeightedFlowLimits limits;limits.queue=4;limits.cache=4;limits.stages=4;
    for(int step=0;step<2;++step){
      device_online::WeightedFlow owner(g,m,q,device,limits);auto value=step?at::tensor({.4f,-3.f}):at::tensor({.6f,2.f});
      auto result=owner.advance({{0,0,step,2*step,value}},2*step+1);
      if(result.outputs.size()!=1||result.trace.size()!=1||!result.messages.empty()||!result.continuation.pending.empty()
          ||!at::allclose(result.outputs[0].value,value,1e-5,1e-6)||result.continuation.states.at({0,0}).observations!=step+1)
        throw std::runtime_error("installed weighted consumer differs from hand identity/continuation");
      q=owner.checkpoint();owner.close();
    }
    std::cout<<"weighted-resident-consumer: passed hand_identity=true exact_continuation=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
