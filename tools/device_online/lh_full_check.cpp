#include "device_backend.h"
#include "packed_lh_full.h"
#include "content_fixture.h"
#include "lh_component_check.h"
#include "portable_torch/runtime.hpp"
#include "tide/lh_full.h"
#include "tide/stream.h"
#include "tide/greedy.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
Index windows(at::Device device) {
  const auto names=test::lh_profiles();Index cases=0;
  for(int shape:{0,1})for(int variant:{0,1})for(bool prefill:{false,true})for(Index mode=0;mode<3;++mode) {
    auto f=test::fixture(shape,variant);
    for(size_t n=0;n<f.graph.nodes.size();++n)if(!f.graph.nodes[n].identity) {
      f.graph.nodes[n].full=names[(n+3*variant+mode)%9];f.graph.nodes[n].readout="norm-fp32-v1";
      f.model.nodes[n].extra["lh_norm_weight"]=at::tensor({.75f,1.f,1.25f});
      f.model.nodes[n].extra["lh_norm_bias"]=at::tensor({.03125f,-.015625f,0.f});
    }
    for(auto& r:f.graph.regions)r.read_mode=mode==0?"content":mode==1?"old":"proposal";
    f.graph.compile();f.initial.identity=f.graph.identity;
    ContentLimits l;l.queue=96;l.arrivals=128;l.outputs=256;l.trace=1024;l.prefill=prefill;l.full_chunk_rows=3;
    l.workspace_bytes=512*1024*1024;
    ContentFlow flow(f.graph,f.model,f.initial,device,l);Streaming oracle(f.graph,f.model,{});Greedy greedy(f.graph,f.model,{});
    auto q=f.initial;Index previous=q.cut;
    for(Index step:{2,6,11}) {
      const auto stop=f.initial.cut+step;std::vector<External> xs;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)xs.push_back(x);
      auto expected=oracle.run(q,xs,stop,stop);tide_bench::compare(greedy.run(q,xs,stop,stop),expected,true,at::kFloat);
      tide_bench::compare(flow.advance(xs,stop),expected,true,at::kFloat);q=expected.continuation;previous=stop;++cases;
    }
    l.prefill=!prefill;ContentFlow restored(f.graph,f.model,flow.snapshot(),device,l);
    tide_bench::compare(restored.advance({},previous+3),oracle.run(q,{},previous+3,previous+3),true,at::kFloat);++cases;
  }
  return cases;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("LH Full gate requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);
    if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("LH Full gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    test::LhPrecision precision;const auto cells=test::lh_component(device,precision),count=windows(device);
    std::cout<<"device-lh-full: passed components="<<cells<<" windows="<<count<<" profiles=9 scope=FP32_broadcast_inference"
      <<" norm_rows="<<precision.normalized_rows<<" strict_component_misses="<<precision.strict_misses
      <<" cpu_fp64_max_abs="<<precision.max_cpu_error<<" device_fp64_max_abs="<<precision.max_device_error
      <<" max_condition_budget_fraction="<<precision.max_budget_fraction<<'\n';
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
