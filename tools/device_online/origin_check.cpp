#include "device_backend.h"
#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include "tide/greedy.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <stdexcept>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
ContentLimits limits(bool prefill,bool vectorized) {
  ContentLimits l;l.queue=128;l.arrivals=256;l.outputs=256;l.trace=2048;
  l.prefill=prefill;l.vectorized_aggregate=vectorized;l.workspace_bytes=512*1024*1024;return l;
}
test::Fixture one_node(Index width,Index edges,Index cut=1) {
  test::Fixture f;auto& g=f.graph;g.nodes={{0}};g.nodes[0].memory="identity";g.nodes[0].full="identity";
  g.regions={{1,true,true,"content","count-v1"}};g.outputs={0};
  for(Index e=0;e<edges;++e)g.edges.push_back({0,0,1});
  f.model.nodes={{at::zeros({width},at::kFloat),at::zeros({width,width},at::kFloat),
    at::zeros({width},at::kFloat),at::zeros({width},at::kFloat)}};
  f.model.agg_scale=f.model.edge_scale=std::vector<Tensor>(edges,at::ones({},at::kFloat));
  f.model.output_scale={at::ones({},at::kFloat)};
  f.initial.cut=cut;return f;
}
Index ordering(at::Device device) {
  Index cases=0;
  for(Index width:{1,33,257})for(bool ties:{false,true})for(bool prefill:{false,true})for(bool vectorized:{false,true}) {
    auto f=one_node(width,ties?32:3);
    if(ties)for(Index e=0;e<32;++e)f.graph.origins.push_back({e,(Index(1)<<55)+7,1});
    else f.graph.origins={{2,7,1}};
    f.graph.compile();f.initial.identity=f.graph.identity;
    for(Index e=0;e<Index(f.graph.edges.size());++e) {
      const float value=ties?(float(e%5)-2)/8:e==0?33554432.f:e==1?1.f:-33554432.f;
      f.initial.pending.push_back({0,0,1,1,e,0,at::full({width},value,at::kFloat)});
    }
    Streaming oracle(f.graph,f.model,{});const auto expected=oracle.run(f.initial,{},2,2);
    ContentFlow flow(f.graph,f.model,f.initial,device,limits(prefill,vectorized));const auto actual=flow.advance({},2);
    tide_bench::compare(actual,expected,true,at::kFloat);
    const auto& e=actual.trace.at(0);
    if(ties)for(Index i=0;i<32;++i)require(e.sources[i].slot==i,"equal projected keys lost physical order");
    else {
      require(at::equal(e.content,at::ones({width},at::kFloat)),"sum ignored projected order cancellation witness");
      require(e.fiber[0].source==0&&e.fiber[2].kind==1&&e.sources[0].atom.kind==0&&e.sources[0].atom.source==7,
        "projection changed physical atoms or did not change source view");
    }
    ++cases;
  }
  return cases;
}
Index windows(at::Device device) {
  Index cases=0;
  for(int shape:{0,1})for(int variant:{0,1})for(bool prefill:{false,true})for(bool vectorized:{false,true})for(Index stride:{1,3}) {
    auto f=test::fixture(shape,variant);const Index original=f.initial.cut;
    const Index base=original-original%stride;f.initial.cut=base;
    for(auto& [_,s]:f.initial.states)s.last_time=base-1;
    for(auto& [_,h]:f.initial.history)h.last_time=base-1;
    for(auto& x:f.input)x.time=base+(x.time-original)*stride;
    for(size_t e=0;e<f.graph.edges.size();++e) {
      f.graph.edges[e].delay*=stride;
      if(e%3)f.graph.origins.push_back({Index(e),variant?(Index(1)<<55)+3:Index((f.graph.edges.size()-e)%2),stride});
    }
    f.graph.compile();f.initial.identity=f.graph.identity;
    auto l=limits(prefill,vectorized);ContentFlow flow(f.graph,f.model,f.initial,device,l);
    Streaming oracle(f.graph,f.model,{});Greedy greedy(f.graph,f.model,{});
    auto q=f.initial;Index previous=base;
    for(Index step:{2,6,11}) {
      const Index stop=base+step*stride;std::vector<External> xs;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)xs.push_back(x);
      const auto expected=oracle.run(q,xs,stop,stop);
      tide_bench::compare(greedy.run(q,xs,stop,stop),expected,true,at::kFloat);
      tide_bench::compare(flow.advance(xs,stop),expected,true,at::kFloat);
      q=expected.continuation;previous=stop;++cases;
    }
    l.prefill=!prefill;l.vectorized_aggregate=!vectorized;
    ContentFlow restored(f.graph,f.model,flow.snapshot(),device,l);
    tide_bench::compare(restored.advance({},previous+3*stride),oracle.run(q,{},previous+3*stride,previous+3*stride),true,at::kFloat);
    ++cases;
  }
  return cases;
}
template<class F> void rejects(F f,const char* why) {bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,why);}
Index refusals(at::Device device) {
  Index cases=0;
  for(bool alias:{false,true})for(bool prefill:{false,true})for(bool vectorized:{false,true}) {
    auto f=one_node(7,alias?2:1,3);
    for(Index e=0;e<Index(f.graph.edges.size());++e) {
      f.graph.origins.push_back({e,7,alias?1:3});
      f.initial.pending.push_back({0,0,3,1,e,2,at::zeros({7},at::kFloat)});
    }
    if(alias)f.graph.source_domain=SourceDomain{{0,0},{}};
    f.graph.compile();f.initial.identity=f.graph.identity;
    Streaming oracle(f.graph,f.model,{});rejects([&]{oracle.run(f.initial,{},4,4);},"CPU accepted origin/source-domain error");
    ContentFlow flow(f.graph,f.model,f.initial,device,limits(prefill,vectorized));
    bool failed=false;try{flow.advance({},4);}catch(const std::exception& e) {
      failed=std::string(e.what()).find(alias?"code=2":"code=10")!=std::string::npos;
    }
    require(failed,"origin/source-domain device refusal code lost");
    rejects([&]{flow.snapshot();},"failed origin flow exposed snapshot");
    rejects([&]{flow.advance({},4);},"failed origin flow reentered");++cases;
  }
  return cases;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("origin gate requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);
    if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("origin gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    const auto ordered=ordering(device),count=windows(device),bad=refusals(device);
    std::cout<<"device-origins: passed ordering="<<ordered<<" windows="<<count<<" refusals="<<bad
      <<" physical_identity=true stable_projection=true scope=FP32_inference\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
