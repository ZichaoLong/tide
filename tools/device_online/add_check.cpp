#include "device_backend.h"
#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include "tide/greedy.h"
#include "tide/stream.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
template<class F> void refuses(F f,const std::string& marker) {
  try{f();}catch(const std::exception& e){if(std::string(e.what()).find(marker)!=std::string::npos)return;throw;}
  throw std::runtime_error("missing refusal: "+marker);
}
test::Fixture fixture(int shape,int variant,Index width=3) {
  auto f=test::fixture(shape,variant);
  auto expand=[&](const Tensor& x){auto y=x.repeat({(width+2)/3}).narrow(0,0,width).clone();return y;};
  for(size_t n=0;n<f.graph.nodes.size();++n) {
    auto& node=f.graph.nodes[n];auto& w=f.model.nodes[n];
    w.bias=expand(w.bias);w.decay=expand(w.decay);w.read=expand(w.read)/float(width);
    w.weight=at::eye(width,at::kFloat);if(!node.identity)node.full="tanh";
    if(n==0||n==3) {
      node.memory="lh-add-repeat-v1";
      const float rho=shape==0?.99f:shape==1?(n==0?0.f:1.f):shape==2?-.5f:1.01f;
      w.extra["add_retention"]=at::full({},rho,at::kFloat);
      if(variant)for(Index b=0;b<2;++b)if(!f.initial.states.count({b,Index(n)}))
        f.initial.states[{b,Index(n)}]={at::tensor({.25f,-.5f,.125f}),f.initial.cut-1,(Index(1)<<55)+b+1};
    }
  }
  for(auto& [_,s]:f.initial.states)s.value=expand(s.value);
  for(auto& x:f.input)x.value=expand(x.value);
  return f;
}
Index windows(at::Device device,test::Fixture f,bool prefill,bool vectorized,const std::string& mode,bool diagnostics=true) {
  for(auto& r:f.graph.regions)r.read_mode=mode;
  f.graph.compile();f.initial.identity=f.graph.identity;
  ContentLimits limits;limits.queue=96;limits.arrivals=128;limits.outputs=256;limits.trace=diagnostics?1024:0;
  limits.diagnostics=diagnostics;limits.prefill=prefill;limits.vectorized_state=vectorized;
  limits.full_chunk_rows=3;limits.workspace_bytes=512*1024*1024;
  ContentFlow flow(f.graph,f.model,f.initial,device,limits);Streaming oracle(f.graph,f.model,{});Greedy greedy(f.graph,f.model,{});
  auto q=f.initial;Index previous=q.cut,cases=0;
  for(Index step:{2,6,11}) {
    const auto stop=f.initial.cut+step;std::vector<External> input;
    for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)input.push_back(x);
    auto expected=oracle.run(q,input,stop,stop);
    tide_bench::compare(greedy.run(q,input,stop,stop),expected,true,at::kFloat);
    auto actual=flow.advance(input,stop);tide_bench::compare(actual,expected,diagnostics,at::kFloat);
    q=expected.continuation;previous=stop;++cases;
  }
  auto empty=flow.advance({},previous);tide_bench::compare(empty,oracle.run(q,{},previous,previous),diagnostics,at::kFloat);++cases;
  limits.prefill=!prefill;limits.vectorized_state=!vectorized;
  ContentFlow switched(f.graph,f.model,empty.continuation,device,limits);
  tide_bench::compare(switched.advance({},previous+3),oracle.run(q,{},previous+3,previous+3),diagnostics,at::kFloat);
  return cases+1;
}
void exact_repeat(at::Device device,bool vectorized) {
  Graph g;g.nodes={{0}};g.nodes[0].memory="lh-add-repeat-v1";g.nodes[0].full="identity";
  g.regions={{1,true,false,"proposal","count-v1"}};g.inputs={0};g.compile();
  Model m;m.nodes={{at::zeros({1},at::kFloat),at::ones({1,1},at::kFloat),at::zeros({1},at::kFloat),at::ones({1},at::kFloat)}};
  m.nodes[0].extra["add_retention"]=at::full({},.99123f,at::kFloat);m.input_scale={at::ones({},at::kFloat)};
  Continuation q;q.identity=g.identity;q.batch_size=1;q.states[{0,0}]={at::full({1},.73f,at::kFloat),-1,0};
  std::vector<External> input{{0,0,0,36,at::zeros({1},at::kFloat)}};
  Streaming oracle(g,m,{});auto expected=oracle.run(q,input,37,37);
  auto repeated=q.states.at({0,0}).value.clone();for(int tick=0;tick<37;++tick)repeated=repeated*m.nodes[0].extra.at("add_retention");
  const auto shortcut=q.states.at({0,0}).value*at::pow(m.nodes[0].extra.at("add_retention"),37);
  require(!at::equal(shortcut,repeated),"rounding fixture does not distinguish power from repeat");
  ContentLimits l;l.vectorized_state=vectorized;l.max_repeat_ticks=37;
  ContentFlow flow(g,m,q,device,l);
  // The flow must own its extra parameters, not alias a caller's CPU values.
  m.nodes[0].extra.at("add_retention").fill_(std::numeric_limits<float>::quiet_NaN());
  auto actual=flow.advance(input,37);tide_bench::compare(actual,expected,true,at::kFloat);
  require(at::equal(actual.continuation.states.at({0,0}).value,repeated),"Add changed literal multiplication order");
}
void failures(at::Device device) {
  for(bool vectorized:{false,true})for(const std::string mode:{"content","old","proposal"}) {
    auto f=fixture(1,0);for(auto& r:f.graph.regions)r.read_mode=mode;
    f.graph.compile();f.initial.identity=f.graph.identity;
    ContentLimits l;l.vectorized_state=vectorized;l.max_repeat_ticks=1;
    ContentFlow flow(f.graph,f.model,f.initial,device,l);
    refuses([&]{flow.advance(f.input,11);},"code=8");
    refuses([&]{flow.snapshot();},"failed");refuses([&]{flow.advance({},11);},"failed");
  }
  auto f=fixture(0,1);f.graph.compile();f.initial.identity=f.graph.identity;
  f.initial.states.at({0,0}).observations=std::numeric_limits<Index>::max();
  ContentFlow flow(f.graph,f.model,f.initial,device);
  refuses([&]{flow.advance(f.input,f.initial.cut+11);},"code=5");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("Add check requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);
    if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("Add check requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;Index cases=0;
    for(int shape=0;shape<4;++shape)for(int variant=0;variant<2;++variant)for(bool prefill:{false,true})
    for(bool vectorized:{false,true})for(const std::string mode:{"content","old","proposal"})
      cases+=windows(device,fixture(shape,variant),prefill,vectorized,mode);
    for(Index width:{1,7,8,17,255,256,257,513})for(bool vectorized:{false,true})
      cases+=windows(device,fixture(2,0,width),true,vectorized,"content",false);
    exact_repeat(device,false);exact_repeat(device,true);failures(device);
    std::cout<<"device-add: passed windows="<<cases<<" literal_rounding=2 work_limit_refusals=6 counter_overflow=1 scope=FP32_inference\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
