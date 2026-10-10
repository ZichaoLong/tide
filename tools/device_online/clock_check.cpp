#include "device_backend.h"
#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include "tide/greedy.h"
#include "tide/stream.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
template<class F> void refuses(F f,const std::string& marker) {
  try{f();}catch(const std::exception& e){if(std::string(e.what()).find(marker)!=std::string::npos)return;throw;}
  throw std::runtime_error("missing refusal: "+marker);
}
test::Fixture fixture(int shape,int variant) {
  auto f=test::fixture(shape,variant);auto& g=f.graph;
  auto phase=[&](Index n){return g.nodes[n].region?2:0;};
  for(size_t n=0;n<g.nodes.size();++n) {
    auto& node=g.nodes[n];node.state_clock=node.identity?StateClock{}:StateClock{5,phase(n),1};
    if(!node.identity)node.full="tanh";
    if(n==0||n==3) {
      node.memory="lh-add-repeat-v1";f.model.nodes[n].extra["add_retention"]=at::full({},.99f,at::kFloat);
      if(variant)for(Index b=0;b<2;++b)if(!f.initial.states.count({b,Index(n)}))
        f.initial.states[{b,Index(n)}]={at::tensor({.25f,-.5f,.125f}),f.initial.cut-1,(Index(1)<<55)+b};
    }
  }
  for(auto& e:g.edges)e.delay=e.delay*5+phase(e.target)-phase(e.source);
  for(auto& x:f.input)x.time=x.time*5+phase(g.inputs[x.port]);
  for(auto& [owner,s]:f.initial.states)s.last_time=g.nodes[owner.second].state_clock.to_global(s.last_time);
  for(auto& [owner,h]:f.initial.history)h.last_time=StateClock{5,owner.second?2:0,1}.to_global(h.last_time);
  f.initial.cut*=5;return f;
}
Index windows(at::Device device,int shape,int variant,bool prefill,bool vectorized,const std::string& mode) {
  auto f=fixture(shape,variant);for(auto& r:f.graph.regions)r.read_mode=mode;
  f.graph.compile();f.initial.identity=f.graph.identity;
  ContentLimits l;l.queue=96;l.arrivals=128;l.outputs=256;l.trace=1024;l.prefill=prefill;l.vectorized_state=vectorized;
  l.full_chunk_rows=3;
  ContentFlow flow(f.graph,f.model,f.initial,device,l);Streaming oracle(f.graph,f.model,{});Greedy greedy(f.graph,f.model,{});
  auto q=f.initial;Index previous=q.cut,cases=0;
  for(Index step:{2,6,11}) {
    const auto stop=f.initial.cut+5*step;std::vector<External> input;
    for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)input.push_back(x);
    auto expected=oracle.run(q,input,stop,stop);auto actual=flow.advance(input,stop);
    tide_bench::compare(greedy.run(q,input,stop,stop),expected,true,at::kFloat);
    tide_bench::compare(actual,expected,true,at::kFloat);
    q=expected.continuation;previous=stop;++cases;
  }
  auto empty=flow.advance({},previous);tide_bench::compare(empty,oracle.run(q,{},previous,previous),true,at::kFloat);++cases;
  l.prefill=!prefill;l.vectorized_state=!vectorized;
  ContentFlow restored(f.graph,f.model,empty.continuation,device,l);
  tide_bench::compare(restored.advance({},previous+15),oracle.run(q,{},previous+15,previous+15),true,at::kFloat);
  return cases+1;
}
void several_phases(at::Device device,bool vectorized,const std::string& mode) {
  Graph g;g.nodes={{0}};g.nodes[0].memory="lh-add-repeat-v1";g.nodes[0].full="identity";g.nodes[0].state_clock={5,1,3};
  g.regions={{1,true,false,mode,"count-v1"}};g.inputs={0};g.outputs={0};g.compile();
  Model m;m.nodes={{at::zeros({1},at::kFloat),at::ones({1,1},at::kFloat),at::zeros({1},at::kFloat),at::ones({1},at::kFloat)}};
  m.nodes[0].extra["add_retention"]=at::full({},.5f,at::kFloat);
  m.input_scale=m.output_scale={at::ones({},at::kFloat)};
  Continuation q;q.identity=g.identity;q.batch_size=1;q.states[{0,0}]={at::full({1},2.f,at::kFloat),-1,0};
  std::vector<External> input;
  const std::vector<Index> times{1,2,6,8};const std::vector<float> values{1,0,2,-1};
  for(Index i=0;i<4;++i)input.push_back({0,0,i,times[i],at::full({1},values[i],at::kFloat)});
  ContentLimits l;l.vectorized_state=vectorized;l.max_repeat_ticks=2;
  ContentFlow flow(g,m,q,device,l);Streaming oracle(g,m,{});Index previous=0;
  for(Index stop:{3,7,10}) {
    std::vector<External> xs;for(const auto& x:input)if(x.time>=previous&&x.time<stop)xs.push_back(x);
    auto expected=oracle.run(q,xs,stop,stop);auto actual=flow.advance(xs,stop);
    tide_bench::compare(actual,expected,true,at::kFloat);q=expected.continuation;previous=stop;
  }
  auto s=flow.snapshot().states.at({0,0});
  require(s.last_time==8&&s.observations==4&&s.value.item<float>()==-.4375f,"local clock analytic recurrence differs");
  // Invalid phases are execution failures; persistent off-phase states are
  // rejected during restoration, including when they would remain idle.
  ContentFlow invalid(g,m,q,device,l);
  refuses([&]{invalid.advance({{0,0,4,10,at::zeros({1},at::kFloat)}},11);},"code=9");
  refuses([&]{invalid.snapshot();},"failed");
  auto bad=q;bad.states.at({0,0}).last_time=9;
  refuses([&]{ContentFlow invalid_state(g,m,bad,device,l);},"phase");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("clock check requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);
    if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("clock check requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;Index cases=0;
    for(int shape=0;shape<4;++shape)for(int variant=0;variant<2;++variant)for(bool prefill:{false,true})
    for(bool vectorized:{false,true})for(const std::string mode:{"content","old","proposal"})
      cases+=windows(device,shape,variant,prefill,vectorized,mode);
    for(bool vectorized:{false,true})for(const std::string mode:{"content","old","proposal"})several_phases(device,vectorized,mode);
    std::cout<<"device-clock: passed windows="<<cases<<" multi_phase_windows=18 phase_refusals=12 int64=true scope=FP32_inference\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
