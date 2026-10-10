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
test::Fixture fixture(int shape,int variant,Index width) {
  auto f=test::fixture(shape,variant);
  auto expand=[&](const Tensor& x){return x.repeat({(width+2)/3}).narrow(0,0,width).clone();};
  for(size_t n=0;n<f.graph.nodes.size();++n) {
    auto& spec=f.graph.nodes[n];auto& w=f.model.nodes[n];
    w.bias=expand(w.bias);w.decay=expand(w.decay);w.read=expand(w.read);w.weight=at::eye(width,at::kFloat);
    if(!spec.identity&&n!=2)spec.readout="norm-fp32-v1"; // heterogeneous Read contracts
    if(n==0||n==3){spec.memory="lh-add-repeat-v1";w.extra["add_retention"]=at::full({},.5f,at::kFloat);
      if(variant)for(Index b=0;b<f.initial.batch_size;++b)if(!f.initial.states.count({b,Index(n)}))
        f.initial.states[{b,Index(n)}]={at::tensor({.25f,-.5f,.125f}),f.initial.cut-1,(Index(1)<<55)+b+1};}
    if(!spec.identity)spec.full="tanh";
  }
  for(auto& [_,s]:f.initial.states)s.value=expand(s.value);
  for(auto& x:f.input)x.value=expand(x.value);
  return f;
}
Index windows(at::Device device,test::Fixture f,bool prefill,bool vectorized,const std::string& mode,bool diagnostics=true) {
  for(auto& r:f.graph.regions)r.read_mode=mode;
  f.graph.compile();f.initial.identity=f.graph.identity;
  ContentLimits l;l.queue=96;l.arrivals=128;l.outputs=256;l.trace=diagnostics?1024:0;
  l.diagnostics=diagnostics;l.prefill=prefill;l.vectorized_read=vectorized;l.vectorized_state=!vectorized;l.full_chunk_rows=3;
  l.workspace_bytes=512*1024*1024;
  // Large coordinates test int64 identity, not billions of literal CPU ticks.
  // Fail fixture preparation before either executor if its work is unbounded.
  for(Index b=0;b<f.initial.batch_size;++b)for(Index n=0;n<Index(f.graph.nodes.size());++n)
    if(f.graph.nodes[n].memory=="lh-add-repeat-v1") {
      const auto it=f.initial.states.find({b,n});const auto last=it==f.initial.states.end()?-1:it->second.last_time;
      require(f.initial.cut-last<l.max_repeat_ticks-14,"norm fixture exceeds declared CPU tick-work bound");
    }
  ContentFlow flow(f.graph,f.model,f.initial,device,l);Streaming oracle(f.graph,f.model,{});Greedy greedy(f.graph,f.model,{});
  auto q=f.initial;Index previous=q.cut;
  for(Index step:{2,6,11}) {
    const auto stop=f.initial.cut+step;std::vector<External> input;
    for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)input.push_back(x);
    auto expected=oracle.run(q,input,stop,stop);
    tide_bench::compare(greedy.run(q,input,stop,stop),expected,true,at::kFloat);
    auto actual=flow.advance(input,stop);tide_bench::compare(actual,expected,diagnostics,at::kFloat);
    for(const auto& e:actual.trace)require(e.descriptor.scalar_type()==at::kFloat,"norm descriptor dtype");
    q=expected.continuation;previous=stop;
  }
  auto empty=flow.advance({},previous);tide_bench::compare(empty,oracle.run(q,{},previous,previous),diagnostics,at::kFloat);
  l.prefill=!prefill;l.vectorized_state=vectorized;l.vectorized_read=!vectorized;
  ContentFlow switched(f.graph,f.model,empty.continuation,device,l);
  tide_bench::compare(switched.advance({},previous+3),oracle.run(q,{},previous+3,previous+3),diagnostics,at::kFloat);
  return 5;
}
void analytic(at::Device device,bool prefill) {
  Graph g;g.nodes={{0},{0}};g.regions={{1,true,false,"content","positive-v1"}};g.inputs={0,1};g.outputs={0,1};
  for(auto& n:g.nodes){n.memory="identity";n.full="identity";n.readout="norm-fp32-v1";}
  g.compile();Model m;
  for(int n=0;n<2;++n)m.nodes.push_back({at::zeros({2},at::kFloat),at::eye(2,at::kFloat),
    at::zeros({2},at::kFloat),at::full({2},-7.f,at::kFloat)});
  m.input_scale=m.output_scale={at::ones({},at::kFloat),at::ones({},at::kFloat)};
  Continuation q;q.identity=g.identity;ContentLimits l;l.prefill=prefill;l.vectorized_read=prefill;
  ContentFlow flow(g,m,q,device,l);Streaming oracle(g,m,{});
  const std::vector<std::vector<float>> rows{{-3.f,4.f},{0.f,0.f},{1e6f,1.f},{1e6f,2.f}};
  std::vector<External> xs;
  for(Index t=0;t<3;++t)for(Index n=0;n<2;++n)
    xs.push_back({0,n,t,t,at::tensor(rows[t==0?0:t==1?1:2+n],at::kFloat)});
  auto expected=oracle.run(q,xs,3,3),actual=flow.advance(xs,3);
  tide_bench::compare(actual,expected,true,at::kFloat);
  for(const auto& e:actual.trace) {
    const auto want=e.time==0?5.f:e.time==1?0.f:1e6f;
    require(e.descriptor.item<float>()==want,"norm32 analytic descriptor");
    require(e.active==(e.time!=1&&e.node==0),"norm32 zero or stable rounded tie changed selection");
  }
  auto unsupported=g;unsupported.nodes[0].readout="norm-fp64-v1";unsupported.compile();
  auto bad=q;bad.identity=unsupported.identity;
  refuses([&]{ContentFlow rejected(unsupported,m,bad,device,l);},"module contract unavailable");
  // Finite input may overflow the declared FP32 norm. Never use an FP64 or
  // CPU substitution to hide it; the window fails and cannot be continued.
  ContentFlow overflow(g,m,q,device,l);
  refuses([&]{overflow.advance({{0,0,0,0,at::full({2},1e30f,at::kFloat)}},1);},"code=6");
  refuses([&]{overflow.snapshot();},"failed");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("norm gate requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);
    if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("norm gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;Index count=0;
    for(int shape=0;shape<4;++shape)for(int variant=0;variant<2;++variant)for(bool prefill:{false,true})
    for(bool vectorized:{false,true})for(const std::string mode:{"content","old","proposal"})
      count+=windows(device,fixture(shape,variant,3),prefill,vectorized,mode);
    for(Index width:{1,7,257,513})for(const std::string mode:{"content","proposal"})
      count+=windows(device,fixture(2,0,width),true,true,mode,false);
    for(bool vectorized:{false,true}) {
      auto wide=fixture(2,0,2048);for(auto& n:wide.graph.nodes)n.full="identity";
      count+=windows(device,wide,true,vectorized,"proposal",false);
    }
    analytic(device,false);analytic(device,true);
    std::cout<<"device-norm32: passed windows="<<count<<" analytic=2 overflow_refusals=2 fp64_refusals=2 scope=FP32_inference\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
