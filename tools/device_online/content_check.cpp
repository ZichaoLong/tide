#include "content_flow.h"
#include "portable_torch/runtime.hpp"
#include "tide/greedy.h"
#include "tide/stream.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Fixture {Graph graph;Model model;Continuation initial;std::vector<External> input;};
Fixture fixture(int shape,int variant) {
  Fixture f;auto& g=f.graph;g.nodes={{0},{1},{0},{1}};
  g.regions={{1,variant==0,variant==0,"content","count-v1"},{1,true,true,"content","positive-v1"}};
  const std::vector<std::vector<Edge>> edges{
    {{0,1,2},{0,1,2},{2,3,1},{1,2,3},{3,0,2}},
    {{2,1,3},{2,1,1},{0,3,2},{1,3,4}},
    {},{{0,0,2},{1,1,1},{2,0,3},{0,2,1},{3,3,4}}};
  g.edges=edges[shape];g.inputs={0,2,3};g.outputs={2,1,2};
  for(size_t i=0;i<g.nodes.size();++i) {
    auto& node=g.nodes[i];node.memory=i==2?"identity":"ema";node.full="identity";node.clear=(i%2==0);
    node.identity=i==1&&variant==0;
    NodeWeights w{at::zeros({3},at::kFloat),at::eye(3,at::kFloat),at::zeros({3},at::kFloat),at::tensor({1.f,.5f,-.25f})};
    f.model.nodes.push_back(w);
  }
  g.compile();auto scale=[](float value){return at::full({},value,at::kFloat);};
  for(size_t p=0;p<g.inputs.size();++p)f.model.input_scale.push_back(scale(.5f+p*.125f));
  for(size_t e=0;e<g.edges.size();++e){f.model.agg_scale.push_back(scale(.5f));f.model.edge_scale.push_back(scale(e%2?.25f:.5f));}
  for(size_t p=0;p<g.outputs.size();++p)f.model.output_scale.push_back(scale(p%2?.5f:-.25f));
  const Index base=variant?(Index(1)<<55)+17:0;
  auto& q=f.initial;q.identity=g.identity;q.batch_size=2;q.cut=base;
  q.states[{0,0}]={at::tensor({.25f,-.5f,.125f}),base-1,(Index(1)<<55)+5};
  History history;history.last_time=base-1;history.node_maps["selected"]={{0,0},{2,(Index(1)<<55)+1}};q.history[{0,0}]=history;
  for(Index b=0;b<2;++b)for(Index p=0;p<3;++p)for(Index t=0;t<(b==0?3:2);++t) {
    auto value=at::tensor({float(1+p+t),float(b-p),float(t-b)})*.125f;
    if(variant)value=-value;
    if(p==2&&t==1)value.zero_(); // present zero still creates a candidate.
    f.input.push_back({b,p,t,base+(t==2?5:t),value});
  }
  return f;
}
void parity(at::Device device) {
  Index cases=0,total_events=0,total_messages=0,batched=0;
  for(int shape=0;shape<4;++shape)for(int variant=0;variant<2;++variant)for(bool prefill:{false,true}) {
    auto f=fixture(shape,variant);ContentLimits limits;limits.queue=96;limits.arrivals=128;limits.outputs=256;limits.trace=1024;limits.prefill=prefill;
    std::cerr<<"content-case shape="<<shape<<" variant="<<variant<<" prefill="<<prefill<<'\n';
    ContentFlow candidate(f.graph,f.model,f.initial,device,limits);
    Streaming oracle(f.graph,f.model,{});Greedy cpu_prefill(f.graph,f.model,{});
    auto q=f.initial;Index previous=q.cut;
    for(Index step:{2,6,11}) {
      Index stop=f.initial.cut+step;std::vector<External> input;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)input.push_back(x);
      auto reference=oracle.run(q,input,stop,stop);
      tide_bench::compare(cpu_prefill.run(q,input,stop,stop),reference,true,at::kFloat);
      auto actual=candidate.advance(input,stop);
      require(actual.continuation.identity==reference.continuation.identity,"graph identity changed");
      tide_bench::compare(actual,reference,true,at::kFloat);
      total_events+=actual.trace.size();total_messages+=actual.messages.size();
      if(prefill&&actual.stats.at("max_node_time_batch")>1)++batched;
      q=reference.continuation;previous=stop;++cases;
    }
    auto empty=candidate.advance({},previous);auto reference=oracle.run(q,{},previous,previous);
    tide_bench::compare(empty,reference,true,at::kFloat);++cases;
    // Restore the candidate's own cut with the other scheduling policy.
    limits.prefill=!prefill;ContentFlow switched(f.graph,f.model,empty.continuation,device,limits);
    auto continuation=switched.advance({},previous+3);auto continued=oracle.run(q,{},previous+3,previous+3);
    tide_bench::compare(continuation,continued,true,at::kFloat);++cases;
  }
  require(total_messages>0&&batched>0,"test did not exercise real recursive messages and batches");
  std::cout<<"content-flow: passed cases="<<cases<<" events="<<total_events<<" emitted_messages="<<total_messages
    <<" nontrivial_batches="<<batched<<" scope=FP32_sum_content_read_identity_full_ema_or_identity_state_inference\n";
}
template<class F> void rejects(F f,const char* message) {bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected,message);}
void refusal(at::Device device) {
  auto f=fixture(0,0);ContentLimits l;l.queue=64;l.arrivals=64;l.outputs=128;l.trace=512;
  auto bad=f.graph;bad.nodes[0].full="tanh";
  rejects([&]{ContentFlow x(bad,f.model,f.initial,device,l);},"unsupported Full accepted");
  auto small=l;small.workspace_bytes=1;rejects([&]{ContentFlow x(f.graph,f.model,f.initial,device,small);},"buffer preflight absent");
  // Fail independently on an output budget, iteration budget and debug capacity.
  for(int kind=0;kind<3;++kind) {
    auto limits=l;if(kind==0)limits.outputs=1;else if(kind==1)limits.stages=1;else limits.trace=1;
    ContentFlow flow(f.graph,f.model,f.initial,device,limits);
    rejects([&]{flow.advance(f.input,11);},"device capacity/stage failure silently succeeded");
    rejects([&]{flow.advance({},11);},"failed flow was allowed to resume");
  }
  // Invalid input is refused before submission; the same owner remains usable.
  ContentFlow flow(f.graph,f.model,f.initial,device,l);auto invalid=f.input;invalid[0].position=99;
  rejects([&]{flow.advance(invalid,11);},"invalid input accepted");
  Streaming oracle(f.graph,f.model,{});tide_bench::compare(flow.advance(f.input,11),oracle.run(f.initial,f.input,11,11),true,at::kFloat);
  std::cout<<"content-refusal: passed invalid_input_retry=true execution_failure_poisoned=true\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("content-flow check requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);
    if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("content-flow check requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    parity(device);refusal(device);runtime.close();return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
