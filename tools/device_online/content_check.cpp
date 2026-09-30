#include "content_flow.h"
#include "content_fixture.h"
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
using tide::device_online::test::fixture;

void parity(at::Device device) {
  Index cases=0,total_events=0,total_messages=0,batched=0,state_batched=0;
  for(int shape=0;shape<4;++shape)for(int variant=0;variant<2;++variant)for(bool prefill:{false,true})
  for(bool nonlinear:{false,true})for(const std::string mode:{"content","old","proposal","mixed"}) {
    auto f=fixture(shape,variant);ContentLimits limits;limits.queue=96;limits.arrivals=128;limits.outputs=256;limits.trace=1024;limits.prefill=prefill;
    limits.full_chunk_rows=prefill?3:1;
    if(nonlinear)for(auto& n:f.graph.nodes)if(!n.identity)n.full="tanh";
    for(size_t r=0;r<f.graph.regions.size();++r)f.graph.regions[r].read_mode=mode=="mixed"?(r?"proposal":"content"):mode;
    f.graph.compile();f.initial.identity=f.graph.identity;
    std::cerr<<"content-case shape="<<shape<<" variant="<<variant<<" prefill="<<prefill<<" nonlinear="<<nonlinear<<" read="<<mode<<'\n';
    ContentFlow candidate(f.graph,f.model,f.initial,device,limits);
    Streaming oracle(f.graph,f.model,{});Greedy cpu_prefill(f.graph,f.model,{});
    auto q=f.initial;Index previous=q.cut;
    for(Index step:{2,6,11}) {
      Index stop=f.initial.cut+step;std::vector<External> input;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)input.push_back(x);
      auto reference=oracle.run(q,input,stop,stop);
      tide_bench::compare(cpu_prefill.run(q,input,stop,stop),reference,true,at::kFloat);
      auto actual=candidate.advance(input,stop);
      require(actual.stats.at("max_causal_node_time_batch")<=1,"state-dependent selection was prepared across causal Next");
      if(nonlinear&&!actual.trace.empty())require(actual.stats.at("full_chunk_rows")==limits.full_chunk_rows,"Full chunk selection changed");
      require(actual.continuation.identity==reference.continuation.identity,"graph identity changed");
      tide_bench::compare(actual,reference,true,at::kFloat);
      total_events+=actual.trace.size();total_messages+=actual.messages.size();
      if(prefill&&actual.stats.at("max_node_time_batch")>1)++batched;
      if(prefill&&actual.stats.at("max_state_read_node_time_batch")>1)++state_batched;
      q=reference.continuation;previous=stop;++cases;
    }
    auto empty=candidate.advance({},previous);auto reference=oracle.run(q,{},previous,previous);
    tide_bench::compare(empty,reference,true,at::kFloat);++cases;
    // Restore the candidate's own cut with the other scheduling policy.
    limits.prefill=!prefill;ContentFlow switched(f.graph,f.model,empty.continuation,device,limits);
    auto continuation=switched.advance({},previous+3);auto continued=oracle.run(q,{},previous+3,previous+3);
    tide_bench::compare(continuation,continued,true,at::kFloat);++cases;
  }
  require(total_messages>0&&batched>0&&state_batched>0,"test did not exercise recursive messages and content/state Read batches");
  std::cout<<"content-flow: passed cases="<<cases<<" events="<<total_events<<" emitted_messages="<<total_messages
    <<" nontrivial_batches="<<batched<<" state_read_batches="<<state_batched
    <<" scope=FP32_sum_content_old_proposal_read_identity_or_tanh_full_ema_or_identity_state_inference\n";
}
template<class F> void rejects(F f,const char* message) {bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected,message);}
void refusal(at::Device device) {
  auto f=fixture(0,0);ContentLimits l;l.queue=64;l.arrivals=64;l.outputs=128;l.trace=512;
  auto bad=f.graph;bad.nodes[0].full="swiglu";
  rejects([&]{ContentFlow x(bad,f.model,f.initial,device,l);},"unsupported Full accepted");
  bad=f.graph;bad.nodes[0].emit_phases.assign(bad.outgoing_ports.offsets[1]-bad.outgoing_ports.offsets[0],bad.nodes[0].emit_period);
  rejects([&]{ContentFlow x(bad,f.model,f.initial,device,l);},"invalid emission phase accepted");
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
