#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include "tide/greedy.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
template<class F> void refuses(F f,const std::string& marker) {
  try{f();}catch(const std::exception& e){if(std::string(e.what()).find(marker)!=std::string::npos)return;throw;}
  throw std::runtime_error("missing fiber refusal: "+marker);
}
void weights(NodeWeights& w,Index width) {
  auto eye=at::eye(width,at::kFloat);
  w.extra["fiber_qkv"]=at::cat({eye*.25f,eye*.5f,eye},1);
  w.extra["fiber_qkv_bias"]=at::arange(3*width,at::kFloat).remainder(7)*.015625f;
  w.extra["fiber_out"]=eye*.5f;w.extra["fiber_out_bias"]=at::full({width},.03125f,at::kFloat);
  w.extra["fiber_decay"]=at::full({},.03125f,at::kFloat);
}
State initial(Index width,Index heads,Index time=-1) {
  return {at::full({width},.125f,at::kFloat),time,5,
    {{"key",at::arange(2*width,at::kFloat).reshape({2,heads,width/heads})*.015625f},
     {"value",at::full({2,heads,width/heads},.25f,at::kFloat)},
     {"log_bias",at::tensor({-.0625f,.125f})}}};
}
ContentLimits limits(Index chunk=4) {
  ContentLimits l;l.queue=96;l.arrivals=192;l.outputs=192;l.trace=1024;
  l.kv_rows=32;l.kv_trace_rows=8192;l.attention_chunk_rows=chunk;l.workspace_bytes=256*1024*1024;
  return l;
}
test::Fixture one_node(Index width,Index heads) {
  test::Fixture f;auto& g=f.graph;g.nodes={{0}};auto& node=g.nodes[0];
  node.memory="lh-fiber-attention-sum-repeat-v1";node.full="tanh";node.query_heads=node.kv_heads=heads;
  g.regions={{1,true,false,"proposal","count-v1"}};g.inputs={0,0};g.outputs={0};
  g.compile();g.layout->input={1,0};g.source_domain->input={1,0};g.compile();
  NodeWeights w{at::zeros({width},at::kFloat),at::eye(width,at::kFloat)*.5f,at::zeros({width},at::kFloat),at::ones({width},at::kFloat)};
  weights(w,width);f.model.nodes={w};f.model.input_scale={at::ones({},at::kFloat),at::ones({},at::kFloat)};
  f.model.output_scale={at::ones({},at::kFloat)};f.initial.identity=g.identity;f.initial.batch_size=1;
  return f;
}
Index anchors(at::Device device) {
  Index cases=0;
  for(Index width:{1,7,33,257})for(Index chunk:{1,4})for(bool cached:{false,true}) {
    auto f=one_node(width,width==33?3:1);auto& w=f.model.nodes[0];
    if(width==1) {
      w.extra["fiber_qkv"]=at::tensor({cached?0.f:1.f,cached?0.f:1.f,1.f}).reshape({1,3});
      w.extra["fiber_qkv_bias"].zero_();w.extra["fiber_out"].fill_(1);w.extra["fiber_out_bias"].zero_();w.extra["fiber_decay"].zero_();
    }
    if(cached) {
      f.initial.states[{0,0}]=initial(width,f.graph.nodes[0].query_heads);
      if(width==1)f.initial.states[{0,0}]={at::zeros({1},at::kFloat),-1,0,
        {{"key",at::zeros({1,1,1},at::kFloat)},{"value",at::full({1,1,1},5.f,at::kFloat)},{"log_bias",at::zeros({1},at::kFloat)}}};
    }
    f.input={{0,0,0,0,at::ones({width},at::kFloat)},{0,1,0,0,at::full({width},3.f,at::kFloat)}};
    Streaming cpu(f.graph,f.model,{});const auto expected=cpu.run(f.initial,f.input,1,1);
    ContentFlow flow(f.graph,f.model,f.initial,device,limits(chunk));auto actual=flow.advance(f.input,1);
    tide_bench::compare(actual,expected,true,at::kFloat);
    if(width==1) {
      const double exact=cached?6.:2+2*(1/(1+std::exp(-2.))+1/(1+std::exp(-6.)));
      require(std::abs(actual.trace[0].proposal.item<double>()-exact)<1e-5,"same-fiber all-key analytic anchor failed");
    }
    const auto& s=actual.continuation.states.at({0,0});
    require(s.slots.at("key").size(0)==(cached?(width==1?3:4):2),"cache rows confused with observation count");
    require(actual.stats.at("attention_chunks")>0,"fiber never executed packed work");++cases;
  }
  return cases;
}
Index windows(at::Device device) {
  Index cases=0;
  for(int shape:{0,1})for(int variant:{0,1})for(bool prefill:{false,true})
  for(const std::string mode:{"content","old","proposal"})for(bool vectorized:{false,true}) {
    auto f=test::fixture(shape,variant);auto& g=f.graph;
    for(auto& r:g.regions)r.read_mode=mode;
    for(Index n:{0,2}) {
      auto& node=g.nodes[n];node.memory="lh-fiber-attention-sum-repeat-v1";node.full="tanh";
      node.query_heads=node.kv_heads=n==0?1:3;node.clear=variant==0;
      if(n==2)node.readout="norm-fp32-v1";
      weights(f.model.nodes[n],3);
    }
    f.initial.states[{0,0}]=initial(3,1,f.initial.cut-1);
    f.initial.states[{0,0}].observations=(Index(1)<<55)+5;
    for(size_t e=0;e<g.edges.size();++e)if(e%2)g.origins.push_back({Index(e),7,1});
    g.compile();f.initial.identity=g.identity;
    auto l=limits(vectorized?4:1);l.prefill=prefill;l.vectorized_read=vectorized;l.vectorized_state=vectorized;
    ContentFlow flow(g,f.model,f.initial,device,l);Streaming cpu(g,f.model,{});Greedy greedy(g,f.model,{});
    auto q=f.initial;Index previous=q.cut;
    for(Index offset:{2,6,11}) {
      const auto stop=f.initial.cut+offset;std::vector<External> xs;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)xs.push_back(x);
      const auto expected=cpu.run(q,xs,stop,stop);tide_bench::compare(greedy.run(q,xs,stop,stop),expected,true,at::kFloat);
      const auto actual=flow.advance(xs,stop);tide_bench::compare(actual,expected,true,at::kFloat);
      require(actual.stats.at("max_causal_node_time_batch")<=1,"fiber state causal contract violated");
      q=expected.continuation;previous=stop;++cases;
    }
    // A lean device boundary can be consumed without downloading the live cache.
    l.prefill=!prefill;l.diagnostics=false;l.trace=0;l.kv_trace_rows=0;
    ContentFlow restored(g,f.model,flow.snapshot(),device,l);restored.advance_device({},previous+3);
    auto expected=cpu.run(q,{},previous+3,previous+3);
    tide_bench::compare(restored.result(),expected,false,at::kFloat);++cases;
  }
  return cases;
}
Index refusals(at::Device device) {
  Index cases=0;
  for(int kind=0;kind<6;++kind) {
    auto f=one_node(3,1);auto l=limits();
    f.input={{0,0,0,0,at::zeros({3},at::kFloat)},{0,1,0,0,at::zeros({3},at::kFloat)}};
    if(kind==0){l.kv_rows=0;refuses([&]{ContentFlow x(f.graph,f.model,f.initial,device,l);},"invalid fiber cache limits");}
    if(kind==1){l.kv_rows=1;refuses([&]{ContentFlow x(f.graph,f.model,f.initial,device,l);x.advance(f.input,1);},"code=11");}
    if(kind==2){l.kv_trace_rows=1;refuses([&]{ContentFlow x(f.graph,f.model,f.initial,device,l);x.advance(f.input,1);},"code=12");}
    if(kind==3){l.kv_rows=1;f.initial.states[{0,0}]=initial(3,1);refuses([&]{ContentFlow x(f.graph,f.model,f.initial,device,l);},"initial fiber cache exceeds");}
    if(kind==4||kind==5) {
      f.initial.states[{0,0}]=initial(3,1);std::string code="code=5";
      if(kind==4)f.initial.states[{0,0}].observations=std::numeric_limits<Index>::max();
      else {l.max_repeat_ticks=1;for(auto& x:f.input)x.time=3;code="code=8";}
      ContentFlow x(f.graph,f.model,f.initial,device,l);
      refuses([&]{x.advance(f.input,4);},code);refuses([&]{x.snapshot();},"failed");refuses([&]{x.advance({},4);},"failed");
    }
    ++cases;
  }
  return cases;
}
Index lifecycle(at::Device device) {
  Index cases=0;
  for(bool clear:{false,true})for(bool active:{false,true}) {
    auto f=one_node(7,1);auto& node=f.graph.nodes[0];node.clear=clear;node.state_clock={5,1,3};
    f.graph.regions[0].observe_all=false;
    if(!active) {
      // Legal empty selection comes from the declared positive selector, not
      // an invalid zero region budget. Real zero messages still form a fiber.
      f.graph.regions[0].read_mode="content";f.graph.regions[0].selector="positive-v1";
      f.model.nodes[0].read.fill_(-1);
    }
    f.graph.compile();f.initial.identity=f.graph.identity;
    if(!clear)f.initial.states[{0,0}]=initial(7,1);
    Streaming cpu(f.graph,f.model,{});auto l=limits(1);if(clear)l.kv_rows=2;
    ContentFlow flow(f.graph,f.model,f.initial,device,l);auto q=f.initial;
    Index position=0;
    for(Index time:{1,3,6,8,11,13,16,18,21,23}) {
      std::vector<External> xs{{0,0,position,time,at::zeros({7},at::kFloat)},
        {0,1,position,time,at::full({7},.25f,at::kFloat)}};
      auto expected=cpu.run(q,xs,time+1,time+1);auto actual=flow.advance(xs,time+1);
      tide_bench::compare(actual,expected,true,at::kFloat);q=expected.continuation;++position;++cases;
    }
    require(q.states.at({0,0}).slots.at("key").size(0)==(clear?0:active?22:2),"clear/adoption changed cache retention");
  }
  return cases;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("fiber gate requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("fiber gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    const auto a=anchors(device);std::cout<<"fiber anchors="<<a<<" passed\n"<<std::flush;
    const auto b=windows(device);std::cout<<"fiber windows="<<b<<" passed\n"<<std::flush;
    const auto c=refusals(device);std::cout<<"fiber refusals="<<c<<" passed\n"<<std::flush;
    const auto d=lifecycle(device);
    std::cout<<"device-fiber: passed anchors="<<a<<" windows="<<b<<" refusals="<<c<<" lifecycle="<<d<<" scope=FP32_HARD_inference\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
