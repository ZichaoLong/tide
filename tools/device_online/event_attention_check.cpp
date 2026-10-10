#include "device_backend.h"
#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include "tide/greedy.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F> void refuses(F f,const std::string& marker) {
  try{f();}catch(const std::exception& e){if(std::string(e.what()).find(marker)!=std::string::npos)return;throw;}
  throw std::runtime_error("missing event attention refusal: "+marker);
}
ContentLimits limits(Index chunk=4) {
  ContentLimits l;l.queue=96;l.arrivals=192;l.outputs=192;l.trace=1024;
  l.kv_rows=32;l.kv_trace_rows=8192;l.attention_chunk_rows=chunk;l.workspace_bytes=256*1024*1024;
  l.attention_key_rows=chunk==1?1:7;
  return l;
}
void weights(NodeWeights& w,Index width,Index heads,Index kv_heads) {
  auto eye=at::eye(width,at::kFloat);const auto kv=width/heads*kv_heads;
  w.extra["attn_q"]=eye*.25f;w.extra["attn_k"]=eye.narrow(1,0,kv)*.5f;
  w.extra["attn_v"]=eye.narrow(1,0,kv).clone();w.extra["attn_out"]=eye*.5f;
}
State initial(Index width,Index heads,Index kv_heads,Index rows,Index time=-1) {
  const auto kv=kv_heads*width/heads;
  return {at::full({width},.125f,at::kFloat),time,5,
    {{"key",at::arange(rows*kv,at::kFloat).remainder(13).reshape({rows,kv_heads,width/heads})*.015625f},
     {"value",at::arange(rows*kv,at::kFloat).remainder(7).reshape({rows,kv_heads,width/heads})*.03125f}}};
}
test::Fixture one_node(Index width,Index heads,Index kv_heads,Index window) {
  test::Fixture f;auto& g=f.graph;g.nodes={{0}};auto& n=g.nodes[0];
  n.memory="attention";n.full="tanh";n.query_heads=heads;n.kv_heads=kv_heads;n.window=window;
  g.regions={{1,true,false,"proposal","count-v1"}};g.inputs={0,0};g.outputs={0};g.compile();
  NodeWeights w{at::zeros({width},at::kFloat),at::eye(width,at::kFloat)*.5f,at::zeros({width},at::kFloat),at::ones({width},at::kFloat)};
  weights(w,width,heads,kv_heads);f.model.nodes={w};
  f.model.input_scale={at::ones({},at::kFloat),at::ones({},at::kFloat)};f.model.output_scale={at::ones({},at::kFloat)};
  f.initial.identity=g.identity;f.initial.batch_size=1;return f;
}
Index anchors(at::Device device) {
  Index cases=0;
  for(Index width:{1,4,7,33,257}) {
    const Index heads=width==4?4:width==33?3:1;
    for(Index kv_heads=1;kv_heads<=heads;++kv_heads)if(heads%kv_heads==0)
    for(Index window:{0,1,3})for(Index chunk:{1,4})for(bool cached:{false,true}) {
      auto f=one_node(width,heads,kv_heads,window);const Index cache_rows=window==1?1:2;
      if(cached)f.initial.states[{0,0}]=initial(width,heads,kv_heads,cache_rows);
      f.input={{0,0,0,0,at::arange(width,at::kFloat).remainder(5)*.125f+.25f},
        {0,1,0,0,at::zeros({width},at::kFloat)}};
      auto l=limits(chunk);if(window)l.kv_rows=window;
      Streaming cpu(f.graph,f.model,{});auto expected=cpu.run(f.initial,f.input,1,1);
      ContentFlow flow(f.graph,f.model,f.initial,device,l);auto actual=flow.advance(f.input,1);
      try{tide_bench::compare(actual,expected,true,at::kFloat);}
      catch(const std::exception&) {
        std::cerr<<"event case width="<<width<<" heads="<<heads<<" kv_heads="<<kv_heads<<" window="<<window<<" chunk="<<chunk<<" cached="<<cached<<'\n';
        for(const auto* r:{&actual,&expected}) {
          const auto& s=r->continuation.states.at({0,0});
          std::cerr<<"state="<<s.value<<" key="<<s.slots.at("key")<<" value="<<s.slots.at("value")<<'\n';
          if(!r->trace.empty())std::cerr<<"old="<<r->trace[0].old.value<<" proposal="<<r->trace[0].proposal<<'\n';
        }
        throw;
      }
      const auto& s=actual.continuation.states.at({0,0});const Index proposed=(cached?cache_rows:0)+1;
      require(s.slots.size()==2&&s.slots.at("key").size(1)==kv_heads,"event KV is not compact");
      require(s.slots.at("key").size(0)==(window?std::min(window,proposed):proposed),"event cache counted individual messages");++cases;
    }
  }
  // Independent MQA/GQA head-sharing anchor: zero queries give a uniform
  // denominator; two messages produce one new event, not two new KV rows.
  for(Index window:{0,1}) {
    auto f=one_node(4,4,2,window);auto& w=f.model.nodes[0];
    w.extra["attn_q"].zero_();w.extra["attn_k"].zero_();w.extra["attn_out"]=at::eye(4,at::kFloat);
    w.extra["attn_v"]=at::tensor({1.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f}).reshape({4,2});
    f.initial.states[{0,0}]={at::zeros({4},at::kFloat),-1,1,
      {{"key",at::zeros({1,2,1},at::kFloat)},{"value",at::tensor({2.f,10.f}).reshape({1,2,1})}}};
    f.input={{0,0,0,0,at::tensor({1.f,3.f,5.f,7.f})},{0,1,0,0,at::zeros({4},at::kFloat)}};
    ContentFlow flow(f.graph,f.model,f.initial,device,limits(1));auto actual=flow.advance(f.input,1);
    const auto exact=window?at::tensor({1.f,1.f,7.f,7.f}):at::tensor({1.5f,1.5f,8.5f,8.5f});
    require(at::equal(actual.trace[0].proposal,exact),"GQA head-sharing/window analytic anchor failed");++cases;
  }
  return cases;
}
Index windows(at::Device device) {
  Index cases=0;
  for(int shape:{0,1})for(int variant:{0,1})for(bool prefill:{false,true})
  for(const std::string mode:{"content","old","proposal"})for(bool fiber:{false,true}) {
    auto f=test::fixture(shape,variant);auto& g=f.graph;
    for(auto& r:g.regions)r.read_mode=mode;
    for(Index n:{0,2}) {
      auto& node=g.nodes[n];node.memory="attention";node.full="tanh";
      node.query_heads=3;node.kv_heads=n==0?1:3;node.window=n==0?2:0;node.clear=variant==0;
      if(n==2)node.readout="norm-fp32-v1";weights(f.model.nodes[n],3,3,node.kv_heads);
    }
    if(fiber) {
      auto& n=g.nodes[3];n.memory="lh-fiber-attention-all-softmax-repeat-v1";n.full="tanh";n.query_heads=n.kv_heads=1;
      auto& w=f.model.nodes[3];auto eye=at::eye(3,at::kFloat);
      w.extra["fiber_qkv"]=at::cat({eye*.25f,eye*.5f,eye},1);w.extra["fiber_qkv_bias"]=at::zeros({9},at::kFloat);
      w.extra["fiber_out"]=eye*.5f;w.extra["fiber_out_bias"]=at::zeros({3},at::kFloat);
      w.extra["fiber_decay"]=at::zeros({},at::kFloat);w.extra["fiber_pool"]=at::zeros({g.source_counts[3]},at::kFloat);
    }
    f.initial.states[{0,0}]=initial(3,3,1,2,f.initial.cut-1);
    f.initial.states[{0,0}].observations=(Index(1)<<55)+5;
    for(size_t e=0;e<g.edges.size();++e)if(e%2)g.origins.push_back({Index(e),7,1});
    g.compile();f.initial.identity=g.identity;
    auto l=limits(prefill?4:1);l.prefill=prefill;l.vectorized_read=prefill;l.vectorized_state=prefill;
    ContentFlow flow(g,f.model,f.initial,device,l);Streaming cpu(g,f.model,{});Greedy greedy(g,f.model,{});
    auto q=f.initial;Index previous=q.cut;
    for(Index offset:{2,6,11}) {
      const auto stop=f.initial.cut+offset;std::vector<External> xs;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)xs.push_back(x);
      const auto expected=cpu.run(q,xs,stop,stop);tide_bench::compare(greedy.run(q,xs,stop,stop),expected,true,at::kFloat);
      const auto actual=flow.advance(xs,stop);tide_bench::compare(actual,expected,true,at::kFloat);
      require(actual.stats.at("max_causal_node_time_batch")<=1,"event attention causal contract violated");
      q=expected.continuation;previous=stop;++cases;
    }
    l.prefill=!prefill;l.diagnostics=false;l.trace=0;l.kv_trace_rows=0;
    ContentFlow restored(g,f.model,flow.snapshot(),device,l);restored.advance_device({},previous+3);
    tide_bench::compare(restored.result(),cpu.run(q,{},previous+3,previous+3),false,at::kFloat);++cases;
  }
  return cases;
}
Index lifecycle(at::Device device) {
  Index cases=0;
  for(bool clear:{false,true})for(bool active:{false,true}) {
    auto f=one_node(4,4,2,clear?0:1);f.graph.nodes[0].clear=clear;f.graph.nodes[0].state_clock={5,1,3};
    f.graph.regions[0].observe_all=false;
    if(!active){f.graph.regions[0].read_mode="content";f.graph.regions[0].selector="positive-v1";f.model.nodes[0].read.fill_(-1);}
    f.graph.compile();f.initial.identity=f.graph.identity;auto l=limits(1);l.kv_rows=1;
    ContentFlow flow(f.graph,f.model,f.initial,device,l);Streaming cpu(f.graph,f.model,{});auto q=f.initial;Index position=0;
    for(Index time:{1,3,6,8,11,13,16,18,21,23}) {
      std::vector<External> xs{{0,0,position,time,at::full({4},.25f,at::kFloat)},{0,1,position,time,at::zeros({4},at::kFloat)}};
      auto expected=cpu.run(q,xs,time+1,time+1);auto actual=flow.advance(xs,time+1);
      tide_bench::compare(actual,expected,true,at::kFloat);q=expected.continuation;++position;++cases;
    }
    require(q.states.at({0,0}).slots.at("key").size(0)==(!clear&&active?1:0),"event selected-only/clear retained wrong cache");
  }
  return cases;
}
Index refusals(at::Device device) {
  Index cases=0;
  for(int kind=0;kind<5;++kind) {
    auto f=one_node(4,4,2,0);auto l=limits();
    f.input={{0,0,0,0,at::ones({4},at::kFloat)},{0,0,1,1,at::ones({4},at::kFloat)}};
    if(kind==0){l.kv_rows=0;refuses([&]{ContentFlow x(f.graph,f.model,f.initial,device,l);},"invalid event attention cache limits");}
    if(kind==1){l.kv_rows=1;ContentFlow x(f.graph,f.model,f.initial,device,l);refuses([&]{x.advance(f.input,2);},"code=11");refuses([&]{x.snapshot();},"failed");}
    if(kind==2){l.kv_trace_rows=1;ContentFlow x(f.graph,f.model,f.initial,device,l);refuses([&]{x.advance(f.input,2);},"code=12");}
    if(kind==3){l.kv_rows=1;f.initial.states[{0,0}]=initial(4,4,2,2);refuses([&]{ContentFlow x(f.graph,f.model,f.initial,device,l);},"initial event attention cache exceeds");}
    if(kind==4){f.initial.states[{0,0}]=initial(4,4,2,2);f.initial.states[{0,0}].observations=std::numeric_limits<Index>::max();
      ContentFlow x(f.graph,f.model,f.initial,device,l);refuses([&]{x.advance(f.input,2);},"code=5");refuses([&]{x.snapshot();},"failed");}
    ++cases;
  }
  return cases;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("event attention gate requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("event attention gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    const auto a=anchors(device);std::cout<<"event attention anchors="<<a<<" passed\n"<<std::flush;
    const auto b=windows(device);std::cout<<"event attention windows="<<b<<" passed\n"<<std::flush;
    const auto c=lifecycle(device);std::cout<<"event attention lifecycle="<<c<<" passed\n"<<std::flush;
    const auto d=refusals(device);
    std::cout<<"device-event-attention: passed anchors="<<a<<" windows="<<b<<" lifecycle="<<c<<" refusals="<<d<<" scope=FP32_HARD_inference\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
