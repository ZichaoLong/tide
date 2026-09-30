#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <stdexcept>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void weights(NodeWeights& w,Index width,Index heads,Index kv_heads) {
  auto eye=at::eye(width,at::kFloat);const auto kv=width/heads*kv_heads;
  w.extra["attn_q"]=eye*.25f;w.extra["attn_k"]=eye.narrow(1,0,kv)*.5f;
  w.extra["attn_v"]=eye.narrow(1,0,kv).clone();w.extra["attn_out"]=eye.clone();
}
State initial(Index width,Index heads,Index time) {
  return {at::full({width},.25f,at::kFloat),time,(Index(1)<<55)+5,
    {{"key",at::zeros({1,1,width/heads},at::kFloat)},
     {"value",at::full({1,1,width/heads},2.f,at::kFloat)}}};
}
ContentLimits limits() {
  ContentLimits l;l.queue=96;l.arrivals=192;l.outputs=256;l.trace=1024;
  l.kv_rows=32;l.kv_trace_rows=8192;l.attention_chunk_rows=4;l.attention_key_rows=7;
  l.workspace_bytes=256*1024*1024;l.chunk_policy=ChunkPolicy::aggressive;return l;
}
Index anchors(at::Device device) {
  Index cases=0;
  for(Index width:{4,33,257})for(Index window:{0,1,3})for(Index chunk:{1,4})for(bool tiled:{false,true}) {
    test::Fixture f;auto& g=f.graph;g.nodes={{0}};auto& node=g.nodes[0];
    const Index heads=width==4?2:width==33?3:1;
    node.memory="attention";node.full="identity";node.query_heads=heads;node.kv_heads=1;node.window=window;
    g.regions={{1,true,false,"proposal","count-v1"}};g.inputs={0};g.outputs={0};g.compile();
    NodeWeights w{at::zeros({width},at::kFloat),at::eye(width,at::kFloat),at::zeros({width},at::kFloat),at::ones({width},at::kFloat)};
    weights(w,width,heads,1);w.extra["attn_q"].zero_();w.extra["attn_k"].zero_();
    f.model.nodes={w};f.model.input_scale={at::ones({},at::kFloat)};f.model.output_scale={at::ones({},at::kFloat)};
    f.initial.identity=g.identity;f.initial.batch_size=1;f.initial.states[{0,0}]=initial(width,heads,-1);
    Index position=0;for(Index time:{0,1,4,9}) {
      f.input.push_back({0,0,position,time,at::full({width},float(2*position+1),at::kFloat)});++position;
    }
    auto l=limits();l.queue=l.arrivals=l.outputs=16;l.trace=32;l.kv_trace_rows=128;
    l.kv_rows=window?window:8;l.attention_chunk_rows=chunk;l.attention_key_rows=tiled?1:8;
    ContentFlow flow(g,f.model,f.initial,device,l);Streaming cpu(g,f.model,{});
    auto expected=cpu.run(f.initial,f.input,10,10),actual=flow.advance(f.input,10);
    try{tide_bench::compare(actual,expected,true,at::kFloat);}
    catch(...){std::cerr<<"event batch anchor width="<<width<<" window="<<window<<" chunk="<<chunk<<" tiled="<<tiled<<'\n';throw;}
    require(actual.stats.at("device_stages")==1&&actual.stats.at("max_node_time_batch")==4,"device did not form actual node-time batch");
    std::vector<float> prefix{2.f};
    for(size_t i=0;i<actual.trace.size();++i) {
      prefix.push_back(float(2*i+1));if(window&&prefix.size()>size_t(window))prefix.erase(prefix.begin());
      float sum=0;for(float x:prefix)sum+=x;
      require(at::allclose(actual.trace[i].proposal,at::full({width},sum/prefix.size(),at::kFloat),1e-5,1e-6),"batched query changed prefix/window denominator");
    }
    ++cases;
    // Restore the compact final cache under a different physical plan and
    // streaming policy; no replay of preceding inputs supplies the new owner.
    l.prefill=false;l.attention_key_rows=tiled?8:1;l.attention_chunk_rows=chunk==1?4:1;
    ContentFlow restored(g,f.model,flow.snapshot(),device,l);
    std::vector<External> next{{0,0,4,15,at::full({width},9.f,at::kFloat)}};
    tide_bench::compare(restored.advance(next,16),cpu.run(expected.continuation,next,16,16),true,at::kFloat);++cases;
  }
  return cases;
}
Index windows(at::Device device) {
  Index cases=0,multi=0;
  for(int shape:{0,1,2})for(int policy:{0,1,2})for(const std::string mode:{"content","old","proposal"})for(bool prefill:{false,true}) {
    auto f=test::fixture(shape,1);auto& g=f.graph;
    for(auto& r:g.regions){r.observe_all=policy!=1;r.read_mode=mode;r.selector="count-v1";}
    for(Index n=0;n<4;++n) {
      auto& node=g.nodes[n];node.memory="attention";node.full=n%2?"identity":"tanh";
      node.query_heads=n==3?1:3;node.kv_heads=1;node.window=n%2?0:3;node.clear=policy==2&&n%2==0;
      weights(f.model.nodes[n],3,node.query_heads,1);
    }
    f.initial.states[{0,0}]=initial(3,3,f.initial.cut-1);g.compile();f.initial.identity=g.identity;
    auto l=limits();l.prefill=prefill;ContentFlow flow(g,f.model,f.initial,device,l);Streaming cpu(g,f.model,{});
    auto q=f.initial;Index previous=q.cut;
    for(Index step:{2,6,11}) {
      const auto stop=f.initial.cut+step;std::vector<External> xs;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)xs.push_back(x);
      auto expected=cpu.run(q,xs,stop,stop),actual=flow.advance(xs,stop);
      try{tide_bench::compare(actual,expected,true,at::kFloat);}
      catch(...){std::cerr<<"event batch shape="<<shape<<" policy="<<policy<<" Read="<<mode<<" prefill="<<prefill<<" step="<<step<<'\n';throw;}
      require(actual.stats.at("max_causal_node_time_batch")<=1,"selection-dependent cache escaped causal fallback");
      if(prefill&&policy==0&&actual.stats.at("max_node_time_batch")>1)++multi;
      q=expected.continuation;previous=stop;++cases;
    }
    l.prefill=!prefill;ContentFlow restored(g,f.model,flow.snapshot(),device,l);
    tide_bench::compare(restored.advance({},previous+3),cpu.run(q,{},previous+3,previous+3),true,at::kFloat);++cases;
  }
  require(multi>0,"topology-independent event time batching was not exercised");
  std::cout<<"event node-time multi-windows="<<multi<<'\n';return cases;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("event batch gate requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("event batch gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    const auto a=anchors(device);std::cout<<"event batch anchors/restores="<<a<<" passed\n"<<std::flush;
    const auto b=windows(device);
    std::cout<<"device-event-batch: passed anchors/restores="<<a<<" windows/restores="<<b<<" scope=FP32_HARD_inference\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
