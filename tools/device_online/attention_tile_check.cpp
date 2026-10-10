#include "device_backend.h"
#include "content_fixture.h"
#include "tiled_attention.h"
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
test::Fixture fixture(Index width,bool fiber,bool uniform) {
  test::Fixture f;auto& g=f.graph;g.nodes={{0}};auto& n=g.nodes[0];
  n.memory=fiber?"lh-fiber-attention-sum-repeat-v1":"attention";
  n.full="tanh";n.query_heads=width==4?4:width==33?3:1;n.kv_heads=fiber?n.query_heads:1;
  g.regions={{1,true,false,"proposal","count-v1"}};g.inputs={0,0};g.outputs={0};g.compile();
  const auto h=n.query_heads,kh=n.kv_heads,d=width/h,kv=kh*d;
  auto eye=at::eye(width,at::kFloat);
  NodeWeights w{at::zeros({width},at::kFloat),eye*.25f,at::zeros({width},at::kFloat),at::ones({width},at::kFloat)};
  if(fiber) {
    w.extra["fiber_qkv"]=at::cat({eye*(uniform?0.f:.5f),eye*.5f,eye},1);
    w.extra["fiber_qkv_bias"]=at::zeros({3*width},at::kFloat);w.extra["fiber_out"]=eye;
    w.extra["fiber_out_bias"]=at::zeros({width},at::kFloat);w.extra["fiber_decay"]=at::zeros({},at::kFloat);
  }else {
    w.extra["attn_q"]=eye*(uniform?0.f:.5f);w.extra["attn_k"]=eye.narrow(1,0,kv)*.5f;
    w.extra["attn_v"]=eye.narrow(1,0,kv).clone();w.extra["attn_out"]=eye;
  }
  f.model.nodes={w};f.model.input_scale={at::ones({},at::kFloat),at::ones({},at::kFloat)};f.model.output_scale={at::ones({},at::kFloat)};
  f.initial.identity=g.identity;f.initial.batch_size=3;
  for(Index b=0;b<3;++b) {
    const Index rows=b==0?0:b==1?5:257;
    auto key=at::arange(rows*kv,at::kFloat).remainder(11).reshape({rows,kh,d})*.125f;
    auto value=at::arange(rows*kv,at::kFloat).remainder(7).reshape({rows,kh,d})*.25f;
    State state{at::zeros({width},at::kFloat),-1,(Index(1)<<55)+17,{{"key",key},{"value",value}}};
    if(fiber)state.slots["log_bias"]=at::zeros({rows},at::kFloat);
    f.initial.states[{b,0}]=state;
    for(Index t=0;t<2;++t) {
      f.input.push_back({b,0,t,t,at::arange(width,at::kFloat).remainder(5)*.125f+.25f});
      if(b!=1)f.input.push_back({b,1,t,t,at::zeros({width},at::kFloat)});
    }
  }
  return f;
}
Index windows(at::Device device) {
  Index cases=0;
  for(bool fiber:{false,true})for(Index width:{1,4,33,257})for(Index tile:{1,7,128,300}) {
    auto f=fixture(width,fiber,width==1);ContentLimits l;
    l.queue=96;l.arrivals=96;l.outputs=96;l.trace=256;l.kv_rows=300;l.kv_trace_rows=8192;
    l.attention_chunk_rows=4;l.attention_key_rows=tile;l.workspace_bytes=256*1024*1024;
    ContentFlow flow(f.graph,f.model,f.initial,device,l);Streaming cpu(f.graph,f.model,{});auto q=f.initial;
    for(Index time=0;time<2;++time) {
      std::vector<External> xs;for(const auto& x:f.input)if(x.time==time)xs.push_back(x);
      auto expected=cpu.run(q,xs,time+1,time+1),actual=flow.advance(xs,time+1);
      try{tide_bench::compare(actual,expected,true,at::kFloat);}
      catch(const std::exception&){std::cerr<<"tile fiber="<<fiber<<" width="<<width<<" keys="<<tile<<" time="<<time<<'\n';throw;}
      const std::string prefix=fiber?"attention_":"event_attention_";
      require(actual.stats.at(prefix+"key_rows")==tile,"requested key bound not recorded");
      require((actual.stats.at(prefix+"key_tiles")>0)==(tile<300),"tiled/dense path mismatch");
      if(tile<300)require(actual.stats.at(prefix+"tiled_score_entries")>0,"actual key work not counted");
      // Independent exact uniform denominator includes all retained keys and
      // all present current fiber rows, including the numerical zero message.
      if(width==1&&time==0)for(Index b=0;b<3;++b) {
        const auto& old=f.initial.states.at({b,0}).slots.at("value");
        const Index added=fiber&&b!=1?2:1;
        const float mean=(old.sum().item<float>()+.25f)/float(old.size(0)+added);
        const float predicted=fiber?mean*added:mean;
        const auto actual_value=actual.continuation.states.at({b,0}).value.item<float>();
        require(std::abs(actual_value-predicted)<1e-6f,"key tiles changed the global denominator");
      }
      q=expected.continuation;++cases;
    }
    // Restoring with a different physical key size must not change cache state.
    l.attention_key_rows=tile==300?7:300;l.diagnostics=false;l.trace=0;l.kv_trace_rows=0;
    ContentFlow restored(f.graph,f.model,flow.snapshot(),device,l);restored.advance_device({},3);
    tide_bench::compare(restored.result(),cpu.run(q,{},3,3),false,at::kFloat);++cases;
  }
  return cases;
}
Index extremes(at::Device device) {
  Index cases=0;
  for(bool fiber:{false,true})for(Index tile:{1,3,7}) {
    auto f=fixture(1,fiber,false);f.initial.batch_size=1;
    f.initial.states.clear();f.input={{0,0,0,0,at::ones({1},at::kFloat)}};
    State s{at::zeros({1},at::kFloat),-1,4,
      {{"key",at::tensor({-160.f,160.f,-160.f,158.f}).reshape({4,1,1})},
       {"value",at::tensor({1.f,2.f,3.f,4.f}).reshape({4,1,1})}}};
    if(fiber)s.slots["log_bias"]=at::zeros({4},at::kFloat);f.initial.states[{0,0}]=s;
    ContentLimits l;l.attention_key_rows=tile;l.attention_chunk_rows=4;l.kv_rows=16;
    ContentFlow flow(f.graph,f.model,f.initial,device,l);Streaming cpu(f.graph,f.model,{});
    auto actual=flow.advance(f.input,1);tide_bench::compare(actual,cpu.run(f.initial,f.input,1,1),true,at::kFloat);
    require(at::isfinite(actual.trace[0].proposal).all().item<bool>(),"extreme tiled softmax is nonfinite");++cases;
  }
  // Finite repeated decay can produce an all-minus-inf old-key tile while
  // current keys remain finite. Only the latter belong to the denominator.
  for(Index tile:{1,3,7,16}) {
    auto f=fixture(1,true,true);f.initial.batch_size=1;f.initial.states.clear();
    f.graph.nodes[0].clear=true;f.graph.compile();f.initial.identity=f.graph.identity;
    f.model.nodes[0].extra["fiber_decay"]=at::full({},2e38f,at::kFloat);
    f.initial.states[{0,0}]={at::zeros({1},at::kFloat),-1,4,
      {{"key",at::zeros({4,1,1},at::kFloat)},{"value",at::ones({4,1,1},at::kFloat)},
       {"log_bias",at::zeros({4},at::kFloat)}}};
    f.input={{0,0,0,1,at::ones({1},at::kFloat)},{0,1,0,1,at::zeros({1},at::kFloat)}};
    ContentLimits l;l.attention_key_rows=tile;l.kv_rows=16;
    ContentFlow flow(f.graph,f.model,f.initial,device,l);Streaming cpu(f.graph,f.model,{});
    auto expected=cpu.run(f.initial,f.input,2,2),actual=flow.advance(f.input,2);
    // The general benchmark gate intentionally requires finite trace tensors.
    // Check the exceptional bias slots exactly, then apply its unchanged full
    // comparator to every remaining observable using copies of these results.
    require(actual.trace.size()==1&&expected.trace.size()==1,"saturated-bias event count");
    auto checked=actual,reference=expected;
    for(const bool comparison:{false,true}) {
      auto& a=comparison?checked.trace[0].comparison_state:checked.trace[0].proposed_state;
      auto& e=comparison?reference.trace[0].comparison_state:reference.trace[0].proposed_state;
      const auto ab=a.slots.at("log_bias"),eb=e.slots.at("log_bias");
      require(at::equal(ab,eb)&&at::isneginf(ab).sum().item<Index>()==4,"saturated bias slots differ");
      a.slots["log_bias"]=at::zeros_like(ab);e.slots["log_bias"]=at::zeros_like(eb);
    }
    tide_bench::compare(checked,reference,true,at::kFloat);
    require(actual.trace[0].proposal.item<float>()==1.f,"zero-mass old tile poisoned later finite keys");++cases;
  }
  // A requested large key tile is reduced to fit the budget before rejection.
  const auto split=plan_attention_tiles(100,100,10,280,8,4,128,128);
  require(split.keys<128&&split.queries>=1&&split.reserved<=280,"key budget did not split");++cases;
  bool refused=false;try{plan_attention_tiles(100,100,10,209,8,4,128,128);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"minimum attention work was not refused");++cases;
  return cases;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("key tiling requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("key tiling requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    const auto a=windows(device);std::cout<<"key tile windows="<<a<<" passed\n"<<std::flush;
    const auto b=extremes(device);
    std::cout<<"device-attention-tile: passed windows="<<a<<" extreme/budget="<<b<<" scope=FP32_HARD_inference\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
