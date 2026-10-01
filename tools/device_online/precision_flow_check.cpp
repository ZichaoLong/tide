#include "content_fixture.h"
#include "tide/resident.h"
#include "tide/resident_training.h"
#include "tide/stream.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <stdexcept>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
test::Fixture fixture(int shape,int kind,at::ScalarType dtype) {
  auto f=test::fixture(shape,1);auto eye=at::eye(3,at::kFloat);
  const std::vector<std::string> pools{"sum","mean","linear","active-softmax","all-softmax"};
  const std::vector<std::string> aggregates{"mean","weighted_mean","active_softmax","all_softmax"};
  for(Index n=0;n<4;++n) {
    auto& node=f.graph.nodes[n];auto& w=f.model.nodes[n];
    node.readout="norm-fp32-v1";node.full="tanh";w.weight=eye*.5f;
    if(kind==1) {node.memory="lh-add-repeat-v1";w.extra["add_retention"]=at::full({},.5f,at::kFloat);}
    if(kind==2) {
      node.memory="attention";node.query_heads=3;node.kv_heads=n%2?3:1;node.window=n%2?0:3;
      w.extra["attn_q"]=eye*.25f;w.extra["attn_k"]=eye.narrow(1,0,node.kv_heads)*.5f;
      w.extra["attn_v"]=eye.narrow(1,0,node.kv_heads).clone();w.extra["attn_out"]=eye*.5f;
    }
    if(kind>=3&&kind<=7) {
      node.memory="lh-fiber-attention-"+pools[kind-3]+"-repeat-v1";
      node.query_heads=node.kv_heads=n%2?3:1;
      w.extra["fiber_qkv"]=at::cat({eye*.25f,eye*.5f,eye},1);
      w.extra["fiber_qkv_bias"]=at::arange(9,at::kFloat)*.015625f;
      w.extra["fiber_out"]=eye*.5f;w.extra["fiber_out_bias"]=at::full({3},.03125f,at::kFloat);
      w.extra["fiber_decay"]=at::full({},.03125f,at::kFloat);
      if(kind>=5)w.extra["fiber_pool"]=at::arange(f.graph.source_counts[n],at::kFloat)*.125f-.25f;
    }
    if(kind==8) {
      node.aggregation=aggregates[n];
      if(n>0)for(Index slot=0;slot<f.graph.source_counts[n];++slot)
        w.extra[(n==1?"agg_mass_":"agg_logit_")+std::to_string(slot)]=at::full({},.25f*slot-.5f,at::kFloat);
    }
    if(kind==9) {
      node.full="swiglu";
      w.extra["ffn_gate"]=at::cat({eye*.5f,eye*-.25f},1);
      w.extra["ffn_up"]=at::cat({eye*.25f,eye*.5f},1);
      w.extra["ffn_down"]=at::cat({eye*.5f,eye*.25f},0);
      node.emission="slot_affine";node.emit_period=2;
      for(Index slot=0;slot<f.graph.outgoing_ports.offsets[n+1]-f.graph.outgoing_ports.offsets[n];++slot) {
        node.emit_phases.push_back(slot%2?-2:-1);
        w.extra["emit_w_"+std::to_string(slot)]=eye*.5f;
        w.extra["emit_b_"+std::to_string(slot)]=at::full({3},.03125f,at::kFloat);
      }
    }
    if(kind>=10) {
      const std::vector<std::string> activations{"relu","silu","identity"},norms{"identity","rms","layer"};
      node.full="lh-"+activations[kind-10]+"-"+norms[n%3]+"-v1";
      w.extra["lh_norm_weight"]=at::tensor({.75f,1.f,1.25f});
      w.extra["lh_norm_bias"]=at::tensor({.03125f,-.015625f,0.f});
    }
  }
  for(auto& region:f.graph.regions)region.read_mode=kind%2?"content":"proposal";
  f.graph.compile();f.initial.identity=f.graph.identity;
  // Large int64 clocks exercise identity without asking the independent CPU
  // Add kernel to replay 2^55 empty ticks from its default initial clock.
  if(kind==1)for(Index b=0;b<2;++b)for(Index n=0;n<4;++n)
    if(!f.initial.states.count({b,n}))f.initial.states[{b,n}]={at::zeros({3},at::kFloat),f.initial.cut-1,0};
  // Initial caches belong to this common public fixture, never to an oracle run.
  if(kind>=2&&kind<=7)for(auto& [owner,state]:f.initial.states) {
    const auto& node=f.graph.nodes[owner.second];const Index h=node.kv_heads;
    state.slots={{"key",at::full({2,h,3/node.query_heads},.125f,at::kFloat)},
                 {"value",at::full({2,h,3/node.query_heads},.25f,at::kFloat)}};
    if(kind>=3)state.slots["log_bias"]=at::tensor({-.03125f,.0625f});
  }
  test::model_dtype(f.model,dtype);
  for(auto& [_,s]:f.initial.states){s.value=s.value.to(dtype);for(auto& [__,v]:s.slots)v=v.to(dtype);}
  for(auto& x:f.input)x.value=x.value.to(dtype);
  return f;
}
void compare(const Result& actual,const Result& expected,at::ScalarType dtype,bool trace=true) {
  tide_bench::compare(actual,expected,trace,dtype,std::nullopt,dtype==at::kHalf?2e-2:1e-5,dtype==at::kHalf?2e-3:1e-6);
  for(const auto& [_,s]:actual.continuation.states) {
    require(s.value.scalar_type()==dtype,"resident changed state dtype");
    for(const auto& [__,v]:s.slots)require(v.scalar_type()==dtype,"resident changed cache dtype");
  }
  for(const auto& a:actual.continuation.pending)require(a.value.scalar_type()==dtype,"resident changed message dtype");
  for(const auto& e:actual.trace) {
    require(e.content.scalar_type()==dtype&&e.control.scalar_type()==dtype,"resident changed exported payload/control dtype");
    require(e.descriptor.scalar_type()==at::kFloat,"resident changed FP32 Read dtype");
  }
}
void check(at::Device device,at::ScalarType dtype,bool smoke) {
  Index cases=0,windows=0;
  for(int shape:{0,1,3})for(int kind=0;kind<13;++kind)for(bool prefill:{false,true}) {
    if(smoke&&(shape!=1||(kind!=2&&kind!=7)||!prefill))continue;
    auto f=fixture(shape,kind,dtype);ResidentLimits l;
    l.queue=128;l.arrivals=256;l.outputs=256;l.trace=2048;l.kv_rows=96;l.kv_trace_rows=16384;
    l.prefill=prefill;l.attention_key_rows=prefill?7:96;l.attention_chunk_rows=prefill?4:1;
    l.full_chunk_rows=prefill?4:1;l.workspace_bytes=512*1024*1024;
    l.chunk_policy=prefill?ChunkPolicy::aggressive:ChunkPolicy::conservative;
    l.vectorized_state=prefill;l.vectorized_read=prefill;l.vectorized_aggregate=prefill;
    std::cout<<"precision flow shape="<<shape<<" kind="<<kind<<" prefill="<<prefill<<std::endl;
    ResidentSession candidate(f.graph,f.model,f.initial,device,l);Streaming cpu(f.graph,f.model,{});
    auto q=f.initial;Index previous=q.cut;
    for(Index offset:{2,6,11}) {
      const auto stop=f.initial.cut+offset;std::vector<External> xs;
      for(const auto& x:f.input)if(previous<=x.time&&x.time<stop)xs.push_back(x);
      const auto expected=cpu.run(q,xs,stop,stop);
      if(offset==6)for(auto& x:xs)x.value=x.value.to(device);
      const auto view=candidate.advance(xs,stop,stop);
      require(view.values.scalar_type()==dtype&&view.values.device()==device,"device output contract changed");
      try{compare(candidate.result(),expected,dtype);}
      catch(...){std::cerr<<"precision flow shape="<<shape<<" kind="<<kind<<" prefill="<<prefill<<" offset="<<offset<<'\n';throw;}
      q=expected.continuation;previous=stop;++windows;
    }
    // Restore only the candidate's own continuation, change physical batching,
    // and keep recursive pending messages alive across the new owner.
    auto saved=candidate.snapshot();candidate.close();l.prefill=!prefill;l.attention_key_rows=prefill?96:1;
    l.diagnostics=false;l.trace=0;l.kv_trace_rows=0;
    ResidentSession restored(f.graph,f.model,saved,device,l);
    const auto expected=cpu.run(q,{},previous+3,previous+3);restored.advance({},previous+3,previous+3);
    compare(restored.result(),expected,dtype,false);++windows;++cases;
  }
  std::cout<<"precision-flow: passed configurations="<<cases<<" windows="<<windows
    <<" source=independent_CPU_streaming schedules=streaming,greedy scope="<<(smoke?"profile-smoke":"HARD_inference")<<'\n';
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    bool smoke=false;std::vector<char*> argsv{argv[0]};
    for(int i=1;i<argc;++i)if(std::string(argv[i])=="--profile-smoke")smoke=true;else argsv.push_back(argv[i]);
    auto args=portable_torch::parse_cli(argsv.size(),argsv.data(),true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("precision flow requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;auto d=portable_torch::resolve_device(args);
    if(d.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("precision flow requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;check(d,args.dtype,smoke);runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
