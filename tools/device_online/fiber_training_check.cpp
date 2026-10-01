#include "training_test.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <array>
#include <iostream>
#include <limits>

namespace {
using namespace tide;using namespace tide::device_online;
const std::array<std::string,5> kinds{"sum","mean","linear","active-softmax","all-softmax"};
test::Fixture fixture(int shape,int variant,int kind,Index width=4,bool mixed=false) {
  auto f=test::retained_fixture(shape,variant,width);auto& g=f.graph;
  if(variant) {
    // Dense legal logical domains, permuted independently of physical edge identity.
    for(size_t i=0;i<g.inputs.size();++i)g.source_domain->input[i]=g.source_counts[g.inputs[i]]-1-g.source_domain->input[i];
    for(size_t i=0;i<g.edges.size();++i)g.source_domain->edge_target[i]=g.source_counts[g.edges[i].target]-1-g.source_domain->edge_target[i];
    g.compile();
  }
  for(Index n:{0,2,3}) {
    auto& node=g.nodes[n];auto& w=f.model.nodes[n];node.identity=false;
    node.memory="lh-fiber-attention-"+kinds[kind]+"-repeat-v1";node.query_heads=node.kv_heads=width==4?(n==3?2:4):1;
    node.window=0;node.clear=variant&&n!=3;auto eye=at::eye(width,at::kFloat);
    w.extra.clear();w.extra["fiber_qkv"]=at::cat({eye*.25f,eye*.5f,eye*.75f},1);
    w.extra["fiber_qkv_bias"]=at::arange(3*width,at::kFloat).remainder(5)*.00390625f;
    w.extra["fiber_out"]=eye*.5f;w.extra["fiber_out_bias"]=at::full({width},.03125f);
    w.extra["fiber_decay"]=at::full({},.03125f);
    if(kind>=2)w.extra["fiber_pool"]=at::arange(g.source_counts[n],at::kFloat)*.125f-.125f;
    for(Index b=0;b<f.initial.batch_size;++b) {
      const Index rows=b?0:3;auto& s=f.initial.states.at({b,n});
      s.slots={{"key",at::arange(rows*width,at::kFloat).remainder(7).reshape({rows,node.kv_heads,width/node.query_heads})/32.},
        {"value",at::arange(rows*width,at::kFloat).remainder(11).reshape({rows,node.kv_heads,width/node.query_heads})/16.},
        {"log_bias",at::arange(rows,at::kFloat)/32.}};
    }
  }
  f.model.nodes[2].extra["fiber_qkv"]=f.model.nodes[0].extra.at("fiber_qkv");
  f.model.nodes[2].extra["fiber_out"]=f.model.nodes[0].weight; // Cross-role Full/state alias.
  f.model.nodes[2].extra["fiber_decay"]=f.model.nodes[0].extra.at("fiber_decay");
  if(mixed) {
    auto& n=g.nodes[3];n.memory="attention";n.window=2;auto eye=at::eye(width,at::kFloat);auto& w=f.model.nodes[3];
    w.extra={{"attn_q",eye*.25f},{"attn_k",eye*.5f},{"attn_v",eye*.75f},{"attn_out",eye*.5f}};
    for(Index b=0;b<f.initial.batch_size;++b) {
      auto& s=f.initial.states.at({b,3});s.slots.erase("log_bias");s.observations=std::max<Index>(s.observations,3);
    }
    n.window=4;
  }
  g.compile();f.initial.identity=g.identity;f.model=test::train_model(f.model,at::kFloat);return f;
}
ResidentTrainingLimits limits(bool prefill,Index width) {
  ResidentTrainingLimits l;l.forward.prefill=prefill;l.forward.queue=96;l.forward.arrivals=192;l.forward.outputs=192;
  l.forward.trace=512;l.forward.kv_rows=64;l.forward.kv_trace_rows=8192;l.forward.attention_key_rows=2;
  l.forward.attention_chunk_rows=3;l.forward.workspace_bytes=256*1024*1024;l.reverse_chunk_rows=3;
  l.backward_bytes=Index(width>64?8:1)*1024*1024*1024;return l;
}
void roots(at::Device device,test::Fixture f,bool prefill,int mode,const std::string& emit="hard") {
  at::NoGradGuard guard;const auto width=f.model.nodes[0].bias.numel();auto l=limits(prefill,width);l.forward.mode=emit;l.forward.zeta=.75;
  ResidentTrainingSession session(f.graph,f.model,f.initial,device,ResidentOptimizerKind::sgd,{},l);
  std::vector<ResidentCotangents> cot;Index start=f.initial.cut;int window=0;std::vector<Result> actual;
  Options o;o.mode=emit;o.zeta=.75;
  for(auto stop:test::retained_stops(start)) {
    std::vector<External> input;for(auto x:f.input)if(x.time>=start&&x.time<stop)input.push_back(x);
    auto w=session.advance(input,stop,stop);auto r=test::train_roots(w,window,mode);
    if(mode==1){r.outputs=at::full_like(w.outputs.values,.0625);r.outputs_connected=w.outputs.valid.clone();}
    for(size_t j=0;j<r.cache.size();++j) {
      const auto& c=w.cache[j];auto rows=at::arange(c.key.size(1),c.lengths.options());
      auto padding=rows.unsqueeze(0)>=c.lengths.unsqueeze(1);
      for(auto* x:{&r.cache[j].key,&r.cache[j].value})if(x->defined())
        x->masked_fill_(padding.unsqueeze(-1).unsqueeze(-1),std::numeric_limits<float>::quiet_NaN());
      if(r.cache[j].log_bias.defined())r.cache[j].log_bias.masked_fill_(padding,std::numeric_limits<float>::quiet_NaN());
    }
    // Incomplete masks must fail without consuming retained windows.
    if(window==0) {
      auto bad=r;bad.cache.resize(w.cache.size());bad.cache.back().log_bias=Tensor{};bad.cache.back().log_bias_connected=w.cache.back().present;
      std::vector<ResidentCotangents> invalid{bad};
      test::train_reject([&]{session.backward(invalid);},"incomplete log-bias root accepted");
    }
    cot.push_back(r);actual.push_back(session.result());start=stop;++window;
  }
  auto gradient=session.backward(cot);
  for(auto dtype:{at::kFloat,at::kDouble}) {
    auto expected=test::retained_reference(f,mode,dtype,o);
    if(dtype==at::kFloat)for(size_t i=0;i<actual.size();++i)tide_bench::compare(actual[i],expected.windows[i],true,at::kFloat);
    test::train_gradients(gradient,expected,f);
  }
  session.detach();session.close();
}
void clock_case(at::Device device,Index width) {
  auto f=fixture(2,0,4,width);f.graph.nodes[0].state_clock={5,1,3};f.initial.states.at({0,0}).last_time=-1;
  f.initial.states.at({1,0}).last_time=-1;
  for(auto& x:f.input)if(x.port==0)x.time=x.time==0?1:x.time==1?3:6;
  f.graph.compile();f.initial.identity=f.graph.identity;roots(device,std::move(f),true,9);
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    bool profile=false;std::vector<char*> arguments{argv[0]};
    for(int i=1;i<argc;++i)if(std::string(argv[i])=="--profile-smoke")profile=true;else arguments.push_back(argv[i]);
    auto args=portable_torch::parse_cli(arguments.size(),arguments.data(),true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);std::cout<<"--profile-smoke  bounded cache/optimizer placement trace, not the full gate\n";return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("fiber training requires explicit NPU FP32");
    const auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0,trajectories=0;
    if(profile) {
      roots(device,fixture(0,1,4,4,true),true,9);
      test::train_trajectory(device,fixture(0,1,4),true,ResidentOptimizerKind::adamw,at::kFloat,1e-5);
      std::cout<<"resident-fiber-training: passed scope=profile-smoke root_cases=1 trajectories=1\n";
      runtime.close();return 0;
    }
    for(int kind=0;kind<5;++kind)for(int variant:{0,1})for(bool prefill:{false,true})for(int mode:{0,1,4,6,7,8,9,10}) {
      const auto shape=(kind+variant)%2;
      try{roots(device,fixture(shape,variant,kind),prefill,mode);++cases;}
      catch(...){std::cerr<<"fiber roots pool="<<kinds[kind]<<" shape="<<shape<<" variant="<<variant<<" prefill="<<prefill<<" mode="<<mode<<'\n';throw;}
      if(mode==10)std::cout<<"PASS fiber roots pool="<<kinds[kind]<<" variant="<<variant<<" prefill="<<prefill<<'\n'<<std::flush;
    }
    for(const std::string emit:{"hst","softp"})for(int variant:{0,1})for(bool prefill:{false,true}) {
      roots(device,fixture(0,variant,4),prefill,9,emit);++cases;
    }
    for(bool prefill:{false,true}){roots(device,fixture(0,1,3,4,true),prefill,9);++cases;}
    for(Index width:{1,257}){clock_case(device,width);++cases;}
    for(int kind=0;kind<5;++kind)for(bool prefill:{false,true})for(auto opt:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw}) {
      const auto dtype=prefill?at::kDouble:at::kFloat;
      try{test::train_trajectory(device,fixture(0,1,kind),prefill,opt,dtype,1e-5);++trajectories;}
      catch(...){std::cerr<<"fiber trajectory pool="<<kinds[kind]<<" prefill="<<prefill<<" optimizer="<<int(opt)<<" dtype="<<dtype<<'\n';throw;}
    }
    std::cout<<"resident-fiber-training: passed root_cases="<<cases<<" trajectories="<<trajectories
      <<" CPU=FP32_FP64 retained_KV_bias=true shared_parameters=true optimizer_resume=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
