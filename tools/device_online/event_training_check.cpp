#include "training_test.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <iostream>
#include <limits>

namespace {
using namespace tide;using namespace tide::device_online;
test::Fixture fixture(int shape,int variant,Index width=4) {
  auto f=test::retained_fixture(shape,variant,width);
  for(Index n:{0,2,3}) {
    auto& node=f.graph.nodes[n];auto& w=f.model.nodes[n];node.identity=false;node.memory="attention";
    node.query_heads=width==4?(n==3?2:4):1;node.kv_heads=width==4?(n==3?1:2):1;
    node.window=n==0?(variant?1:3):n==2?0:2;node.clear=variant&&n!=3;
    const auto kv=width/node.query_heads*node.kv_heads;auto eye=at::eye(width,at::kFloat);
    w.extra.clear();w.extra["attn_q"]=eye*.25f;w.extra["attn_k"]=eye.narrow(1,0,kv).clone()*.5f;
    w.extra["attn_v"]=eye.narrow(1,0,kv).clone()*.75f;w.extra["attn_out"]=eye*.5f;
    for(Index b=0;b<f.initial.batch_size;++b) {
      const Index rows=b==1?0:node.window==1?1:2;
      auto& s=f.initial.states.at({b,n});
      s.observations=std::max<Index>(s.observations,rows);
      s.slots={{"key",at::arange(rows*kv,at::kFloat).remainder(7).reshape({rows,node.kv_heads,width/node.query_heads})/32.},
        {"value",at::arange(rows*kv,at::kFloat).remainder(11).reshape({rows,node.kv_heads,width/node.query_heads})/16.}};
    }
  }
  f.model.nodes[2].extra["attn_q"]=f.model.nodes[0].extra.at("attn_q");
  f.model.nodes[2].extra["attn_v"]=f.model.nodes[0].extra.at("attn_k"); // Cross-role alias.
  f.graph.compile();f.initial.identity=f.graph.identity;f.model=test::train_model(f.model,at::kFloat);return f;
}
ResidentTrainingLimits limits(bool prefill) {
  ResidentTrainingLimits l;l.forward.prefill=prefill;l.forward.queue=96;l.forward.arrivals=192;l.forward.outputs=192;
  l.forward.trace=512;l.forward.kv_rows=32;l.forward.kv_trace_rows=8192;l.forward.attention_key_rows=2;
  l.forward.attention_chunk_rows=3;l.forward.workspace_bytes=256*1024*1024;
  l.backward_bytes=Index(1)*1024*1024*1024;l.reverse_chunk_rows=3;return l;
}
void roots(at::Device device,int shape,int variant,bool prefill,int mode,const std::string& emit="hard",Index width=4) {
  at::NoGradGuard guard;auto f=fixture(shape,variant,width);auto l=limits(prefill);l.forward.mode=emit;l.forward.zeta=.75;
  if(width>64)l.backward_bytes=Index(8)*1024*1024*1024;
  ResidentTrainingSession session(f.graph,f.model,f.initial,device,ResidentOptimizerKind::sgd,{},l);
  std::vector<ResidentCotangents> cot;Index start=f.initial.cut;int window=0;
  Options o;o.mode=emit;o.zeta=.75;std::vector<Result> actual;
  for(auto stop:test::retained_stops(start)) {
    std::vector<External> input;for(auto x:f.input)if(x.time>=start&&x.time<stop)input.push_back(x);
    auto w=session.advance(input,stop,stop);auto r=test::train_roots(w,window,mode);
    if(mode==1){r.outputs=at::full_like(w.outputs.values,.0625);r.outputs_connected=w.outputs.valid.clone();}
    // Disconnected and padded roots may be poisoned; masking must precede math.
    for(size_t j=0;j<r.cache.size();++j)for(bool key:{false,true}) {
      auto& x=key?r.cache[j].key:r.cache[j].value;if(!x.defined())continue;
      const auto& cache=w.cache[j];auto rows=at::arange(cache.key.size(1),cache.lengths.options());
      auto padding=rows.unsqueeze(0)>=cache.lengths.unsqueeze(1);
      x.masked_fill_(padding.unsqueeze(-1).unsqueeze(-1),std::numeric_limits<float>::quiet_NaN());
    }
    cot.push_back(r);actual.push_back(session.result());start=stop;++window;
  }
  auto wrong=cot;wrong[0].cache.resize(3);
  test::train_reject([&]{session.backward(wrong);},"invalid cache group roots accepted");
  auto gradient=session.backward(cot);
  for(auto dtype:{at::kFloat,at::kDouble}) {
    auto expected=test::retained_reference(f,mode,dtype,o);
    if(dtype==at::kFloat)for(size_t i=0;i<actual.size();++i)tide_bench::compare(actual[i],expected.windows[i],true,at::kFloat);
    test::train_gradients(gradient,expected,f);
  }
  session.detach();session.close();
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("event training requires explicit NPU FP32");
    const auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0,trajectories=0;
    for(int shape:{0,1})for(int variant:{0,1})for(bool prefill:{false,true})for(int mode:{0,1,4,6,7,8,9}) {
      try{roots(device,shape,variant,prefill,mode);++cases;
        if(mode==9)std::cout<<"PASS event roots shape="<<shape<<" variant="<<variant<<" prefill="<<prefill<<'\n'<<std::flush;}
      catch(...){std::cerr<<"event roots shape="<<shape<<" variant="<<variant<<" prefill="<<prefill<<" mode="<<mode<<'\n';throw;}
    }
    for(const std::string mode:{"hst","softp"})for(int variant:{0,1})for(bool prefill:{false,true}) {
      try{roots(device,0,variant,prefill,9,mode);++cases;}
      catch(...){std::cerr<<"event control="<<mode<<" variant="<<variant<<" prefill="<<prefill<<'\n';throw;}
    }
    for(Index width:{1,257}){roots(device,1,0,true,9,"hard",width);++cases;}
    for(bool prefill:{false,true})for(auto kind:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw})for(auto dtype:{at::kFloat,at::kDouble}) {
      try{test::train_trajectory(device,fixture(0,1),prefill,kind,dtype,1e-5);++trajectories;}
      catch(...){std::cerr<<"event trajectory prefill="<<prefill<<" optimizer="<<int(kind)<<" dtype="<<dtype<<'\n';throw;}
    }
    std::cout<<"resident-event-training: passed root_cases="<<cases<<" trajectories="<<trajectories
      <<" CPU=FP32_FP64 retained_KV=true shared_QKV=true optimizer_resume=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
