#include "training_test.h"
#include "emission_training_fixture.h"
#include "precision_graph_fixture.h"
#include "precision_graph_profiles.h"
#include "retained_cache_fixture.h"
#include "retained_attention_check.h"
#include "retained_full_check.h"
#include "reverse_gather_check.h"
#include "parameter_accumulate_check.h"
#include "full_vjp_fixture.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>
namespace {
using namespace tide;using namespace tide::device_online;
void require(bool yes,const char* why){test::train_require(yes,why);}
void exact(const ResidentOptimizerState& a,const ResidentOptimizerState& b) {
  for(const auto& pair:{std::make_pair(a.values,b.values),{a.first,b.first},{a.second,b.second},
      {a.maximum,b.maximum},{a.steps,b.steps},{a.corrections,b.corrections}})
    require(at::equal(pair.first,pair.second),"portable optimizer state changed on refusal/restore/None step");
}
void updated(const ResidentTrainingCheckpoint& c,const Model& master,const NamedOptimizer& optimizer,bool half) {
  const auto all=master.parameters(false),registry=master.parameters(true);const auto owners=registry.owners();auto yes=at::ones({},at::kBool);
  require(c.aliases==all.alias_partitions()&&c.trainable==registry.names(),"checkpoint owner identities changed");
  for(const auto& o:all.owners())test::full_same_precision(c.parameters.at(o.canonical).to(at::kFloat),yes,
    o.value.to(half?at::kHalf:at::kFloat).to(at::kFloat),o.canonical.c_str(),half);
  for(size_t i=0;i<owners.size();++i) {
    const auto& o=owners[i];const auto offset=c.offsets[i];
    auto same=[&](const Tensor& a,const Tensor& b){if(b.defined())test::full_same_precision(a.narrow(0,offset,b.numel()).reshape(b.sizes()),yes,b,o.canonical.c_str(),half);};
    if(offset>=0) {
      same(c.state.values,o.value);
      require(at::equal(c.state.values.narrow(0,offset,o.value.numel()).reshape(o.value.sizes()).to(half?at::kHalf:at::kFloat),c.parameters.at(o.canonical)),"payload differs from master publication");
    }
    const auto found=optimizer.state().find(o.canonical);
    if(found==optimizer.state().end()){require(!c.state.steps[i].item<Index>(),"None acquired optimizer progress");continue;}
    const auto& s=found->second;same(c.state.first,s.momentum_buffer.defined()?s.momentum_buffer:s.exp_avg);
    same(c.state.second,s.exp_avg_sq);same(c.state.maximum,s.max_exp_avg_sq);
    if(c.optimizer==ResidentOptimizerKind::adamw)require(c.state.steps[i].item<Index>()==s.step,"Adam counter differs");
  }
}
ResidentTrainingLimits limits(ResidentPlacement placement,bool prefill,bool cache,const std::string& emit) {
  ResidentTrainingLimits l;l.placement=std::move(placement);l.forward.prefill=prefill;l.forward.mode=emit;l.forward.zeta=.75;
  l.forward.trace=512;l.forward.full_chunk_rows=3;l.reverse_chunk_rows=3;l.forward.workspace_bytes=Index(1)*1024*1024*1024;
  l.backward_bytes=Index(8)*1024*1024*1024;l.retained_bytes=512*1024*1024;l.optimizer_bytes=512*1024*1024;
  if(cache){l.forward.queue=96;l.forward.arrivals=192;l.forward.outputs=192;l.forward.kv_rows=128;
    l.forward.kv_trace_rows=8192;l.forward.attention_key_rows=2;l.forward.attention_chunk_rows=3;}
  return l;
}
void trajectory(ResidentPlacement placement,int resume_count,at::ScalarType dtype,int profile,int cache,bool prefill,
                ResidentOptimizerKind kind,const std::string& emit,bool explicit_owners,bool emission,bool accumulation=false,bool contexts=false,bool compact=false,bool compact_journals=false) {
  at::NoGradGuard guard;const auto d=placement.devices[0];const bool half=dtype==at::kHalf;
  auto f=cache<0?test::precision_graph_profile(profile%2?0:3,profile%2,4,profile):test::retained_cache_fixture(0,1,4,cache);
  if(emission)test::emission_training_fixture(f,profile%3);
  if(explicit_owners)for(size_t n=0;n<f.graph.nodes.size();++n) {
    placement.full_owners.push_back(n%placement.devices.size());
    placement.state_owners.push_back((n+1)%placement.devices.size());
  }
  test::fixture_dtype(f,dtype);f.model=test::train_model(f.model,dtype);
  auto cpu=f;cpu.model=test::train_model(f.model,dtype);cpu.initial=test::train_boundary(f.initial,dtype);
  auto master=test::train_model(f.model,at::kFloat);auto registry=master.parameters(true);
  OptimizerGroup a;a.lr=.0001;a.momentum=.875;a.nesterov=true;a.weight_decay=.0125;a.amsgrad=true;a.eps=.001;
  auto b=a;b.lr=.00005;b.amsgrad=false;b.maximize=true;
  size_t ordinal=0;for(const auto& o:registry.owners())(ordinal++%2?a:b).parameters.push_back(o.aliases.back());
  std::vector<OptimizerGroup> groups{a,b};std::unique_ptr<NamedOptimizer> optimizer;
  if(kind==ResidentOptimizerKind::sgd)optimizer=std::make_unique<SGD>(registry,groups);else optimizer=std::make_unique<AdamW>(registry,groups);
  auto l=limits(placement,prefill,cache>=0,emit);
  if(compact_journals)l.forward.chunk_policy=ResidentChunkPolicy::aggressive;
  l.forward.diagnostics=prefill;
  auto session=std::make_unique<ResidentTrainingSession>(f.graph,f.model,f.initial,d,kind,groups,l);
  auto previous=session->checkpoint();updated(previous,master,*optimizer,half);
  std::vector<ResidentContinuation> saved_contexts;
  std::vector<Continuation> cpu_contexts;
  if(contexts) {
    auto initial=session->snapshot_device(512*1024*1024,compact);saved_contexts={initial,initial};
    if(compact)require(initial.tensor_bytes()<session->snapshot_device(512*1024*1024).tensor_bytes(),"empty compact context did not shrink");
    auto budgets=initial.device_bytes();Index total=0;for(const auto& [_,n]:budgets)total+=n;
    require(total==initial.tensor_bytes(),"snapshot per-device bytes do not sum");
    require(session->snapshot_device(total,compact,budgets).device_bytes()==budgets,"snapshot exact per-device budget changed");
    --budgets.begin()->second;
    test::train_reject([&]{session->snapshot_device(total,compact,budgets);},"snapshot exceeded per-device budget");
    cpu_contexts={test::train_boundary(cpu.initial,dtype),test::train_boundary(cpu.initial,dtype)};
    test::train_reject([&]{session->snapshot_device(1,compact);},"device snapshot capacity ignored");
  }
  require(session->placement().devices==placement.devices,"resolved devices changed");
  Options options;options.mode=emit;options.zeta=.75;
  std::map<std::pair<Index,Index>,Index> positions;for(const auto& x:f.input)++positions[{x.batch,x.port}];
  ResidentTrainingWindow kept;ResidentGradients kept_gradient;
  for(int step=0;step<(accumulation?6:4);++step) {
    if(contexts) {
      session->restore_device(saved_contexts[step%2]);cpu.initial=cpu_contexts[step%2];
      const auto cleared=session->result();
      require(cleared.trace.empty()&&cleared.messages.empty()&&cleared.outputs.empty(),"restore retained another context's diagnostics");
    }
    const int mode=accumulation?(step<2?(cache>=0?9:4):step==2?(cache>=0?8:5):0):
      step==1?(cache>=0?8:5):step==2?0:cache>=0?9:4;
    auto input=f.input;for(auto& x:input){x.time+=(contexts?step/2:step)*11;x.position+=(contexts?step/2:step)*positions.at({x.batch,x.port});
      if(contexts&&step%2)x.value=x.value*1.125;}cpu.input=input;
    const auto ref=test::retained_reference_precision(cpu,mode,at::kFloat,half,options);
    const auto wide=test::retained_reference_precision(cpu,mode,at::kDouble,half,options);
    std::vector<ResidentCotangents> roots;auto start=session->cut();size_t w=0;
    for(auto stop:test::retained_stops(start)) {
      std::vector<External> local;for(auto x:input)if(start<=x.time&&x.time<stop){if(step==1)x.value=x.value.to(d);local.push_back(x);}
      auto window=session->advance(local,stop,stop);auto root=test::train_roots(window,w,mode);
      require(window.outputs.values.scalar_type()==dtype,"payload dtype changed");
      if(!l.placement.devices.empty()) {
        require(!window.state_values.defined()&&window.cache.empty()&&window.states.size()==l.placement.devices.size(),"sharded output created a dense state replica");
        for(size_t i=0;i<window.states.size();++i)require(window.states[i].values.device()==l.placement.devices[i],"state output left owner");
      }
      const auto observed=session->result();
      require(observed.stats.at("diagnostics")==l.forward.diagnostics,"training diagnostic request changed");
      if(!l.forward.diagnostics)require(observed.trace.empty()&&observed.messages.empty(),"disabled diagnostic records were exported");
      tide_bench::compare(observed,ref.windows[w],l.forward.diagnostics,dtype,std::nullopt,half?2e-2:1e-5,half?2e-3:1e-6);
      roots.push_back(root);kept=window;start=stop;++w;
    }
    test::train_reject([&]{session->checkpoint();},"checkpoint accepted retained tapes");
    if(contexts) {
      test::train_reject([&]{session->snapshot_device(512*1024*1024,compact);},"device snapshot accepted live tapes");
      test::train_reject([&]{session->restore_device(saved_contexts[0]);},"device restore dropped live tapes");
    }
    auto wrong=roots;std::swap(wrong[0],wrong[1]);test::train_reject([&]{session->backward(wrong);},"stale root order accepted");
    wrong=roots;wrong[0].token.session++;test::train_reject([&]{session->backward(wrong);},"foreign token accepted");
    auto grad=session->backward(roots);test::train_gradients(grad,ref,cpu);test::train_gradients(grad,wide,cpu);kept_gradient=grad;
    const auto attention=grad.statistics.at("retained_attention_bytes");
    require((attention>0)==(cache>=0),"attention snapshot scope differs from model");
    const auto dense=grad.statistics.at("retained_windows")*grad.statistics.at("retained_window_bytes")+
      grad.statistics.at("retained_projection_bytes")+attention+grad.statistics.at("retained_full_bytes");
    require(grad.statistics.at("retained_dense_bytes")==dense,"shared parameter snapshot counted per window");
    const bool streamed=compact_journals&&!l.placement.devices.empty();
    require(grad.statistics.at("streamed_parameter_windows")==int64_t(streamed?roots.size():0),"window reduction placement differs from policy");
    require((grad.statistics.at("reused_projection_gradient_bytes")>0)==(streamed&&emission&&roots.size()>1),"projection gradient reuse differs from policy");
    require((grad.statistics.at("reused_attention_gradient_bytes")>0)==(streamed&&cache>=0&&roots.size()>1),"attention adjoint reuse differs from policy");
    require(grad.statistics.at("retained_compact_journals")==compact_journals,"retained journal policy changed");
    if(compact_journals)require(grad.statistics.at("retained_bytes")<grad.statistics.at("retained_dense_bytes"),"retained journals did not shrink");
    test::train_reject([&]{session->backward(roots);},"consumed roots reused");
    for(const auto& o:registry.owners()) {
      auto g=ref.gradients.at(o.canonical);g=g.defined()?g.to(at::kFloat):Tensor{};
      if(accumulation&&step%2) {
        if(g.defined())o.value.mutable_grad()=o.value.grad().defined()?o.value.grad()+g:g;
      } else o.value.mutable_grad()=g;
    }
    if(accumulation) {
      test::train_reject([&]{session->accumulate(1);},"accumulation capacity ignored");
      if(step%2)test::train_reject([&]{session->step();},"step dropped the final unaccumulated backward");
      session->accumulate();
      require(session->accumulated_batches()==step%2+1&&session->generation()==step/2,"accumulation changed generation/count");
      test::train_reject([&]{session->accumulate();},"backward was accumulated twice");
      test::train_reject([&]{session->checkpoint();},"checkpoint dropped accumulated gradients");
      if(contexts) {
        saved_contexts[step%2]=session->snapshot_device(512*1024*1024,compact);
        cpu_contexts[step%2]=test::train_boundary(ref.windows.back().continuation,dtype);
      }
      if(step%2==0){cpu.initial=test::train_boundary(ref.windows.back().continuation,dtype);continue;}
    }
    optimizer->step();require(session->step().applied,"public sharded step refused");
    require(session->accumulated_batches()==0,"step retained accumulated gradients");
    auto c=session->checkpoint();updated(c,master,*optimizer,half);if(step==(accumulation?5:2))exact(previous.state,c.state);previous=c;
    for(const auto& o:cpu.model.parameters(false).owners())o.value.copy_(registry.value(o.canonical).to(dtype));
    cpu.initial=test::train_boundary(ref.windows.back().continuation,dtype);
    if(step==1&&!contexts) {
      session->close();l.forward.prefill=!prefill;l.forward.diagnostics=!l.forward.diagnostics;l.placement.devices.clear();
      l.placement.full_owners.clear();l.placement.state_owners.clear();
      for(int i=0;i<resume_count;++i)l.placement.devices.emplace_back(d.type(),d.index()+i);
      l.placement.policy=placement.policy=="memory"?"locality":"memory";
      session=std::make_unique<ResidentTrainingSession>(f.graph,f.model,c,d,l);exact(c.state,session->checkpoint().state);
      auto bad=c;bad.state.values=c.state.values.clone();bad.state.values[0].add_(1.f);
      test::train_reject([&]{ResidentTrainingSession invalid(f.graph,f.model,bad,d,l);},"inconsistent master checkpoint accepted");
    }
  }
  session->close();require(kept.outputs.values.defined(),"window lost lifetime after close");
  for(const auto& p:kept_gradient.parameter_shards)require(at::isfinite(p.values.cpu()).all().item<bool>(),"gradient lost lifetime after close");
}
void refusal(ResidentPlacement placement,at::ScalarType dtype,bool accumulation=false) {
  at::NoGradGuard guard;auto f=test::retained_fixture(0,0,3);test::fixture_dtype(f,dtype);f.model=test::train_model(f.model,dtype);
  auto l=limits(placement,true,false,"hard");auto d=placement.devices[0];
  auto bad=l;bad.retained_bytes=1;test::train_reject([&]{ResidentTrainingSession invalid(f.graph,f.model,f.initial,d,ResidentOptimizerKind::sgd,{},bad);},"retention budget ignored");
  ResidentTrainingSession s(f.graph,f.model,f.initial,d,ResidentOptimizerKind::adamw,{},l);const auto before=s.checkpoint();
  std::vector<External> input;auto stop=f.initial.cut+3;for(const auto& x:f.input)if(x.time<stop)input.push_back(x);
  const auto w=s.advance(input,stop,stop);auto root=test::train_roots(w,0,4);
  root.outputs.fill_(std::numeric_limits<float>::quiet_NaN());for(auto& state:root.states)state.final.fill_(std::numeric_limits<float>::quiet_NaN());
  s.backward({root});if(accumulation)s.accumulate();
  auto result=s.step();require(!result.applied&&result.refusal_code==20&&s.generation()==0,"nonfinite update partially committed");
  s.detach();exact(before.state,s.checkpoint().state);s.close();
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    int count=2,resume=2;bool smoke=false,explicit_owners=false,emission=false,accumulation=false,contexts=false,compact=false,compact_journals=false;std::string policy="locality";std::vector<char*> forwarded{argv[0]};
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a.rfind("--devices=",0)==0)count=std::stoi(a.substr(10));
      else if(a.rfind("--resume-devices=",0)==0)resume=std::stoi(a.substr(17));else if(a.rfind("--placement=",0)==0)policy=a.substr(12);
      else if(a=="--accumulate")accumulation=true;
      else if(a=="--compact-journals")compact_journals=true;
      else if(a=="--compact-contexts"){compact=true;contexts=true;accumulation=true;}
      else if(a=="--contexts"){contexts=true;accumulation=true;}
      else if(a=="--profile-smoke")smoke=true;else if(a=="--explicit-owners")explicit_owners=true;else if(a=="--emission")emission=true;else forwarded.push_back(argv[i]);}
    auto args=portable_torch::parse_cli(forwarded.size(),forwarded.data(),true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf)||count<1||count>4||resume<0||resume>4)throw std::invalid_argument("explicit NPU FP32/FP16 and 1..4 owners required; resume=0 tests legacy single owner");
    args.allow_npu_float16=true;auto d=portable_torch::resolve_device(args);if(d.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("NPU required");
    at::set_num_threads(1);at::set_num_interop_threads(1);ResidentPlacement placement;placement.policy=policy;
    test::retained_attention_check(d,args.dtype);
    test::retained_full_check(d,args.dtype);
    test::reverse_gather_check(d,args.dtype);
    test::private_accumulation_check(d);
    for(int i=0;i<count;++i)placement.devices.emplace_back(d.type(),d.index()+i);
    int cases=0;auto run=[&](int profile,int cache,bool prefill,ResidentOptimizerKind kind,const std::string& emit) {
      try{trajectory(placement,resume,args.dtype,profile,cache,prefill,kind,emit,explicit_owners,emission,accumulation,contexts,compact,compact_journals);++cases;
        std::cout<<"public-sharded trajectory="<<cases<<" profile="<<profile<<" cache="<<cache<<" prefill="<<prefill<<" emit="<<emit<<std::endl;}
      catch(...){std::cerr<<"public-sharded failed profile="<<profile<<" cache="<<cache<<" prefill="<<prefill<<" emit="<<emit<<'\n';throw;}
    };
    if(accumulation) {
      refusal(placement,args.dtype,true);
      if(smoke)run(0,6,true,ResidentOptimizerKind::adamw,"hard");
      else {
        for(bool prefill:{false,true})for(auto kind:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw})
          run(0,-1,prefill,kind,"hard");
        for(int cache:{0,6})for(auto emit:{"hst","softp"})run(0,cache,true,ResidentOptimizerKind::adamw,emit);
      }
    } else if(emission) {
      if(smoke)run(0,6,true,ResidentOptimizerKind::adamw,"hard");
      else {
        for(int profile:{0,4,11})for(bool prefill:{false,true})for(auto kind:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw})run(profile,-1,prefill,kind,"hard");
        for(int cache:{0,6})for(bool prefill:{false,true})run(0,cache,prefill,ResidentOptimizerKind::adamw,"hard");
      }
    } else if(smoke)run(0,6,true,ResidentOptimizerKind::adamw,"softp");
    else {
      refusal(placement,args.dtype);
      for(int profile:{0,4,11,16})for(bool prefill:{false,true})for(auto kind:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw})run(profile,-1,prefill,kind,"hard");
      for(int cache:{0,5,6})for(bool prefill:{false,true})for(auto kind:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw})run(0,cache,prefill,kind,"hard");
      for(auto emit:{"hst","softp"}){run(16,-1,true,ResidentOptimizerKind::adamw,emit);run(0,6,true,ResidentOptimizerKind::adamw,emit);}
    }
    std::cout<<"resident-sharded-session: passed trajectories="<<cases<<" windows="<<cases*(accumulation?24:16)<<" updates="<<cases*(accumulation?3:4)
      <<" devices="<<count<<" resume_devices="<<resume<<" CPU=FP32_FP64 payload="<<args.dtype<<" emission="<<emission
      <<" compact_journals="<<compact_journals<<" accumulation="<<accumulation<<" compact="<<compact<<" contexts="<<contexts<<" public_api=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
