#include "training_test.h"
#include "emission_training_fixture.h"
#include "precision_graph_fixture.h"
#include "precision_graph_profiles.h"
#include "retained_cache_fixture.h"
#include "full_vjp_fixture.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <iostream>
#include <limits>

namespace {
using namespace tide;using namespace tide::device_online;
void require(bool yes,const char* why){test::train_require(yes,why);}
void checkpoint(const ResidentTrainingCheckpoint& c,const Model& master,const NamedOptimizer& optimizer) {
  const auto all=master.parameters(false),registry=master.parameters(true);const auto owners=registry.owners();
  require(c.aliases==all.alias_partitions()&&c.trainable==registry.names(),"half checkpoint aliases/trainable owners differ");
  auto yes=at::ones({},at::kBool);
  for(const auto& o:all.owners()) {
    const auto& actual=c.parameters.at(o.canonical);
    require(actual.scalar_type()==at::kHalf,"checkpoint failed to retain half named payload");
    test::full_same_precision(actual.to(at::kFloat),yes,o.value.to(at::kHalf).to(at::kFloat),o.canonical.c_str(),true);
  }
  for(const auto& x:{c.state.values,c.state.first,c.state.second,c.state.maximum,c.state.corrections})
    require(x.scalar_type()==at::kFloat,"checkpoint rounded master or optimizer slots");
  for(size_t i=0;i<owners.size();++i) {
    const auto& o=owners[i];const auto offset=c.offsets[i];
    if(offset>=0) {
      auto value=c.state.values.narrow(0,offset,o.value.numel()).reshape(o.value.sizes());
      require(at::equal(value.to(at::kHalf),c.parameters.at(o.canonical)),"named payload is not the rounded master");
      test::full_same_precision(value,yes,o.value,o.canonical.c_str(),true);
    }
    auto found=optimizer.state().find(o.canonical);
    if(found==optimizer.state().end()) {require(!c.state.steps[i].item<Index>(),"None owner acquired optimizer progress");continue;}
    const auto& slot=found->second;
    auto same=[&](const Tensor& actual,const Tensor& expected) {
      if(expected.defined())test::full_same_precision(actual.narrow(0,offset,o.value.numel()).reshape(o.value.sizes()),yes,expected,"master optimizer slot",true);
    };
    same(c.state.first,slot.momentum_buffer.defined()?slot.momentum_buffer:slot.exp_avg);
    same(c.state.second,slot.exp_avg_sq);same(c.state.maximum,slot.max_exp_avg_sq);
    if(c.optimizer==ResidentOptimizerKind::adamw)require(c.state.steps[i].item<Index>()==slot.step,"half AdamW owner step differs");
  }
}
void equal_state(const ResidentOptimizerState& a,const ResidentOptimizerState& b) {
  const std::vector<std::pair<Tensor,Tensor>> fields={{a.values,b.values},{a.first,b.first},{a.second,b.second},
    {a.maximum,b.maximum},{a.steps,b.steps},{a.corrections,b.corrections}};
  for(const auto& [x,y]:fields)require(at::equal(x,y),"checkpoint or disconnected step changed FP32 optimizer state");
}
void finite_refusal(at::Device device,ResidentOptimizerKind kind) {
  at::NoGradGuard guard;auto f=test::retained_fixture(0,0,3);test::fixture_dtype(f,at::kHalf);
  f.model=test::train_model(f.model,at::kHalf);
  ResidentTrainingSession session(f.graph,f.model,f.initial,device,kind);
  const auto before=session.checkpoint();const auto stop=f.initial.cut+3;
  std::vector<External> input;for(auto x:f.input)if(x.time<stop)input.push_back(x);
  auto window=session.advance(input,stop,stop);auto root=test::train_roots(window,0,4);
  root.final.fill_(std::numeric_limits<float>::quiet_NaN());
  session.backward({root});const auto refused=session.step();
  require(!refused.applied&&refused.refusal_code==20&&session.generation()==0,"nonfinite root update partially committed");
  // Use the public skip path; returned gradients are read-only consumer views.
  session.detach();equal_state(before.state,session.checkpoint().state);session.close();
}
void trajectory(at::Device device,test::Fixture f,bool prefill,ResidentOptimizerKind kind,const std::string& mode="hard") {
  at::NoGradGuard guard;test::fixture_dtype(f,at::kHalf);f.model=test::train_model(f.model,at::kHalf);
  auto cpu=f;cpu.model=test::train_model(f.model,at::kHalf);cpu.initial=test::train_boundary(f.initial,at::kHalf);
  auto master=test::train_model(f.model,at::kFloat);auto registry=master.parameters(true);
  const bool cache=std::any_of(f.graph.nodes.begin(),f.graph.nodes.end(),[](const auto& n){return n.memory=="attention"||n.memory.find("lh-fiber-attention-")==0;});
  Options options;options.mode=mode;options.zeta=.75;
  OptimizerGroup group;group.lr=.0001;group.momentum=.875;group.nesterov=true;group.weight_decay=.0125;group.amsgrad=true;group.eps=.001;
  for(const auto& o:registry.owners())group.parameters.push_back(o.canonical);
  std::unique_ptr<NamedOptimizer> optimizer;
  if(kind==ResidentOptimizerKind::sgd)optimizer=std::make_unique<SGD>(registry,std::vector<OptimizerGroup>{group});
  else optimizer=std::make_unique<AdamW>(registry,std::vector<OptimizerGroup>{group});
  ResidentTrainingLimits l;l.forward.prefill=prefill;l.forward.trace=512;l.forward.full_chunk_rows=3;l.reverse_chunk_rows=3;
  l.forward.mode=mode;l.forward.zeta=options.zeta;l.forward.workspace_bytes=256*1024*1024;
  l.backward_bytes=Index(2)*1024*1024*1024;l.retained_bytes=256*1024*1024;
  if(cache){l.forward.queue=96;l.forward.arrivals=192;l.forward.outputs=192;l.forward.kv_rows=128;
    l.forward.kv_trace_rows=8192;l.forward.attention_key_rows=2;l.forward.attention_chunk_rows=3;}
  auto session=std::make_unique<ResidentTrainingSession>(f.graph,f.model,f.initial,device,kind,std::vector<OptimizerGroup>{group},l);
  auto before=session->checkpoint();checkpoint(before,master,*optimizer);
  std::map<std::pair<Index,Index>,Index> positions;for(const auto& x:f.input)++positions[{x.batch,x.port}];
  bool fractional_master=false;
  for(int step=0;step<4;++step) {
    const int roots_mode=step==1?(cache?8:5):step==2?0:cache?9:4;
    auto input=f.input;for(auto& x:input){x.time+=step*11;x.position+=step*positions.at({x.batch,x.port});}
    cpu.input=input;
    const auto ref=test::retained_reference_precision(cpu,roots_mode,at::kFloat,true,options);
    const auto wide=test::retained_reference_precision(cpu,roots_mode,at::kDouble,true,options);
    std::vector<ResidentCotangents> roots;Index start=session->cut();int index=0;
    for(auto stop:test::retained_stops(start)) {
      std::vector<External> local;for(auto x:input)if(start<=x.time&&x.time<stop){if(step==1)x.value=x.value.to(device);local.push_back(x);}
      auto window=session->advance(local,stop,stop);auto r=test::train_roots(window,index,roots_mode);
      require(window.outputs.values.scalar_type()==at::kHalf&&window.state_values.scalar_type()==at::kHalf,"training forward promoted half payload");
      for(size_t j=0;j<r.cache.size();++j) {
        const auto& a=window.cache[j];auto padding=at::arange(a.key.size(1),a.lengths.options()).unsqueeze(0)>=a.lengths.unsqueeze(1);
        for(auto* x:{&r.cache[j].key,&r.cache[j].value})if(x->defined())x->masked_fill_(padding.unsqueeze(-1).unsqueeze(-1),std::numeric_limits<float>::quiet_NaN());
        if(r.cache[j].log_bias.defined())r.cache[j].log_bias.masked_fill_(padding,std::numeric_limits<float>::quiet_NaN());
      }
      tide_bench::compare(session->result(),ref.windows[index],true,at::kHalf,std::nullopt,2e-2,2e-3);
      roots.push_back(r);start=stop;++index;
    }
    test::train_reject([&]{session->checkpoint();},"half checkpoint accepted live windows");
    auto bad=roots;std::swap(bad[0],bad[1]);test::train_reject([&]{session->backward(bad);},"half roots accepted stale order");
    if(step==0) {
      bad=roots;bad[0].outputs=roots[0].outputs.to(at::kHalf);
      test::train_reject([&]{session->backward(bad);},"half cotangent was accepted");
    }
    auto gradient=session->backward(roots);require(gradient.values.scalar_type()==at::kFloat,"parameter gradient lost FP32 precision");
    test::train_gradients(gradient,ref,cpu);test::train_gradients(gradient,wide,cpu);
    for(const auto& o:registry.owners()) {
      auto g=ref.gradients.at(o.canonical);o.value.mutable_grad()=g.defined()?g.to(at::kFloat):Tensor{};
    }
    optimizer->step();const auto updated=session->step();require(updated.applied&&updated.generation==step+1,"half optimizer failed to advance");
    const auto c=session->checkpoint();checkpoint(c,master,*optimizer);
    if(step==2)equal_state(before.state,c.state);
    fractional_master|=!at::equal(c.state.values,c.state.values.to(at::kHalf).to(at::kFloat));
    const auto masters=master.parameters(false);
    for(const auto& o:cpu.model.parameters(false).owners())o.value.copy_(masters.value(o.canonical).to(at::kHalf));
    cpu.initial=test::train_boundary(ref.windows.back().continuation,at::kHalf);before=c;
    if(step==1) {
      const auto names=registry.owners();size_t i=0;while(i<names.size()&&c.offsets[i]<0)++i;
      require(i<names.size(),"half fixture has no trainable packed owner");const auto name=names[i].canonical;
      auto wrong=c;wrong.parameters[name]=wrong.parameters[name].to(at::kFloat);
      test::train_reject([&]{ResidentTrainingSession invalid(f.graph,f.model,wrong,device,l);},"checkpoint accepted wrong payload dtype");
      wrong=c;wrong.state.values=c.state.values.clone();wrong.state.values[c.offsets[i]].add_(.25f);
      test::train_reject([&]{ResidentTrainingSession invalid(f.graph,f.model,wrong,device,l);},"checkpoint accepted inconsistent master/payload");
      session->close();l.forward.prefill=!prefill;
      session=std::make_unique<ResidentTrainingSession>(f.graph,f.model,c,device,l);
      const auto restored=session->checkpoint();equal_state(c.state,restored.state);
      for(const auto& [n,x]:c.parameters)require(at::equal(x,restored.parameters.at(n)),"half named parameter changed on resume");
      auto exported=session->checkpoint();exported.state.values.fill_(123.f);
      for(auto& [_,x]:exported.parameters)x.fill_(123.f);
      equal_state(c.state,session->checkpoint().state);
    }
  }
  require(fractional_master,"training lost sub-half master increments");session->close();
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    bool cache=false,profile=false,emission=false;std::vector<char*> forwarded{argv[0]};
    for(int i=1;i<argc;++i) {
      const std::string arg=argv[i];if(arg=="--cache")cache=true;else if(arg=="--profile-smoke")profile=true;else if(arg=="--emission")emission=true;else forwarded.push_back(argv[i]);
    }
    auto args=portable_torch::parse_cli(forwarded.size(),forwarded.data(),true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kHalf)throw std::invalid_argument("half training gate requires explicit NPU FP16");
    args.allow_npu_float16=true;const auto device=portable_torch::resolve_device(args);
    at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    if(!profile&&!cache)for(auto opt:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw})finite_refusal(device,opt);
    auto run=[&](int p,bool prefill,ResidentOptimizerKind opt,const std::string& mode="hard") {
      try {
        auto f=cache?test::retained_cache_fixture(p%2?0:3,p%2,4,p):test::precision_graph_profile(p%2?0:3,p%2,3,p);
        if(emission)test::emission_training_fixture(f,p%3);
        trajectory(device,std::move(f),prefill,opt,mode);++cases;
      }catch(...){std::cerr<<"half training cache="<<cache<<" profile="<<p<<" prefill="<<prefill<<" optimizer="<<int(opt)<<" mode="<<mode<<'\n';throw;}
    };
    if(emission) {
      if(profile)run(cache?6:0,true,ResidentOptimizerKind::adamw);
      else for(int p:{0,4,11})for(bool prefill:{false,true})for(auto opt:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw})run(p,prefill,opt);
    } else if(profile)run(cache?6:16,true,ResidentOptimizerKind::adamw,"softp");
    else if(cache) {
      for(int p=0;p<=6;++p)for(bool prefill:{false,true})for(auto opt:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw})run(p,prefill,opt);
      for(int p:{0,5,6})for(const std::string mode:{"hst","softp"})for(bool prefill:{false,true})run(p,prefill,ResidentOptimizerKind::adamw,mode);
    } else {
      for(int p=0;p<=16;++p)if(p!=5) {
        if(p==0||p==4||p==11||p==15||p==16)for(bool prefill:{false,true})for(auto opt:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw})run(p,prefill,opt);
        else run(p,true,ResidentOptimizerKind::sgd);
      }
      for(int p:{4,11,16})for(const std::string mode:{"hst","softp"})for(bool prefill:{false,true})run(p,prefill,ResidentOptimizerKind::adamw,mode);
    }
    std::cout<<"resident-half-"<<(cache?"cache-":"")<<"training: passed trajectories="<<cases<<" windows="<<cases*16
      <<" updates="<<cases*4<<" CPU=FP32_FP64 master=FP32 payload=FP16 roots_scale=256 resume=true scope="<<(profile?"profile-smoke":"public_training")<<'\n';
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
