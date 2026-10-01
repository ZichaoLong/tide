#include "state_reverse_test.h"
#include "sharded_parameter_compare.h"
#include "sharded_optimizer.h"
#include "precision_graph_profiles.h"
#include "precision_graph_fixture.h"
#include "retained_cache_fixture.h"
#include "training_test.h"
#include "full_vjp_fixture.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>
namespace {
using namespace tide;using namespace tide::device_online;
void require(bool x,const char* why){if(!x)throw std::runtime_error(why);}
std::vector<Tensor> snapshots(const std::vector<std::unique_ptr<DeviceOptimizer>>& owners) {
  std::vector<Tensor> out;for(const auto& p:owners)for(auto x:{p->values(),p->first(),p->second(),p->maximum(),p->steps(),p->corrections()})out.push_back(x.cpu());return out;
}
void exact(const std::vector<Tensor>& a,const std::vector<Tensor>& b) {
  require(a.size()==b.size(),"sharded snapshot field count changed");
  for(size_t i=0;i<a.size();++i)require(at::equal(a[i].contiguous().reshape({-1}).view(at::kByte),
    b[i].contiguous().reshape({-1}).view(at::kByte)),"refused/disconnected update changed live fields");
}
void updated(const std::vector<ParameterVjp>& partitions,const std::vector<std::unique_ptr<DeviceOptimizer>>& owners,
    const ParameterRegistry& reference,const NamedOptimizer& optimizer,bool half) {
  const auto yes=at::ones({},at::kBool);
  for(size_t d=0;d<partitions.size();++d) {
    const auto& layout=partitions[d];const auto state=owners[d]->snapshot();
    for(size_t i=0;i<layout.owners.size();++i) {
      const auto& o=layout.owners[i];const auto offset=layout.offsets[i],size=o.value.numel();
      auto same=[&](const Tensor& x,const Tensor& y){if(y.defined())test::full_same_precision(x.narrow(0,offset,size).reshape(y.sizes()),yes,y,o.canonical.c_str(),half);};
      if(offset>=0)same(state.values,reference.value(o.canonical));
      const auto found=optimizer.state().find(o.canonical);
      if(found==optimizer.state().end()){require(state.steps[i].item<int64_t>()==0,"None owner acquired optimizer state");continue;}
      const auto& s=found->second;same(state.first,s.momentum_buffer.defined()?s.momentum_buffer:s.exp_avg);
      same(state.second,s.exp_avg_sq);same(state.maximum,s.max_exp_avg_sq);
      if(s.exp_avg.defined())require(state.steps[i].item<int64_t>()==s.step,"canonical AdamW counter differs");
    }
  }
}
void publication(const std::map<std::string,ParameterDestination>& destinations,const std::vector<ParameterVjp>& partitions,
    const std::vector<std::unique_ptr<DeviceOptimizer>>& owners) {
  for(size_t d=0;d<partitions.size();++d) {
    auto masters=owners[d]->values().cpu();const auto& layout=partitions[d];
    for(size_t i=0;i<layout.owners.size();++i) {
      const auto& o=layout.owners[i];const auto value=layout.offsets[i]<0?o.value:
        masters.narrow(0,layout.offsets[i],o.value.numel()).reshape(o.value.sizes());
      for(const auto& name:o.aliases)if(auto it=destinations.find(name);it!=destinations.end()) {
        auto actual=it->second.values.cpu();auto expected=value.to(o.value.scalar_type()).to(actual.scalar_type());
        require(at::equal(actual,expected),"published alias differs from rounded canonical master");
      }
    }
  }
}
void trajectory(std::vector<at::Device> devices,at::ScalarType dtype,int profile,int cache,bool prefill,
                DeviceOptimizerKind kind,const std::string& emit,const std::string& policy,Index width=4,bool state_shards=false) {
  at::NoGradGuard guard;const bool half=dtype==at::kHalf;
  auto f=cache<0?test::precision_graph_profile(profile%2?0:3,profile%2,width,profile):test::retained_cache_fixture(0,1,width,cache);
  test::fixture_dtype(f,dtype);auto registry=f.model.parameters(false);
  auto cpu=f;cpu.model=test::train_model(f.model,dtype);cpu.initial=test::train_boundary(f.initial,dtype);
  auto master=test::train_model(f.model,at::kFloat);auto master_registry=master.parameters(false);
  OptimizerGroup a;a.lr=.0001;a.momentum=.875;a.nesterov=true;a.weight_decay=.0125;a.amsgrad=true;a.eps=.001;
  auto b=a;b.lr=.00005;b.amsgrad=false;b.maximize=true;
  size_t ordinal=0;for(const auto& o:registry.owners())(ordinal++%2?a:b).parameters.push_back(o.aliases.back());
  std::vector<OptimizerGroup> groups{a,b};std::unique_ptr<NamedOptimizer> reference;
  if(kind==DeviceOptimizerKind::sgd)reference=std::make_unique<SGD>(master_registry,groups);else reference=std::make_unique<AdamW>(master_registry,groups);
  ContentLimits limits;limits.prefill=prefill;limits.mode=emit;limits.trace=512;limits.full_chunk_rows=3;limits.zeta=.75;
  if(cache>=0){limits.queue=96;limits.arrivals=192;limits.outputs=192;limits.kv_rows=128;limits.kv_trace_rows=8192;limits.attention_key_rows=2;limits.attention_chunk_rows=3;}
  if(width>64)limits.workspace_bytes=512*1024*1024;
  if(state_shards)limits.workspace_bytes=Index(width>64?2:1)*1024*1024*1024;
  auto candidate=test::reverse_candidate(f,devices[0],limits,place_full(f.graph,f.model,devices,policy),state_shards);auto& flow=*candidate;
  auto banks=flow.sharded_parameter_banks();const auto destinations=sharded_parameter_destinations(banks);
  auto bank_snapshot=[&](){std::vector<Tensor> out;for(const auto& [_,b]:destinations)out.push_back(b.values.cpu());return out;};
  std::vector<std::unique_ptr<DeviceOptimizer>> optimizers;std::map<std::pair<Index,Index>,Index> positions;
  for(const auto& x:f.input)++positions[{x.batch,x.port}];
  Options options;options.mode=emit;options.zeta=limits.zeta;bool fractional=false;
  for(int step=0;step<4;++step) {
    const int mode=step==1?(cache>=0?8:5):step==2?0:cache>=0?9:4;
    auto input=f.input;for(auto& x:input){x.time+=step*11;x.position+=step*positions.at({x.batch,x.port});}cpu.input=input;
    // Independent references use only their own continuation and parameters.
    const auto expected=test::retained_reference_precision(cpu,mode,at::kFloat,half,options);
    const auto wide=test::retained_reference_precision(cpu,mode,at::kDouble,half,options);
    std::vector<RetainedShardedTape> saved;auto start=cpu.initial.cut;size_t window=0;
    for(auto stop:test::retained_stops(start)) {
      std::vector<External> local;for(auto x:input)if(start<=x.time&&x.time<stop)local.push_back(x);
      flow.advance_device(local,stop);saved.push_back(retain_sharded_reverse_tape(flow.sharded_reverse_tape(),512*1024*1024));
      tide_bench::compare(flow.result(),expected.windows[window++],true,dtype,std::nullopt,half?2e-2:1e-5,half?2e-3:1e-6);start=stop;
    }
    auto error=at::zeros({1},at::TensorOptions().device(devices[0]).dtype(at::kInt));
    CannSequence programs(devices[0],saved.size(),128*1024*1024);std::vector<ShardedGraphVjp> gradient(saved.size());
    for(size_t w=saved.size();w>0;) {--w;auto& p=programs.append();const auto& t=saved[w].tape.coordinator;
      auto roots=test::retained_roots(t,w,mode);test::retained_cache_roots(roots,t,w,mode);
      if(w+1<saved.size())roots=append_window_bridge(p,t,roots,saved[w+1].tape.coordinator,gradient[w+1].coordinator,error,32*1024*1024);
      gradient[w]=append_sharded_graph_vjp(p,saved[w].tape,roots,error,prefill?3:1,Index(width>64?4:1)*1024*1024*1024,128*1024*1024,
        w+1<saved.size()?gradient[w+1].state:nullptr,test::owner_cache_roots(saved[w].tape,w,mode));
    }
    programs.finish();run_sharded_graph_vjp(programs,gradient);require(!error.cpu().item<int>(),"training reverse refused");
    auto sources=sharded_parameter_sources(f.graph,registry,gradient,64*1024*1024);
    ShardedParameterReduce reduction(sources,devices,error,256*1024*1024,32*1024*1024);
    if(optimizers.empty())optimizers=make_sharded_optimizers(reduction.gradients(),kind,groups,256*1024*1024);
    std::vector<DeviceOptimizer*> pointers;for(const auto& x:optimizers)pointers.push_back(x.get());
    auto before=snapshots(optimizers),before_banks=bank_snapshot();
    reduction.append_step(pointers);reduction.append_publish(banks,pointers,256*1024*1024);reduction.finish();reduction.run();
    for(auto e:reduction.errors())require(!e.cpu().item<int>(),"complete sharded training step refused");
    auto actual=test::sharded_parameter_observations(reduction.gradients());
    for(const auto* ref:{&expected,&wide})for(const auto& [name,value]:actual)
      test::full_same_precision(value.defined()?value:at::zeros({},at::kFloat),at::full({},value.defined(),at::kBool),ref->gradients.at(name),name.c_str(),half);
    for(const auto& o:master_registry.owners()){auto g=expected.gradients.at(o.canonical);o.value.mutable_grad()=g.defined()?g.to(at::kFloat):Tensor{};}
    reference->step();updated(reduction.gradients(),optimizers,master_registry,*reference,half);publication(destinations,reduction.gradients(),optimizers);
    if(step==2){exact(before,snapshots(optimizers));exact(before_banks,bank_snapshot());}
    if(half)for(const auto& x:optimizers){auto v=x->values().cpu();fractional|=!at::equal(v,v.to(at::kHalf).to(at::kFloat));}
    for(const auto& o:cpu.model.parameters(false).owners())o.value.copy_(master_registry.value(o.canonical).to(dtype));
    cpu.initial=test::train_boundary(expected.windows.back().continuation,dtype);
    if(step==3) {
      const auto state=snapshots(optimizers),payload=bank_snapshot();bool poisoned=false;
      for(const auto& owner:sources.contributions)if(!owner.empty()){owner[0].values.fill_(std::numeric_limits<float>::quiet_NaN());owner[0].connected.fill_(true);poisoned=true;break;}
      require(poisoned,"training fixture has no parameter contribution");reduction.run();
      for(auto e:reduction.errors())require(e.cpu().item<int>()!=0,"nonfinite training update committed");
      exact(state,snapshots(optimizers));exact(payload,bank_snapshot());
    }
    reduction.close();programs.close();close_sharded_graph_vjp(gradient);
  }
  if(half)require(fractional,"canonical FP32 masters lost sub-half increments");flow.close();
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    int count=2;bool smoke=false,state_shards=false;std::string policy="locality";std::vector<char*> forwarded{argv[0]};
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a.rfind("--full-shards=",0)==0)count=std::stoi(a.substr(14));else if(a=="--state-shards")state_shards=true;else if(a=="--profile-smoke")smoke=true;
      else if(a.rfind("--full-placement=",0)==0)policy=a.substr(17);else forwarded.push_back(argv[i]);}
    auto args=portable_torch::parse_cli(forwarded.size(),forwarded.data(),true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf)||count<1||count>4)throw std::invalid_argument("explicit 1..4 NPU FP32/FP16 required");
    args.allow_npu_float16=true;auto d=portable_torch::resolve_device(args);if(d.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("NPU required");
    at::set_num_threads(1);at::set_num_interop_threads(1);std::vector<at::Device> devices;for(int i=0;i<count;++i)devices.emplace_back(d.type(),d.index()+i);int cases=0;
    auto run=[&](int profile,int cache,bool prefill,DeviceOptimizerKind kind,const std::string& emit,Index width=4) {
      try{trajectory(devices,args.dtype,profile,cache,prefill,kind,emit,policy,width,state_shards);++cases;std::cout<<"completed trajectory="<<cases<<" profile="<<profile<<" cache="<<cache<<" width="<<width<<std::endl;}
      catch(...){std::cerr<<"sharded training profile="<<profile<<" cache="<<cache<<" prefill="<<prefill<<" kind="<<int(kind)<<" emit="<<emit<<'\n';throw;}
    };
    if(smoke){run(16,-1,true,DeviceOptimizerKind::adamw,"hst");run(0,6,true,DeviceOptimizerKind::sgd,"softp");}
    else {
      for(int profile:{0,4,11,15,16})for(bool prefill:{false,true})for(auto kind:{DeviceOptimizerKind::sgd,DeviceOptimizerKind::adamw})run(profile,-1,prefill,kind,"hard");
      for(int profile:{4,16})for(auto emit:{"hst","softp"})run(profile,-1,true,DeviceOptimizerKind::adamw,emit);
      for(int cache:{0,5,6})for(bool prefill:{false,true})for(auto kind:{DeviceOptimizerKind::sgd,DeviceOptimizerKind::adamw})run(0,cache,prefill,kind,"hard");
      for(auto emit:{"hst","softp"})run(0,6,true,DeviceOptimizerKind::adamw,emit);
      for(Index width:{1,257})run(0,-1,true,DeviceOptimizerKind::sgd,"hard",width);
    }
    std::cout<<"sharded-training: passed trajectories="<<cases<<" windows="<<cases*16<<" updates="<<cases*4<<" state_shards="<<state_shards<<" devices="<<count
      <<" payload="<<args.dtype<<" CPU=FP32_FP64 atomic=true publication=true continuation=true scope="<<(smoke?"profile-smoke":state_shards?"internal_compact_state_KV_canonical_training":"internal_Full_shards_coordinator_state_KV")<<'\n';
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
