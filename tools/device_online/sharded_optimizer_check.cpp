#include "sharded_parameter_compare.h"
#include "sharded_optimizer.h"
#include "optimizer_layout.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>
namespace {
using namespace tide;using namespace tide::device_online;
void require(bool x,const char* why){if(!x)throw std::runtime_error(why);}
void reject(const std::function<void()>& f){bool yes=false;try{f();}catch(const std::invalid_argument&){yes=true;}require(yes,"invalid sharded optimizer metadata accepted");}
std::vector<Tensor> snapshot(const std::vector<std::unique_ptr<DeviceOptimizer>>& optimizers) {
  std::vector<Tensor> out;for(const auto& p:optimizers)
    for(auto x:{p->values(),p->first(),p->second(),p->maximum(),p->steps(),p->corrections()})out.push_back(x.cpu());
  return out;
}
void unchanged(const std::vector<Tensor>& before,const std::vector<std::unique_ptr<DeviceOptimizer>>& optimizers) {
  auto after=snapshot(optimizers);require(before.size()==after.size(),"optimizer snapshot extent changed");
  for(size_t i=0;i<before.size();++i)require(at::equal(before[i].view(at::kByte),after[i].view(at::kByte)),"one card committed before global refusal");
}
void trajectory(std::vector<at::Device> devices,DeviceOptimizerKind kind,at::ScalarType dtype,int64_t width) {
  at::NoGradGuard guard;ParameterRegistry registry,cpu;
  for(int i=0;i<3;++i){auto x=at::full({width+i},.03125f*(i+1),dtype);auto y=x.to(at::kFloat).clone();
    const auto name=std::string(1,char('a'+i));registry.add(name,x);cpu.add(name,y);
    registry.add(name+"_alias",x);cpu.add(name+"_alias",y);}
  registry.add("unused",at::ones({},dtype));cpu.add("unused",at::ones({},at::kFloat));
  ShardedParameterSources source;source.owners=registry.owners();source.contributions.resize(source.owners.size());
  for(size_t i=0;i<3;++i)for(auto d:devices){auto opt=at::TensorOptions().device(d).dtype(at::kFloat);
    source.contributions[i].push_back({at::zeros_like(source.owners[i].value,opt),at::zeros({1},opt.dtype(at::kBool))});}
  auto error=at::zeros({1},at::TensorOptions().device(devices[0]).dtype(at::kInt));
  ShardedParameterReduce reduction(source,devices,error,16*1024*1024,8*1024*1024);
  OptimizerGroup a;a.parameters={"a_alias","c","unused"};a.lr=.003125;a.weight_decay=.0125;a.momentum=.875;a.nesterov=true;a.amsgrad=true;a.eps=1e-5;
  auto b=a;b.parameters={"b"};b.lr=.0015625;b.maximize=true;b.amsgrad=false;
  std::vector<OptimizerGroup> groups{a,b};
  auto optimizers=make_sharded_optimizers(reduction.gradients(),kind,groups,16*1024*1024);
  std::vector<DeviceOptimizer*> pointers;for(const auto& p:optimizers)pointers.push_back(p.get());
  reduction.append_step(pointers);reduction.finish();std::unique_ptr<NamedOptimizer> reference;
  if(kind==DeviceOptimizerKind::sgd)reference=std::make_unique<SGD>(cpu,groups);else reference=std::make_unique<AdamW>(cpu,groups);
  std::map<std::string,int64_t> steps;
  auto fill=[&](int step) {
    for(size_t i=0;i<3;++i) {
      Tensor expected;
      for(size_t d=0;d<devices.size();++d) {
        const bool live=step!=0&&(i+d+step)%3!=0;auto& part=source.contributions[i][d];
        auto value=at::arange(part.values.numel(),at::kFloat).remainder(7)*.015625f+.03125f*(i+1)-.0078125f*d;
        if(step==4)value.zero_();
        part.values.copy_(live?value:at::full_like(value,std::numeric_limits<float>::quiet_NaN()));part.connected.fill_(live);
        if(live)expected=expected.defined()?expected+value:value;
      }
      const auto name=source.owners[i].canonical;cpu.value(name).mutable_grad()=expected;
      if(expected.defined())++steps[name];
    }
  };
  for(int step=0;step<8;++step) {
    fill(step);reduction.run();for(auto e:reduction.errors())require(!e.cpu().item<int>(),"finite multi-card update refused");
    auto actual=test::sharded_parameter_observations(reduction.gradients());
    for(const auto& o:cpu.owners()) {
      auto expected=o.value.grad();const auto value=actual.at(o.canonical);
      require(value.defined()==expected.defined(),"None/connected zero changed during owner reduction");
      if(value.defined())require(at::allclose(value,expected,1e-5,1e-6),"canonical owner sum differs");
    }
    reference->step();
    for(size_t d=0;d<devices.size();++d) {
      const auto& layout=reduction.gradients()[d];const auto& opt=*optimizers[d];
      auto values=opt.values().cpu(),first=opt.first().cpu(),second=opt.second().cpu(),maximum=opt.maximum().cpu(),counts=opt.steps().cpu();
      for(size_t i=0;i<layout.owners.size();++i) {
        const auto& o=layout.owners[i];const auto offset=layout.offsets[i],size=o.value.numel();
        require(counts[i].item<int64_t>()==steps[o.canonical],"None/zero canonical update counter changed");
        auto same=[&](const Tensor& x,const Tensor& expected){if(expected.defined())require(at::allclose(x.narrow(0,offset,size).reshape(expected.sizes()),expected,1e-5,1e-6),"canonical master/optimizer slot differs");};
        if(offset>=0)same(values,cpu.value(o.canonical));
        auto found=reference->state().find(o.canonical);if(found==reference->state().end())continue;
        const auto& s=found->second;same(first,kind==DeviceOptimizerKind::sgd?s.momentum_buffer:s.exp_avg);same(second,s.exp_avg_sq);same(maximum,s.max_exp_avg_sq);
      }
    }
  }
  // Every physical source injects a connected NaN in turn; every canonical
  // owner and all slots must retain their prior bytes on every card.
  const auto before=snapshot(optimizers);
  for(size_t d=0;d<devices.size();++d)for(float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
    fill(1);source.contributions[0][d].connected.fill_(true);source.contributions[0][d].values.fill_(bad);error.zero_();
    reduction.run();for(auto e:reduction.errors())require(e.cpu().item<int>()==tide_device::optimizer_finite_error,"global finite decision disagrees");unchanged(before,optimizers);
  }
  fill(1);error.fill_(7);reduction.run();for(auto e:reduction.errors())require(e.cpu().item<int>()==7,"upstream error lost");unchanged(before,optimizers);
  fill(0);error.zero_();reduction.run();for(auto e:reduction.errors())require(!e.cpu().item<int>(),"replay kept stale error");unchanged(before,optimizers);
  reduction.close();
}
void boundaries(std::vector<at::Device> devices,at::ScalarType dtype) {
  at::NoGradGuard guard;auto opt=at::TensorOptions().device(devices[0]).dtype(at::kFloat);
  auto error=at::zeros({1},opt.dtype(at::kInt));ShardedParameterSources empty;
  reject([&]{ShardedParameterReduce bad(empty,devices,error,1,4096);});
  auto duplicate=devices;duplicate.push_back(devices[0]);reject([&]{ShardedParameterReduce bad(empty,duplicate,error,4096,4096);});
  ShardedParameterReduce reduction(empty,devices,error,65536,1024*1024);
  auto optimizers=make_sharded_optimizers(reduction.gradients(),DeviceOptimizerKind::adamw,{},65536);
  std::vector<DeviceOptimizer*> pointers;for(const auto& p:optimizers)pointers.push_back(p.get());
  reduction.append_step(pointers);reduction.finish();reduction.run();
  for(auto e:reduction.errors())require(!e.cpu().item<int>(),"empty canonical partition refused");
  for(const auto& p:optimizers)require(!p->steps().cpu().any().item<bool>(),"empty partition fabricated update");reduction.close();
  auto x=at::ones({3},dtype);ParameterVjp a,b;a.owners={{"a",{"a","alias"},x}};b.owners={{"b",{"b"},x.detach()}};
  reject([&]{validate_sharded_optimizer_owners({a,b});});
  a.values=at::zeros({3},opt);a.connected=at::zeros({1},opt.dtype(at::kBool));a.offsets={0};
  OptimizerGroup group;group.parameters={"a","alias"};reject([&]{make_sharded_optimizers({a},DeviceOptimizerKind::sgd,{group},65536);});
  group.parameters={"missing"};reject([&]{make_sharded_optimizers({a},DeviceOptimizerKind::sgd,{group},65536);});
  ShardedParameterSources valid;valid.owners=a.owners;
  valid.contributions={{{at::zeros({3},opt),at::ones({1},opt.dtype(at::kBool))}}};
  auto wrong=valid;wrong.contributions[0][0].values=at::zeros({3},opt.dtype(at::kHalf));
  reject([&]{ShardedParameterReduce bad(wrong,devices,error,65536,1024*1024);});
  wrong=valid;wrong.contributions[0][0].connected=at::ones({2},opt.dtype(at::kBool));
  reject([&]{ShardedParameterReduce bad(wrong,devices,error,65536,1024*1024);});
  wrong=valid;wrong.owners[0].canonical="wrong";
  reject([&]{ShardedParameterReduce bad(wrong,devices,error,65536,1024*1024);});
  auto master=at::zeros({3},opt),bank=at::zeros({3},opt);
  reject([&]{CannProgram p(devices[0]);append_owner_parameter_publish(p,{{0,{bank,at::kFloat}}},master,error,1);});
  reject([&]{CannProgram p(devices[0]);append_owner_parameter_publish(p,{{1,{bank,at::kFloat}}},master,error,65536);});
  reject([&]{CannProgram p(devices[0]);append_owner_parameter_publish(p,{{0,{bank.to(at::kHalf),at::kFloat}}},master,error,65536);});
  if(dtype==at::kHalf) {
    // Finite FP32 proposal that cannot be published to half is also global.
    x.fill_(65504.f);ShardedParameterSources source;source.owners={a.owners[0]};
    source.contributions={{{at::full({3},-128.f,opt),at::ones({1},opt.dtype(at::kBool))}}};
    ShardedParameterReduce half(source,devices,error,65536,1024*1024);group.parameters={"a"};group.lr=1.;
    auto owners=make_sharded_optimizers(half.gradients(),DeviceOptimizerKind::sgd,{group},65536);
    pointers.clear();for(const auto& p:owners)pointers.push_back(p.get());half.append_step(pointers);half.finish();const auto saved=snapshot(owners);
    half.run();for(auto e:half.errors())require(e.cpu().item<int>()==tide_device::optimizer_finite_error,"global half overflow was accepted");unchanged(saved,owners);half.close();
  }
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    int count=2;std::vector<char*> forwarded{argv[0]};
    for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg.rfind("--full-shards=",0)==0)count=std::stoi(arg.substr(14));else forwarded.push_back(argv[i]);}
    auto args=portable_torch::parse_cli(forwarded.size(),forwarded.data(),true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf)||count<1||count>4)throw std::invalid_argument("explicit 1..4 NPU FP32/FP16 required");
    args.allow_npu_float16=true;auto d=portable_torch::resolve_device(args);if(d.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("NPU required");
    at::set_num_threads(1);at::set_num_interop_threads(1);std::vector<at::Device> devices;for(int i=0;i<count;++i)devices.emplace_back(d.type(),d.index()+i);
    for(auto kind:{DeviceOptimizerKind::sgd,DeviceOptimizerKind::adamw})for(int64_t width:{1,257})trajectory(devices,kind,args.dtype,width);
    boundaries(devices,args.dtype);
    std::cout<<"sharded-optimizer: passed trajectories=4 updates=32 devices="<<count<<" payload="<<args.dtype<<" finite_transaction=true None_zero=true scope=canonical_updates_not_bank_publication\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
