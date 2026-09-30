#include "device_optimizer.h"
#include "optimizer_layout.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide;using namespace tide::device_online;
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
void same(const Tensor& actual,const Tensor& expected,const std::string& field) {
  if(!at::allclose(actual,expected.to(at::kFloat),1e-5,1e-6)) {
    std::cerr<<field<<" max_abs="<<(actual-expected.to(at::kFloat)).abs().max().item<double>()<<'\n';
    throw std::runtime_error("device optimizer mismatch: "+field);
  }
}
struct Fixture {ParameterRegistry registry;ParameterVjp gradient;};
Fixture fixture(at::Device device,int64_t width) {
  Fixture f;
  auto a=at::arange(width,at::kFloat).remainder(11)*.03125-.125;
  f.registry.add("a",a);f.registry.add("alias",a);f.registry.add("b",at::full({width+2},.0625f,at::kFloat));
  f.registry.add("c",at::full({},.03125f,at::kFloat));f.registry.add("dead",at::full({},.5f,at::kFloat));
  f.gradient.owners=f.registry.owners();int64_t size=0;
  for(const auto& o:f.gradient.owners){f.gradient.offsets.push_back(o.canonical=="dead"?-1:size);if(o.canonical!="dead")size+=o.value.numel();}
  f.gradient.values=at::zeros({size},at::TensorOptions().device(device).dtype(at::kFloat));
  f.gradient.connected=at::zeros({int64_t(f.gradient.owners.size())},f.gradient.values.options().dtype(at::kBool));return f;
}
std::vector<OptimizerGroup> groups(DeviceOptimizerKind kind,int variant) {
  OptimizerGroup a;a.parameters={"alias","c","dead"};a.lr=.0125;
  if(variant)a.weight_decay=.025;
  if(kind==DeviceOptimizerKind::sgd) {
    if(variant)a.momentum=.875;if(variant==2)a.dampening=.125;
    if(variant==3)a.nesterov=true;
  }else {
    a.beta1=variant==1?0.:.875;a.beta2=variant==2?0.:.999;
    a.amsgrad=variant>=2;a.eps=variant==3?1e-5:1e-8;
  }
  a.maximize=variant==2;
  auto b=a;b.parameters={"b"};b.lr=.03125;b.weight_decay=.0625;b.maximize=!a.maximize;
  if(kind==DeviceOptimizerKind::adamw)b.amsgrad=!a.amsgrad;
  return {a,b};
}
void trajectory(at::Device device,DeviceOptimizerKind kind,int variant,int64_t width,at::ScalarType dtype) {
  at::NoGradGuard guard;auto f=fixture(device,width);auto gs=groups(kind,variant);ParameterRegistry cpu;
  for(const auto& o:f.registry.owners()){auto x=o.value.to(dtype).clone();for(const auto& name:o.aliases)cpu.add(name,x);}
  std::unique_ptr<NamedOptimizer> reference;
  if(kind==DeviceOptimizerKind::sgd)reference=std::make_unique<SGD>(cpu,gs);else reference=std::make_unique<AdamW>(cpu,gs);
  DeviceOptimizer optimizer(f.gradient,kind,gs,16*1024*1024);auto error=at::zeros({1},f.gradient.values.options().dtype(at::kInt));
  CannProgram p(device);optimizer.append_step(p,f.gradient,error);p.finish();std::vector<int64_t> counts(f.gradient.owners.size(),0);
  for(int step=0;step<8;++step) {
    auto gradient=at::full({f.gradient.values.numel()},std::numeric_limits<float>::quiet_NaN(),at::kFloat);
    auto flags=at::zeros({int64_t(counts.size())},at::kBool);
    for(size_t i=0;i<f.gradient.owners.size();++i) {
      const auto& owner=f.gradient.owners[i];const bool on=owner.canonical=="a"&&step!=0||owner.canonical=="b"&&(step%3!=0);
      auto parameter=cpu.value(owner.canonical);parameter.mutable_grad().reset();
      if(on){auto g=at::arange(parameter.numel(),at::kFloat).remainder(7)*.015625f-.03125f+.0078125f*step;
        if(step==4)g.zero_();if(step==6)g.mul_(1e-9f);
        parameter.mutable_grad()=g.to(dtype).reshape(parameter.sizes());flags[i].fill_(true);++counts[i];
        gradient.narrow(0,f.gradient.offsets[i],g.numel()).copy_(g);}
    }
    f.gradient.values.copy_(gradient);f.gradient.connected.copy_(flags);portable_torch::synchronize(device);p.run();
    require(!error.cpu().item<int>(),"finite optimizer trajectory refused");reference->step();
    auto values=optimizer.values().cpu(),first=optimizer.first().cpu(),second=optimizer.second().cpu(),maximum=optimizer.maximum().cpu();
    auto steps=optimizer.steps().cpu();
    for(size_t i=0;i<f.gradient.owners.size();++i) {
      const auto& o=f.gradient.owners[i];const auto offset=f.gradient.offsets[i],n=o.value.numel();
      require(steps[i].item<int64_t>()==counts[i],"None/zero optimizer counter mismatch");
      if(offset>=0)same(values.narrow(0,offset,n).reshape(o.value.sizes()),cpu.value(o.canonical),o.canonical+" value");
      const auto found=reference->state().find(o.canonical);
      if(found==reference->state().end())continue;
      const auto& s=found->second;
      for(const auto& pair:std::vector<std::pair<Tensor,Tensor>>{{first,kind==DeviceOptimizerKind::sgd?s.momentum_buffer:s.exp_avg},
          {second,s.exp_avg_sq},{maximum,s.max_exp_avg_sq}})if(pair.second.defined())
        same(pair.first.narrow(0,offset,n).reshape(o.value.sizes()),pair.second,o.canonical+" slot");
      if(kind==DeviceOptimizerKind::adamw)require(s.step==counts[i],"AdamW step mismatch");
    }
  }
  // A bad connected value must not partially update another owner, slots or
  // counters. None poison was exercised in every ordinary step above.
  if(variant==3&&width==257&&dtype==at::kFloat) {
    auto snapshot=[&](){std::vector<Tensor> x;for(auto t:{optimizer.values(),optimizer.first(),optimizer.second(),optimizer.maximum(),optimizer.steps(),optimizer.corrections()})x.push_back(t.cpu());return x;};
    auto before=snapshot();
    auto unchanged=[&](){auto after=snapshot();for(size_t i=0;i<after.size();++i)
      require(at::equal(after[i].view(at::kByte),before[i].view(at::kByte)),"failed optimizer committed a partial update");};
    for(float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
      f.gradient.values.zero_();f.gradient.connected.zero_();f.gradient.connected[0].fill_(true);f.gradient.connected[1].fill_(true);
      f.gradient.values[f.gradient.offsets[1]+f.gradient.owners[1].value.numel()-1].fill_(bad);error.zero_();
      portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==tide_device::optimizer_finite_error,"nonfinite gradient not rejected");unchanged();
    }
    f.gradient.values.zero_();error.fill_(7);portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==7,"prior error overwritten");unchanged();
    auto first=optimizer.first().clone();optimizer.first()[0].fill_(std::numeric_limits<float>::infinity());error.zero_();before=snapshot();
    portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==tide_device::optimizer_finite_error,"nonfinite optimizer slot accepted");unchanged();optimizer.first().copy_(first);
    if(kind==DeviceOptimizerKind::adamw) {
      auto corrections=optimizer.corrections().clone();optimizer.corrections()[0][0].fill_(std::numeric_limits<float>::infinity());error.zero_();before=snapshot();
      portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==tide_device::optimizer_finite_error,"nonfinite bias correction accepted");unchanged();optimizer.corrections().copy_(corrections);
    }
    error.zero_();optimizer.steps()[0].fill_(std::numeric_limits<int64_t>::max());before=snapshot();portable_torch::synchronize(device);p.run();
    require(error.cpu().item<int>()==tide_device::optimizer_step_error,"int64 optimizer step overflow accepted");unchanged();
  }
}
void refusals(at::Device device) {
  at::NoGradGuard guard;auto f=fixture(device,3);
  auto reject=[&](auto fn){bool failed=false;try{fn();}catch(const std::invalid_argument&){failed=true;}require(failed,"invalid optimizer preflight accepted");};
  reject([&]{DeviceOptimizer small(f.gradient,DeviceOptimizerKind::sgd,{},1);});
  OptimizerGroup duplicate;duplicate.parameters={"a","alias"};
  reject([&]{DeviceOptimizer bad(f.gradient,DeviceOptimizerKind::adamw,{duplicate},1024*1024);});
  OptimizerGroup bad_lr;bad_lr.lr=-1;reject([&]{DeviceOptimizer bad(f.gradient,DeviceOptimizerKind::sgd,{bad_lr},1024*1024);});
  auto malformed=f.gradient;malformed.values=f.gradient.values.to(at::kHalf);
  reject([&]{DeviceOptimizer bad(malformed,DeviceOptimizerKind::sgd,{},1024*1024);});
  malformed=f.gradient;malformed.owners[2].value=malformed.owners[3].value.detach();
  reject([&]{DeviceOptimizer bad(malformed,DeviceOptimizerKind::sgd,{},1024*1024);});
  DeviceOptimizer optimizer(f.gradient,DeviceOptimizerKind::sgd,{},1024*1024);auto error=at::zeros({1},f.gradient.values.options().dtype(at::kInt));
  malformed=f.gradient;malformed.owners[0].value=malformed.owners[0].value.detach();
  reject([&]{CannProgram p(device);optimizer.append_step(p,malformed,error);});
  ParameterVjp empty;empty.values=at::zeros({1},f.gradient.values.options());empty.connected=at::zeros({1},f.gradient.connected.options());
  DeviceOptimizer none(empty,DeviceOptimizerKind::adamw,{},4096);CannProgram p(device);none.append_step(p,empty,error);p.finish();
  portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>()&&!none.steps().cpu().any().item<bool>(),"empty optimizer fabricated an update");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("device optimizer gate requires explicit NPU FP32");
    const auto device=portable_torch::resolve_device(args);if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("optimizer gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(auto kind:{DeviceOptimizerKind::sgd,DeviceOptimizerKind::adamw})for(int variant=0;variant<4;++variant)
      for(int64_t width:{1,257})for(auto dtype:{at::kFloat,at::kDouble}) {
        try{trajectory(device,kind,variant,width,dtype);++cases;}
        catch(...){std::cerr<<"optimizer kind="<<int(kind)<<" variant="<<variant<<" width="<<width<<" reference="<<dtype<<'\n';throw;}
      }
    refusals(device);std::cout<<"device-optimizer: passed trajectories="<<cases<<" updates="<<cases*8<<" CPU=FP32_FP64 finite_transaction=true scope=owner_updates_not_public_training\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
