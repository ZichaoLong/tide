#include "device_optimizer.h"
#include "optimizer_layout.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
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
Fixture fixture(at::Device device,int64_t width,at::ScalarType payload=at::kFloat) {
  Fixture f;
  auto a=at::arange(width,at::kFloat).remainder(11)*.03125-.125;
  a=a.to(payload);
  f.registry.add("a",a);f.registry.add("alias",a);f.registry.add("b",at::full({width+2},.0625f,payload));
  f.registry.add("c",at::full({},.03125f,payload));f.registry.add("dead",at::full({},.5f,payload));
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
int trajectory(at::Device device,DeviceOptimizerKind kind,int variant,int64_t width,at::ScalarType dtype,at::ScalarType payload) {
  at::NoGradGuard guard;auto f=fixture(device,width,payload);auto gs=groups(kind,variant);ParameterRegistry cpu;
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
    std::vector<Tensor> before;
    if(payload==at::kHalf)for(auto x:{optimizer.values(),optimizer.first(),optimizer.second(),optimizer.maximum(),optimizer.steps(),optimizer.corrections()})before.push_back(x.cpu());
    f.gradient.values.copy_(gradient);f.gradient.connected.copy_(flags);portable_torch::synchronize(device);p.run();
    reference->step();
    bool representable=true;
    if(payload==at::kHalf)for(size_t i=0;i<f.gradient.owners.size();++i)if(flags[i].item<bool>())
      representable&=at::isfinite(cpu.value(f.gradient.owners[i].canonical).to(at::kHalf)).all().item<bool>();
    if(!representable) {
      // Tiny AdamW epsilon/beta2=0 can make finite FP32 masters overflow half.
      // The independent CPU proposal establishes the expected refusal; all
      // device fields must stay at the preceding complete update.
      require(error.cpu().item<int>()==tide_device::optimizer_finite_error,"CPU half overflow was not refused on device");
      size_t i=0;for(auto x:{optimizer.values(),optimizer.first(),optimizer.second(),optimizer.maximum(),optimizer.steps(),optimizer.corrections()})
        require(at::equal(x.cpu().view(at::kByte),before[i++].view(at::kByte)),"half overflow partially committed optimizer state");
      return step;
    }
    require(!error.cpu().item<int>(),"finite representable optimizer trajectory refused");
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
  return 8;
}
void master_boundaries(at::Device device) {
  at::NoGradGuard guard;auto f=fixture(device,3,at::kHalf);
  f.registry.value("a").fill_(65504.f);
  OptimizerGroup group;group.parameters={"a","b"};group.lr=1.;group.momentum=.5;
  DeviceOptimizer optimizer(f.gradient,DeviceOptimizerKind::sgd,{group},1024*1024);
  auto error=at::zeros({1},f.gradient.values.options().dtype(at::kInt));
  CannProgram program(device);optimizer.append_step(program,f.gradient,error);program.finish();
  auto snapshot=[](const DeviceOptimizer& owner){const auto s=owner.snapshot();return
    std::vector<Tensor>{s.values,s.first,s.second,s.maximum,s.steps,s.corrections};};
  auto unchanged=[&](const auto& before,const DeviceOptimizer& owner){const auto after=snapshot(owner);
    for(size_t i=0;i<after.size();++i)require(at::equal(after[i].view(at::kByte),before[i].view(at::kByte)),"FP16 master transaction mutated a live field");};
  f.gradient.values.zero_();f.gradient.connected.zero_();
  for(size_t i=0;i<f.gradient.owners.size();++i) {
    const auto& owner=f.gradient.owners[i];
    if(owner.canonical=="a"||owner.canonical=="b") {
      f.gradient.connected[i].fill_(true);
      f.gradient.values.narrow(0,f.gradient.offsets[i],owner.value.numel()).fill_(owner.canonical=="a"?-8.f:-.25f);
    }
  }
  portable_torch::synchronize(device);program.run();
  require(!error.cpu().item<int>(),"finite rounded half master was over-rejected");
  const auto master=optimizer.values().cpu();
  require(master[0].item<float>()==65512.f&&master[0].to(at::kHalf).item<float>()==65504.f,
    "FP32 master lost sub-ULP progress or published a nonfinite payload");
  const auto before=snapshot(optimizer);const auto checkpoint=optimizer.snapshot();
  DeviceOptimizer restored(f.gradient,DeviceOptimizerKind::sgd,{group},1024*1024);
  restored.restore(checkpoint);unchanged(before,restored);
  auto bad=optimizer.snapshot();bad.values[0].fill_(65520.f);bool rejected=false;
  try{restored.restore(bad);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"nonrepresentable FP16 checkpoint master accepted");unchanged(before,restored);
  // The second momentum update overflows half while remaining finite FP32.
  // Every owner, slot and counter must remain at the first complete update.
  portable_torch::synchronize(device);program.run();
  require(error.cpu().item<int>()==tide_device::optimizer_finite_error,"half publication overflow not refused");
  unchanged(before,optimizer);
  error.zero_();f.gradient.connected.zero_();f.gradient.values.fill_(std::numeric_limits<float>::quiet_NaN());
  portable_torch::synchronize(device);program.run();
  require(!error.cpu().item<int>(),"None poison reached half representability gate");unchanged(before,optimizer);
}
void memory_calibration(at::Device device,DeviceOptimizerKind kind,bool extra) {
  at::NoGradGuard guard;constexpr int64_t n=1048579;
  auto value=at::full({n},.125f,at::kFloat);ParameterRegistry registry;registry.add("weight",value);
  auto f=at::TensorOptions().device(device).dtype(at::kFloat);
  ParameterVjp gradient;gradient.owners=registry.owners();gradient.offsets={0};
  gradient.values=at::full({n},.0625f,f);gradient.connected=at::ones({1},f.dtype(at::kBool));
  OptimizerGroup group;group.parameters={"weight"};group.lr=.003125;group.weight_decay=.0125;group.eps=1e-5;
  group.momentum=extra?.875:0.;group.amsgrad=extra;
  const int64_t banks=kind==DeviceOptimizerKind::adamw?(extra?4:3):(extra?2:1);
  // One live set plus bounded tile/metadata/operator overhead. A full proposal
  // set would exceed this envelope, even for SGD without momentum.
  const int64_t budget=4*banks*n+2*1024*1024;
  portable_torch::synchronize(device);
  auto baseline=c10_npu::NPUCachingAllocator::getDeviceStats(device.index()).allocated_bytes[0].current;
  c10_npu::NPUCachingAllocator::resetPeakStats(device.index());
  DeviceOptimizer optimizer(gradient,kind,{group},budget);
  auto error=at::zeros({1},f.dtype(at::kInt));CannProgram p(device);
  p.limit_workspace(1024*1024);optimizer.append_step(p,gradient,error);p.finish();
  std::unique_ptr<NamedOptimizer> reference;
  if(kind==DeviceOptimizerKind::sgd)reference=std::make_unique<SGD>(registry,std::vector<OptimizerGroup>{group});
  else reference=std::make_unique<AdamW>(registry,std::vector<OptimizerGroup>{group});
  for(int step=0;step<3;++step) {
    const bool live=step!=1;gradient.connected.fill_(live);
    gradient.values.fill_(live?(step==2?0.f:.0625f):std::numeric_limits<float>::quiet_NaN());
    value.mutable_grad()=live?at::full_like(value,step==2?0.f:.0625f):Tensor{};
    portable_torch::synchronize(device);p.run();reference->step();
    require(!error.cpu().item<int>(),"large recomputed update refused");same(optimizer.values().cpu(),value,"large master");
    if(extra||kind==DeviceOptimizerKind::adamw) {
      const auto& state=reference->state().at("weight");
      same(optimizer.first().cpu(),kind==DeviceOptimizerKind::sgd?state.momentum_buffer:state.exp_avg,"large first");
      if(kind==DeviceOptimizerKind::adamw){same(optimizer.second().cpu(),state.exp_avg_sq,"large second");if(extra)same(optimizer.maximum().cpu(),state.max_exp_avg_sq,"large maximum");}
    }
    require(optimizer.steps().cpu().item<int64_t>()==(step==2?2:1),"large update counter raced value tiles");
  }
  const auto before=optimizer.snapshot();gradient.connected.fill_(true);gradient.values.zero_();
  gradient.values[n-1].fill_(std::numeric_limits<float>::infinity());portable_torch::synchronize(device);p.run();
  require(error.cpu().item<int>()==tide_device::optimizer_finite_error,"last-tile infinity was accepted");
  const auto after=optimizer.snapshot();
  for(auto pair:{std::make_pair(before.values,after.values),std::make_pair(before.first,after.first),std::make_pair(before.second,after.second),
      std::make_pair(before.maximum,after.maximum),std::make_pair(before.steps,after.steps),std::make_pair(before.corrections,after.corrections)})
    require(at::equal(pair.first.view(at::kByte),pair.second.view(at::kByte)),"last tile partially committed another tile");
  portable_torch::synchronize(device);
  const auto peak=c10_npu::NPUCachingAllocator::getDeviceStats(device.index()).allocated_bytes[0].peak-baseline;
  require(peak<=budget,"optimizer retained parameter-sized proposals");
  std::cout<<"optimizer-memory: {\"elements\":"<<n<<",\"kind\":"<<int(kind)<<",\"extra_slot\":"<<extra
    <<",\"budget\":"<<budget<<",\"peak_allocated_delta\":"<<peak<<",\"program_workspace_bytes\":"<<p.workspace_bytes()<<"}\n";
  p.close();
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
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("device optimizer gate requires explicit NPU FP32/FP16 payload owners");
    args.allow_npu_float16=true;
    const auto device=portable_torch::resolve_device(args);if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("optimizer gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0,updates=0,refusals_count=0;
    for(auto kind:{DeviceOptimizerKind::sgd,DeviceOptimizerKind::adamw})for(int variant=0;variant<4;++variant)
      for(int64_t width:{1,257})for(auto dtype:{at::kFloat,at::kDouble}) {
        try{const int accepted=trajectory(device,kind,variant,width,dtype,args.dtype);updates+=accepted;refusals_count+=accepted<8;++cases;}
        catch(...){std::cerr<<"optimizer kind="<<int(kind)<<" variant="<<variant<<" width="<<width<<" reference="<<dtype<<'\n';throw;}
      }
    refusals(device);if(args.dtype==at::kHalf)master_boundaries(device);
    else for(auto kind:{DeviceOptimizerKind::sgd,DeviceOptimizerKind::adamw})for(bool extra:{false,true})memory_calibration(device,kind,extra);
    std::cout<<"device-optimizer: passed trajectories="<<cases<<" updates="<<updates<<" expected_half_refusals="<<refusals_count
      <<" CPU=FP32_FP64 master_dtype=FP32 finite_transaction=true half_boundaries="<<(args.dtype==at::kHalf)
      <<" scope=owner_updates_not_public_training\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
