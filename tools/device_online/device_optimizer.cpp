#include "device_backend.h"
#include "device_optimizer.h"
#include "optimizer_layout.h"
#include "device_launch_tide_optimizer_plan.h"
#include "device_launch_tide_optimizer_values.h"
#include "device_launch_tide_optimizer_commit.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void buffer(const at::Tensor& x,at::Device device,at::ScalarType dtype,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=dtype||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid device optimizer buffer");
}
}
DeviceOptimizer::DeviceOptimizer(const ParameterVjp& gradient,DeviceOptimizerKind kind,
    std::vector<OptimizerGroup> groups,int64_t budget):identity_(gradient),kind_(kind) {
  if(at::GradMode::is_enabled()||!gradient.values.defined()||gradient.values.dim()!=1||budget<1
      ||gradient.values.device().type()!=tide::device_online::resident_device_type||gradient.offsets.size()!=gradient.owners.size())
    throw std::invalid_argument("device optimizer requires bounded explicit no-grad NPU owners");
  const auto device=gradient.values.device();count_=gradient.owners.size();const auto physical=std::max<int64_t>(1,count_);
  buffer(gradient.values,device,at::kFloat,gradient.values.sizes());buffer(gradient.connected,device,at::kBool,{physical});
  ParameterRegistry registry;
  for(const auto& owner:gradient.owners) {
    if(!owner.value.device().is_cpu()||(owner.value.scalar_type()!=at::kFloat&&owner.value.scalar_type()!=at::kHalf))
      throw std::invalid_argument("device optimizer requires CPU FP32/FP16 payload owners");
    // The portable NamedOptimizer validates canonical groups against FP32
    // masters. Original payload TensorImpl identities remain in identity_.
    auto master=owner.value.to(at::kFloat);
    for(const auto& name:owner.aliases)registry.add(name,master);
  }
  std::unique_ptr<NamedOptimizer> validator;
  if(kind==DeviceOptimizerKind::sgd)validator=std::make_unique<SGD>(registry,std::move(groups));
  else if(kind==DeviceOptimizerKind::adamw)validator=std::make_unique<AdamW>(registry,std::move(groups));
  else throw std::invalid_argument("unknown device optimizer kind");
  groups_=validator->groups();std::map<std::string,int64_t> group_of;
  std::vector<float> options;std::vector<int64_t> flags;
  bool momentum=false,maximum=false;
  for(size_t i=0;i<groups_.size();++i) {
    const auto& g=groups_[i];for(const auto& name:g.parameters)group_of.emplace(name,i);
    for(double value:{g.lr,g.weight_decay,g.momentum,1-g.dampening,g.beta1,1-g.beta1,g.beta2,1-g.beta2,g.eps,1-g.lr*g.weight_decay}) {
      if(!std::isfinite(value)||!std::isfinite(static_cast<float>(value)))throw std::invalid_argument("optimizer option exceeds FP32 range");
      options.push_back(static_cast<float>(value));
    }
    flags.insert(flags.end(),{g.nesterov,g.amsgrad,g.maximize});momentum|=g.momentum!=0;maximum|=g.amsgrad;
  }
  std::vector<int64_t> table,tiles{0};int64_t used=0;std::set<const void*> storage;
  for(size_t i=0;i<gradient.owners.size();++i) {
    const auto& owner=gradient.owners[i];const auto offset=gradient.offsets[i],size=owner.value.numel();
    if(size<1||offset < -1
        ||offset>=0&&(offset!=used||size>gradient.values.numel()-offset))
      throw std::invalid_argument("invalid optimizer payload owner/offset");
    // Separate TensorImpl owners can overlap storage. The gradient registry
    // preserves that distinction, but independent packed updates cannot yet
    // reproduce sequential shared-storage mutations. Refuse before allocation.
    if(!storage.insert(owner.value.storage().unsafeGetStorageImpl()).second)
      throw std::invalid_argument("device optimizer does not support distinct owners sharing storage");
    if(offset>=0)used+=size;
    const auto found=group_of.find(owner.canonical);const int64_t group=found==group_of.end()?-1:found->second;
    table.insert(table.end(),{offset,size,group,int64_t(owner.value.scalar_type()==at::kHalf)});
    tiles.push_back(tiles.back()+(offset>=0&&group>=0?(size+255)/256:0));
  }
  if(std::max<int64_t>(1,used)!=gradient.values.numel())throw std::invalid_argument("optimizer owner extent mismatch");
  tasks_=tiles.back();const bool adam=kind==DeviceOptimizerKind::adamw;
  const int64_t elements=gradient.values.numel(),first=(adam||momentum)?elements:1,second=adam?elements:1,max=adam&&maximum?elements:1;
  const long double bytes=4.L*(elements+static_cast<long double>(first)+second+max)+32.L*physical+
    64.L*std::max<int64_t>(1,tasks_)+8.L*(std::max<size_t>(tide_device::OWNER_FIELDS,table.size())+tiles.size()+flags.size())+4.L*options.size();
  if(bytes>budget)throw std::invalid_argument("device optimizer tensor budget exceeded");
  auto cpu=at::zeros({elements},at::kFloat);
  for(size_t i=0;i<gradient.owners.size();++i)if(gradient.offsets[i]>=0)
    cpu.narrow(0,gradient.offsets[i],gradient.owners[i].value.numel()).copy_(gradient.owners[i].value.detach().reshape({-1}));
  values_=cpu.to(device);first_=at::zeros({first},values_.options());second_=at::zeros({second},values_.options());maximum_=at::zeros({max},values_.options());
  steps_=at::zeros({physical},values_.options().dtype(at::kLong));corrections_=at::zeros({physical,2},values_.options());
  next_steps_=at::empty_like(steps_);next_corrections_=at::empty_like(corrections_);
  // A separate cache line per vector tile avoids concurrent scalar cache-line writes.
  tile_errors_=at::zeros({std::max<int64_t>(1,tasks_),16},values_.options().dtype(at::kInt));
  table_=at::tensor(table.empty()?std::vector<int64_t>{-1,0,-1,0}:table,at::kLong).reshape({-1,tide_device::OWNER_FIELDS}).to(device);
  tiles_=at::tensor(tiles,at::kLong).to(device);options_=at::tensor(options,at::kFloat).reshape({-1,tide_device::OPTION_COUNT}).to(device);
  flags_=at::tensor(flags,at::kLong).reshape({-1,tide_device::FLAG_COUNT}).to(device);
  // Identity checks need names/TensorImpl/offsets, not the initial gradient
  // storage. Retaining it here keeps an entire obsolete FP32 parameter bank.
  identity_.values=at::Tensor{};identity_.connected=at::Tensor{};
}
void DeviceOptimizer::append_step(DeviceProgram& p,const ParameterVjp& g,const at::Tensor& error) {
  append_propose(p,g,error);append_commit(p,g,error);
}
void DeviceOptimizer::append_propose(DeviceProgram& p,const ParameterVjp& g,const at::Tensor& error){append_phase(p,g,error,false);}
void DeviceOptimizer::append_commit(DeviceProgram& p,const ParameterVjp& g,const at::Tensor& error){append_phase(p,g,error,true);}
void DeviceOptimizer::append_phase(DeviceProgram& p,const ParameterVjp& g,const at::Tensor& error,bool commit) {
  const auto device=values_.device();buffer(error,device,at::kInt,{1});
  buffer(g.values,device,at::kFloat,values_.sizes());buffer(g.connected,device,at::kBool,steps_.sizes());
  if(at::GradMode::is_enabled()||g.offsets!=identity_.offsets||g.owners.size()!=identity_.owners.size())
    throw std::invalid_argument("optimizer gradient registry differs from construction");
  for(size_t i=0;i<g.owners.size();++i)if(g.owners[i].aliases!=identity_.owners[i].aliases||
      g.owners[i].value.unsafeGetTensorImpl()!=identity_.owners[i].value.unsafeGetTensorImpl())
    throw std::invalid_argument("optimizer gradient owner changed");
  const auto table=table_,tiles=tiles_,options=options_,flags=flags_,values=values_,first=first_,second=second_,maximum=maximum_;
  const auto steps=steps_,corrections=corrections_;
  const auto nt=next_steps_,nc=next_corrections_,errors=tile_errors_;const auto count=count_,tasks=tasks_,kind=int64_t(kind_);
  auto plan=[&](int64_t mode){p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_optimizer_plan)(1,stream,
    ptr(table),ptr(options),ptr(g.connected),ptr(steps),ptr(corrections),ptr(nt),ptr(nc),ptr(errors),ptr(error),count,tasks,kind,mode),
    "device optimizer preflight/finite gate");},{table,options,g.connected,steps,corrections,nt,nc,errors,error});};
  if(!commit)plan(0);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_optimizer_values)(32,stream,
    ptr(table),ptr(tiles),ptr(options),ptr(flags),ptr(g.values),ptr(g.connected),ptr(values),ptr(first),ptr(second),ptr(maximum),ptr(steps),ptr(nc),
    ptr(errors),ptr(error),count,tasks,kind,int64_t(commit)),"evaluate or recompute finite device optimizer update");},
    {table,tiles,options,flags,g.values,g.connected,values,first,second,maximum,steps,nc,errors,error});
  if(!commit)plan(1);
  else p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_optimizer_commit)(1,stream,
    ptr(table),ptr(g.connected),ptr(steps),ptr(corrections),ptr(nt),ptr(nc),ptr(error),count,kind),"commit device optimizer counters after all value tiles");},
    {table,g.connected,steps,corrections,nt,nc,error});
}
} // namespace tide::device_online
