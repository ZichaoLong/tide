#include "weighted_training_parameters.h"
#include <ATen/core/grad_mode.h>
#include <limits>
namespace tide::device_online {
ParameterVjp weighted_training_layout(const ParameterRegistry& registry,at::Device device,int64_t budget){
  if(at::GradMode::is_enabled()||device.type()!=resident_device_type||device.index()<0||budget<1)
    throw std::invalid_argument("weighted training layout requires bounded explicit no-grad device");
  ParameterVjp out;out.owners=registry.owners();long double count=0;
  for(const auto& owner:out.owners){
    if(!owner.value.device().is_cpu()||(owner.value.scalar_type()!=at::kFloat&&owner.value.scalar_type()!=at::kHalf)||owner.value.numel()<1)
      throw std::invalid_argument("weighted training requires CPU FP32/FP16 canonical parameters");
    if(count>std::numeric_limits<int64_t>::max()-owner.value.numel())throw std::invalid_argument("weighted parameter extent overflow");
    out.offsets.push_back(static_cast<int64_t>(count));count+=owner.value.numel();
  }
  const auto physical=std::max<int64_t>(1,out.owners.size());count=std::max(1.L,count);
  if(4*count+physical>budget)throw std::invalid_argument("weighted training parameter layout exceeds budget");
  auto options=at::TensorOptions().device(device).dtype(at::kFloat);
  out.values=at::zeros({static_cast<int64_t>(count)},options);out.connected=at::zeros({physical},options.dtype(at::kBool));return out;
}
ParameterVjp append_weighted_training_parameters(DeviceProgram& p,const ParameterVjp& part,
    const ParameterVjp& layout,int64_t budget){
  ParameterRegistry registry;for(const auto& owner:layout.owners)for(const auto& alias:owner.aliases)registry.add(alias,owner.value);
  bind_weighted_window_parameters(part,registry);
  if(at::GradMode::is_enabled()||layout.values.device()!=part.values.device()||part.values.scalar_type()!=at::kFloat
      ||layout.offsets.size()!=layout.owners.size()||part.offsets.size()!=part.owners.size()
      ||!part.values.is_contiguous()||part.values.requires_grad()||budget<1
      ||4.L*layout.values.numel()+layout.connected.numel()>budget)
    throw std::invalid_argument("invalid weighted complete parameter partial/budget");
  ParameterVjp out=layout;out.values=at::zeros_like(layout.values);out.connected=at::zeros_like(layout.connected);
  if(part.connected.sizes()!=out.connected.sizes()||part.connected.scalar_type()!=at::kBool||part.connected.device()!=out.values.device())
    throw std::invalid_argument("weighted parameter connection layout differs");
  p.zero(out.values);p.copy(out.connected,part.connected);
  for(size_t i=0;i<out.owners.size();++i){const auto from=part.offsets[i],to=out.offsets[i],size=out.owners[i].value.numel();
    if(from < -1||from>=0&&size>part.values.numel()-from||to<0||size>out.values.numel()-to)
      throw std::invalid_argument("weighted parameter partial slice differs");
    if(from>=0)p.copy(out.values.narrow(0,to,size),part.values.narrow(0,from,size));
  }
  return out;
}
std::vector<ParameterWrite> weighted_training_writes(const ParameterRegistry& all,const ParameterVjp& layout,
    const std::vector<WeightedBankRole>& roles){
  std::map<std::string,int64_t> offsets;for(size_t i=0;i<layout.owners.size();++i)offsets[layout.owners[i].canonical]=layout.offsets.at(i);
  std::vector<ParameterWrite> writes;
  for(const auto& role:roles){const auto x=all.value(role.parameter);
    if(x.sizes().vec()!=role.original_shape||role.offset<0||role.offset>x.numel()||role.bank.numel()>x.numel()-role.offset)
      throw std::invalid_argument("weighted publication role differs from original parameter");
    const auto found=offsets.find(all.canonical_name(role.parameter));if(found==offsets.end())continue; // Frozen owner.
    if(found->second<0)throw std::invalid_argument("weighted training publication omitted a trainable owner");
    writes.push_back({found->second+role.offset,{role.bank,x.scalar_type()}});
  }
  return writes;
}
}
