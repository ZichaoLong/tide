#include "weighted_parameter_vjp.h"
#include "device_launch_tide_weighted_parameter_reduce.h"
#include "device_launch_tide_weighted_allocation_status.h"
#include <ATen/core/grad_mode.h>
#include <cstring>
#include <limits>
#include <map>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
int64_t address(const at::Tensor& t){const auto p=reinterpret_cast<uint64_t>(t.data_ptr());int64_t a;std::memcpy(&a,&p,sizeof(a));return a;}
void check(const at::Tensor& t,at::Device d,at::ScalarType type){
  if(!t.defined()||t.device()!=d||t.scalar_type()!=type||!t.is_contiguous()||t.requires_grad())
    throw std::invalid_argument("invalid resident weighted parameter partial");
}
}
ParameterVjp append_weighted_parameter_vjp(DeviceProgram& p,const ParameterRegistry& registry,
    const std::vector<WeightedParameterPartial>& partials,const at::Tensor& error,int64_t budget){
  if(at::GradMode::is_enabled()||!error.defined()||error.sizes()!=at::IntArrayRef{1}
      ||error.device().type()!=resident_device_type||budget<1)
    throw std::invalid_argument("weighted parameter reduction requires explicit bounded accelerator no-grad");
  const auto device=error.device();validate_kernel_device(device);check(error,device,at::kInt);
  ParameterVjp result;result.owners=registry.owners();std::map<std::string,std::vector<const WeightedParameterPartial*>> groups;
  std::vector<at::Tensor> keep{error};
  for(const auto& part:partials){
    const auto owner=registry.value(part.name);
    if(owner.scalar_type()!=at::kFloat&&owner.scalar_type()!=at::kHalf)
      throw std::invalid_argument("weighted parameter owner must be FP32/FP16");
    check(part.values,device,at::kFloat);check(part.connected,device,at::kBool);
    if(part.values.sizes()!=owner.sizes()||part.values.numel()<1||part.connected.numel()!=1)
      throw std::invalid_argument("weighted parameter partial shape/root differs from owner");
    groups[registry.canonical_name(part.name)].push_back(&part);keep.push_back(part.values);keep.push_back(part.connected);
  }
  std::vector<int64_t> table,references,tiles{0};int64_t used=0;
  for(const auto& owner:result.owners){
    if(!owner.value.defined()||owner.value.numel()<1
        ||(owner.value.scalar_type()!=at::kFloat&&owner.value.scalar_type()!=at::kHalf))
      throw std::invalid_argument("invalid weighted parameter registry owner");
    const auto& sources=groups[owner.canonical];const auto start=references.size()/2;const auto size=owner.value.numel();
    const int64_t offset=sources.empty()?-1:used;
    if(offset>=0){if(size>std::numeric_limits<int64_t>::max()-used)throw std::overflow_error("weighted parameter layout overflow");used+=size;}
    result.offsets.push_back(offset);for(const auto* source:sources)references.insert(references.end(),{address(source->values),address(source->connected)});
    table.insert(table.end(),{static_cast<int64_t>(start),static_cast<int64_t>(references.size()/2),offset,size});
    const int64_t added=offset>=0?size/256+(size%256!=0):0;
    if(added>std::numeric_limits<int64_t>::max()-tiles.back())throw std::overflow_error("weighted parameter tiles overflow");
    tiles.push_back(tiles.back()+added);
  }
  const int64_t count=result.owners.size(),tasks=tiles.back();
  if(tasks>std::numeric_limits<int64_t>::max()/16)throw std::overflow_error("weighted parameter error layout overflow");
  const auto physical=std::max<int64_t>(1,count),elements=std::max<int64_t>(1,used),errors_size=16*std::max<int64_t>(1,tasks);
  if(4.L*elements+physical+4.L*errors_size+8.L*(std::max<size_t>(4,table.size())+std::max<size_t>(2,references.size())+tiles.size())>budget)
    throw std::invalid_argument("weighted parameter reduction tensor budget exceeded");
  auto options=error.options();auto owners=at::tensor(table.empty()?std::vector<int64_t>(4):table,at::kLong).to(device);
  auto refs=at::tensor(references.empty()?std::vector<int64_t>(2):references,at::kLong).to(device);
  auto offsets=at::tensor(tiles,at::kLong).to(device),errors=at::zeros({errors_size},options);
  result.values=at::zeros({elements},options.dtype(at::kFloat));result.connected=at::zeros({physical},options.dtype(at::kBool));
  p.zero(errors);p.zero(result.values);p.zero(result.connected);
  keep.insert(keep.end(),{owners,refs,offsets,errors,result.values,result.connected});
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_parameter_reduce)(32,stream,
    ptr(owners),ptr(refs),ptr(offsets),ptr(result.values),ptr(result.connected),ptr(errors),ptr(error),count,tasks),
    "weighted canonical parameter role reduction");},keep);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_allocation_status)(1,stream,
    ptr(errors),ptr(error),errors_size),"weighted parameter finite decision");},{errors,error});return result;
}
}
