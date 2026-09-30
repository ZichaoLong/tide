#include "parameter_vjp.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_parameter_accumulate.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <stdexcept>

namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
ParameterVjp append_parameter_accumulate(CannProgram& p,const ParameterVjp& a,const ParameterVjp& b,
    const at::Tensor& error,int64_t budget) {
  if(at::GradMode::is_enabled()||a.owners.size()!=b.owners.size()||a.offsets!=b.offsets||a.offsets.size()!=a.owners.size()
      ||!a.values.defined()||a.values.dim()!=1||budget<1)
    throw std::invalid_argument("parameter accumulation requires matching no-grad owner layouts");
  const auto device=a.values.device();const int64_t count=a.owners.size(),physical=std::max<int64_t>(1,count);
  if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("parameter accumulation requires NPU FP32");
  for(const auto& x:{a.values,b.values})if(!x.defined()||x.device()!=device||x.scalar_type()!=at::kFloat||x.sizes()!=a.values.sizes()||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid parameter accumulation values");
  for(const auto& x:{a.connected,b.connected})if(!x.defined()||x.device()!=device||x.scalar_type()!=at::kBool||x.sizes()!=at::IntArrayRef{physical}||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid parameter accumulation connections");
  if(!error.defined()||error.device()!=device||error.scalar_type()!=at::kInt||error.sizes()!=at::IntArrayRef{1}||!error.is_contiguous()||error.requires_grad())
    throw std::invalid_argument("invalid parameter accumulation error");
  std::vector<int64_t> table,tiles{0};int64_t used=0;
  for(size_t i=0;i<a.owners.size();++i) {
    const auto& x=a.owners[i];const auto& y=b.owners[i];const auto offset=a.offsets[i],size=x.value.numel();
    if(x.aliases!=y.aliases||x.canonical!=y.canonical||x.value.unsafeGetTensorImpl()!=y.value.unsafeGetTensorImpl()||size<1||offset < -1
        ||offset>=0&&(offset!=used||size>a.values.numel()-offset))throw std::invalid_argument("parameter accumulation owner differs");
    if(offset>=0)used+=size;
    table.insert(table.end(),{offset,size});tiles.push_back(tiles.back()+(offset>=0?(size+255)/256:0));
  }
  if(a.values.numel()!=std::max<int64_t>(1,used))throw std::invalid_argument("parameter accumulation extent mismatch");
  const long double bytes=4.L*a.values.numel()+physical+8.L*(std::max<size_t>(2,table.size())+tiles.size());
  if(bytes>budget)throw std::invalid_argument("parameter accumulation tensor budget exceeded");
  ParameterVjp out{a.owners,a.offsets,at::empty_like(a.values),at::empty_like(a.connected)};p.zero(out.values);p.zero(out.connected);
  auto owners=at::tensor(table.empty()?std::vector<int64_t>{-1,0}:table,at::kLong).reshape({-1,2}).to(device);
  auto offsets=at::tensor(tiles,at::kLong).to(device);const auto tasks=tiles.back();
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_parameter_accumulate)(32,stream,
    ptr(owners),ptr(offsets),ptr(a.values),ptr(a.connected),ptr(b.values),ptr(b.connected),ptr(out.values),ptr(out.connected),ptr(error),count,tasks),
    "accumulate retained-window parameter owners");},{owners,offsets,a.values,a.connected,b.values,b.connected,out.values,out.connected,error});
  return out;
}
} // namespace tide::device_online
