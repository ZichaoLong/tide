#pragma once
#include "weighted_kv_update.h"
#include "device_launch_tide_weighted_source_guard.h"
#include "device_launch_tide_weighted_allocation_status.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online::kv_update_detail {
inline uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
inline void check(const at::Tensor& t,at::Device d,at::ScalarType type,at::IntArrayRef shape){
  if(!t.defined()||t.device()!=d||t.scalar_type()!=type||t.sizes()!=shape||!t.is_contiguous()||t.requires_grad())
    throw std::invalid_argument("invalid resident KV update buffer");
}
inline void budget(long double bytes,int64_t available){if(available<1||bytes>available)throw std::invalid_argument("resident KV update tensor budget exceeded");}
inline void status(DeviceProgram& p,const at::Tensor& rows,const at::Tensor& error){
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_allocation_status)(1,stream,
    ptr(rows),ptr(error),rows.numel()),"KV all-row decision");},{rows,error});
}
inline at::Tensor guard(DeviceProgram& p,const at::Tensor& x,const at::Tensor& roots,const at::Tensor& error){
  auto clean=at::zeros_like(x),errors=at::zeros({x.size(0)},error.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_source_guard)(1,stream,
    ptr(x),ptr(roots),ptr(clean),ptr(errors),ptr(error),x.size(0),x.numel()/x.size(0),1),"KV actual-role finite guard");},
    {x,roots,clean,errors,error});status(p,errors,error);return clean;
}
}
