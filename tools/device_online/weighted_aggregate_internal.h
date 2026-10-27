#pragma once
#include "weighted_aggregate.h"
#include "device_launch_tide_weighted_allocation_status.h"
#include <ATen/core/grad_mode.h>
#include <cmath>
namespace tide::device_online::aggregate_detail {
inline uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
inline void check(const at::Tensor& x,at::IntArrayRef shape,at::ScalarType type,at::Device device){
  if(!x.defined()||x.sizes()!=shape||x.scalar_type()!=type||x.device()!=device||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid resident weighted Aggregate buffer");
}
struct Shape {
  int64_t rows,width,d;
  Shape(const at::Tensor& x,float alpha,float epsilon){
    if(!x.defined()||x.dim()!=3||x.size(0)<1||x.size(1)<1||x.size(2)<2||at::GradMode::is_enabled()
        ||x.device().type()!=resident_device_type)throw std::invalid_argument("resident Aggregate requires accelerator fibers and explicit no-grad");
    validate_kernel_device(x.device());check(x,x.sizes(),at::kFloat,x.device());
    if(!std::isfinite(alpha)||alpha<0||!std::isfinite(epsilon)||epsilon<=0)
      throw std::invalid_argument("invalid resident Aggregate alpha/epsilon");
    rows=x.size(0);width=x.size(1);d=x.size(2)-1;
  }
};
inline void admit(long double required,int64_t budget){
  if(budget<1||required>budget)throw std::invalid_argument("resident Aggregate scratch budget exceeded");
}
inline void status(DeviceProgram& p,const at::Tensor& rows,const at::Tensor& error){
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_allocation_status)(
    1,stream,ptr(rows),ptr(error),rows.numel()),"Aggregate all-row decision");},{rows,error});
}
}
