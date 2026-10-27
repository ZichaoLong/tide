#pragma once
#include "weighted_kv_update.h"
#include "device_launch_tide_weighted_source_guard.h"
#include "device_launch_tide_weighted_allocation_status.h"
#include "device_launch_tide_weighted_head_reduce.h"
#include <ATen/core/grad_mode.h>
#include <array>
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
inline at::Tensor gather(DeviceProgram& p,const at::Tensor& x,const at::Tensor& indices,const at::Tensor& roots,const at::Tensor& error){
  auto shape=x.sizes().vec();shape[0]=indices.numel();auto rows=at::zeros(shape,x.options());p.index_select(x,0,indices,rows);return guard(p,rows,roots,error);
}
inline at::Tensor transpose(DeviceProgram& p,const at::Tensor& x){auto t=at::zeros({x.size(0),x.size(2),x.size(1)},x.options());p.permute(x,{0,2,1},t);return t;}
inline void reduce(DeviceProgram& p,const at::Tensor& owners,const at::Tensor& roots,
    const std::array<at::Tensor,3>& input,const std::array<at::Tensor,3>& output,
    const at::Tensor& connected,const at::Tensor& error){
  const auto rows=owners.numel(),banks=connected.numel();auto errors=at::zeros({banks},error.options());
  std::array<at::Tensor,3> x,y;std::array<int64_t,3> widths;
  std::vector<at::Tensor> keep{owners,roots,connected,errors,error};
  for(int i=0;i<3;++i){x[i]=input[i].defined()?input[i]:error;y[i]=output[i].defined()?output[i]:error;
    widths[i]=input[i].defined()?input[i].numel()/rows:0;keep.push_back(x[i]);keep.push_back(y[i]);}
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_head_reduce)(1,stream,
    ptr(owners),ptr(roots),ptr(error),ptr(x[0]),ptr(x[1]),ptr(x[2]),ptr(error),ptr(y[0]),ptr(y[1]),ptr(y[2]),ptr(error),
    ptr(connected),ptr(errors),ptr(error),rows,banks,0,widths[0],widths[1],widths[2],0),"KV shared actual parameter role reduction");},keep);
  status(p,errors,error);
}
}
