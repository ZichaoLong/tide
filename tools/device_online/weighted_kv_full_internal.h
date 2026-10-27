#pragma once
#include "weighted_kv_full.h"
#include <cmath>
namespace tide::device_online::kv_full_detail {
inline uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
inline void check(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid resident KV Full buffer");
}
inline void budget(long double bytes,int64_t available){if(available<1||bytes>available)throw std::invalid_argument("resident KV Full tensor budget exceeded");}
// Each per-head row remains a contiguous actual operand for vendor BMM/pool.
inline at::Tensor split(DeviceProgram& p,const at::Tensor& input,int64_t heads){
  const auto r=input.size(0),c=input.dim()==3?input.size(1):1,d=input.size(-1),dh=d/heads;
  auto out=at::zeros({heads,r,c,dh},input.options());p.permute(input.reshape({r,c,heads,dh}),{2,0,1,3},out);return out;
}
inline at::Tensor merge(DeviceProgram& p,const at::Tensor& input){
  const auto heads=input.size(0),r=input.size(1),c=input.size(2),dh=input.size(3);
  auto out=at::zeros({r,c,heads,dh},input.options());p.permute(input,{1,2,0,3},out);return out.reshape({r,c,heads*dh});
}
}
