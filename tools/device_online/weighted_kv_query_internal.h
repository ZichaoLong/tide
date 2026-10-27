#pragma once
#include "weighted_kv_query.h"
#include <cmath>
namespace tide::device_online::kv_query_detail {
inline uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
inline void check(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid resident KV qry buffer");
}
inline void budget(long double bytes,int64_t available){if(available<1||bytes>available)throw std::invalid_argument("resident KV qry tensor budget exceeded");}
inline at::Tensor transpose(DeviceProgram& p,const at::Tensor& x){auto out=at::zeros({x.size(0),x.size(2),x.size(1)},x.options());p.permute(x,{0,2,1},out);return out;}
inline at::Tensor repeat(DeviceProgram& p,const at::Tensor& x,const at::Tensor& ids){auto shape=x.sizes().vec();shape[0]=ids.numel();auto out=at::zeros(shape,x.options());p.index_select(x,0,ids,out);return out;}
}
