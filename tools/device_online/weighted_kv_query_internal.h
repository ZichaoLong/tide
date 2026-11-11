#pragma once
#include "weighted_kv_query.h"
#include <cmath>
#include <sstream>
namespace tide::device_online::kv_query_detail {
inline uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
inline void check(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid resident KV qry buffer");
}
inline void budget(long double bytes,int64_t available,const std::string& detail={}){if(available<1||bytes>available){
  std::ostringstream message;message<<"resident KV qry tensor budget exceeded: required="<<std::fixed<<bytes<<" available="<<available;
  if(!detail.empty())message<<" "<<detail;throw std::invalid_argument(message.str());}}
inline long double reservation(int64_t rows,int64_t capacity,int64_t d,int64_t ds,int64_t banks){
  // Single-bank projections retain matrix/transpose once, not per cache row.
  // Keep the conservative per-element factor for the remaining child tapes.
  if(banks==1)return 256.L*(rows*(capacity*(d+ds+1.L)+1)+3.L*d*ds);
  return rows*(256.L*capacity*(static_cast<long double>(d)*ds+d+ds+1)+256);
}
inline at::Tensor transpose(DeviceProgram& p,const at::Tensor& x){auto out=at::zeros({x.size(0),x.size(2),x.size(1)},x.options());p.permute(x,{0,2,1},out);return out;}
inline at::Tensor repeat(DeviceProgram& p,const at::Tensor& x,const at::Tensor& ids){auto shape=x.sizes().vec();shape[0]=ids.numel();auto out=at::zeros(shape,x.options());p.index_select(x,0,ids,out);return out;}
}
