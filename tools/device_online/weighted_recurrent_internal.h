#pragma once
#include "weighted_recurrent.h"
namespace tide::device_online::recurrent_detail {
inline uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
inline void check(const at::Tensor& x,at::Device d,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=d||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid resident recurrent buffer");
}
inline at::Tensor columns(DeviceProgram& p,const at::Tensor& x,int64_t offset,int64_t count){
  auto out=at::zeros({x.size(0),count},x.options()),ids=at::arange(offset,offset+count,x.options().dtype(at::kLong));
  p.index_select(x,1,ids,out);return out;
}
inline void insert(DeviceProgram& p,const at::Tensor& out,int64_t offset,const at::Tensor& x){
  auto ids=at::arange(offset,offset+x.size(1),out.options().dtype(at::kLong));p.index_copy(out,1,ids,x);
}
void recurrent_algebra(WeightedRecurrentTape&,const WeightedRecurrentParameters&,DeviceProgram&,
  const at::Tensor& times,const at::Tensor& error,int64_t budget);
}
