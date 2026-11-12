#pragma once
#include "block_kv_bank.h"
#include "weighted_storage_budget.h"
namespace tide::device_online::block_bank_detail {
inline uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
inline void check(const at::Tensor& x,const at::Tensor& e,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=e.device()||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid shared KV bank operand");
}
inline void admit(long double bytes,int64_t budget){if(budget<1||bytes>budget)throw std::invalid_argument("shared KV bank tensor budget exceeded");}
inline at::Tensor owner_order(DeviceProgram& p,const BlockKVSnapshot& s,const at::Tensor& x){
  const auto& b=*s.bank;const auto width=x.numel()/(b.batch*s.length*b.nodes);
  auto out=at::zeros({b.batch,b.nodes,s.length,width},x.options());
  p.permute(x.reshape({b.batch,s.length,b.nodes,width}),{0,2,1,3},out);return out.reshape({b.batch*b.nodes,s.length,width});
}
inline at::Tensor row_order(DeviceProgram& p,const BlockKVSnapshot& s,const at::Tensor& x){
  const auto& b=*s.bank;const auto width=x.numel()/(b.batch*s.length*b.nodes);
  auto out=at::zeros({b.batch,s.length,b.nodes,width},x.options());
  p.permute(x.reshape({b.batch,b.nodes,s.length,width}),{0,2,1,3},out);return out.reshape({b.batch*s.length*b.nodes,width});
}
inline at::Tensor transpose(DeviceProgram& p,const at::Tensor& x){
  auto out=at::zeros({x.size(0),x.size(2),x.size(1)},x.options());p.permute(x,{0,2,1},out);return out;
}
inline at::Tensor columns(DeviceProgram& p,const at::Tensor& x,int64_t start,int64_t width){
  auto out=at::zeros({x.size(0),x.size(1),width},x.options());
  p.index_select(x,2,at::arange(start,start+width,x.options().dtype(at::kLong)),out);return out;
}
inline void put(DeviceProgram& p,const at::Tensor& x,int64_t start,const at::Tensor& value){
  p.index_copy(x,2,at::arange(start,start+value.size(2),x.options().dtype(at::kLong)),value);
}
}
