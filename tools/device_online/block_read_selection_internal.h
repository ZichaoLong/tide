#pragma once
#include "block_read_selection.h"
#include "weighted_storage_budget.h"
#include <stdexcept>
namespace tide::device_online::block_selection_detail {
inline uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
inline void check(const at::Tensor& x,const at::Tensor& like,at::ScalarType dtype,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=like.device()||x.scalar_type()!=dtype||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid block read-selection tensor");
}
inline void admit(long double bytes,int64_t budget){if(budget<1||bytes>budget)throw std::invalid_argument("block read-selection tensor budget exceeded");}
inline at::Tensor gather(DeviceProgram& p,const at::Tensor& x,const at::Tensor& ids){
  // One sentinel row, shared across the entire mapping, never one Tensor/event.
  auto flat=x.reshape({x.size(0),-1});
  auto joined=at::zeros({flat.size(0)+1,flat.size(1)},x.options());p.copy(joined.narrow(0,0,flat.size(0)),flat);
  auto out=at::zeros({ids.numel(),flat.size(1)},x.options());p.index_select(joined,0,ids.reshape({-1}),out);return out;
}
inline at::Tensor columns(DeviceProgram& p,const at::Tensor& x,int64_t start,int64_t width){
  auto out=at::zeros({x.size(0),width},x.options());
  p.index_select(x,1,at::arange(start,start+width,x.options().dtype(at::kLong)),out);return out;
}
inline void put(DeviceProgram& p,const at::Tensor& target,int64_t start,const at::Tensor& source){
  p.index_copy(target,1,at::arange(start,start+source.size(1),target.options().dtype(at::kLong)),source);
}
inline std::vector<at::Tensor> borrowed(const BlockReadSelectionInput& x,const BlockReadSelectionParameters& w,const at::Tensor& error){
  std::vector<at::Tensor> out{x.read.floating,x.read.integer,x.mass,x.content,x.present,x.times,x.node_owners,
    w.query.query,w.query.key,w.query.value,error,w.region.context_owners,w.region.query_owners};
  for(const auto& [_,t]:w.describe)out.push_back(t);
  for(const auto& h:{w.region.context,w.region.query})for(const auto& t:{h.matrix,h.hidden_bias,h.projection,h.bias})out.push_back(t);
  if(x.bank){auto a=block_kv_bank_tensors(*x.bank->bank);out.insert(out.end(),a.begin(),a.end());out.insert(out.end(),{x.bank->mapping,x.bank->integer});}
  return out;
}
}
