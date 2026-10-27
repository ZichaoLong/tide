#include "weighted_window_vjp.h"
namespace tide::device_online::weighted_window {
WeightedRowAdjoint row(const at::Tensor& x,int64_t leading){
  auto shape=x.sizes().vec();shape.resize(leading);
  return {at::zeros_like(x),at::zeros(shape,x.options().dtype(at::kBool))};
}
WeightedRawAdjoint raw(const WeightedRawLayout& l,const at::Tensor& x,int64_t leading){
  auto shape=x.sizes().vec();shape.resize(leading);shape.push_back(l.fields().size());
  return {at::zeros_like(x),at::zeros(shape,x.options().dtype(at::kBool))};
}
at::Tensor load(DeviceProgram& p,const at::Tensor& stage,const at::Tensor& x){
  auto shape=x.sizes().vec();shape.erase(shape.begin());auto result=at::zeros(shape,x.options());
  p.index_select(x,0,stage,result.unsqueeze(0));return result;
}
WeightedRowAdjoint load(DeviceProgram& p,const at::Tensor& stage,const WeightedRowAdjoint& x){
  return {load(p,stage,x.value),load(p,stage,x.connected)};
}
WeightedRawAdjoint load(DeviceProgram& p,const at::Tensor& stage,const WeightedRawAdjoint& x){
  return {load(p,stage,x.value),load(p,stage,x.connected)};
}
void route(DeviceProgram& p,const AtomBatch& atoms,const WeightedRowAdjoint& root,const at::Tensor& edges,
    const WeightedReverseEvents& events,const WeightedReverseBoundary& boundary,const at::Tensor& stage,
    const at::Tensor& begin,const at::Tensor& error){
  const auto count=atoms.valid.numel();auto indices=at::arange(count,stage.options()).reshape({1,count});
  auto lengths=at::full({1},count,stage.options());
  append_weighted_reverse_route(p,atoms,indices,lengths,root.value.unsqueeze(0),root.connected.unsqueeze(0),
    edges,events,boundary,stage,begin,error);
}
WeightedRawAdjoint mask(DeviceProgram& p,const WeightedRawLayout& layout,const WeightedRawAdjoint& source,
    const at::Tensor& present,const at::Tensor& error){
  auto out=weighted_reverse::zeros(p,layout,source.value);auto owners=at::zeros_like(present);
  p.copy(owners,present);p.add(owners,at::full_like(owners,-1));
  for(size_t j=0;j<layout.fields().size();++j){const auto& field=layout.fields()[j];const auto width=field.width*(field.ragged?layout.capacity():1);
    auto target=weighted_reverse::zeros(p,at::zeros({source.value.size(0),width},source.value.options()));
    weighted_reverse::add(p,target,{weighted_columns(p,source.value,field.offset,width),
      weighted_columns(p,source.connected,j,1).reshape({-1})},owners,error);
    weighted_copy(p,out.value.narrow(1,field.offset,width),target.value);weighted_copy(p,out.connected.select(1,j),target.connected);
  }
  return out;
}
}
