#pragma once
#include "tide/weighted_sequence_columns.h"
#include "weighted_detail.h"
#include <algorithm>
#include <limits>
namespace tide::weighted::columns {
inline Tensor indices(const std::vector<Index>& ids,const at::Device& device=at::Device(at::kCPU)) {
  return at::tensor(ids,at::TensorOptions().dtype(at::kLong)).to(device);
}
inline std::vector<Index> host_ids(const Tensor& t) {
  auto h=t.to(at::kCPU).contiguous();
  detail::require(h.scalar_type()==at::kLong,"column metadata requires int64");
  if(!h.numel())return {};
  auto p=h.const_data_ptr<Index>();return {p,p+h.numel()};
}
inline Tensor gather(const Tensor& rows,const Tensor& ids) {
  auto shape=ids.sizes().vec(),zero=rows.sizes().vec();zero[0]=1;
  shape.insert(shape.end(),rows.sizes().begin()+1,rows.sizes().end());
  return at::cat({rows,at::zeros(zero,rows.options())}).index_select(0,ids.flatten()).reshape(shape);
}
inline std::pair<Tensor,Tensor> padded(const Layout& layout,const at::Device& device) {
  Index width=1;
  for(Index b=0;b<layout.samples();++b)width=std::max(width,layout.offsets[b+1]-layout.offsets[b]);
  std::vector<Index> rows(layout.samples()*width,layout.times.size()),reverse;
  for(Index b=0;b<layout.samples();++b)for(Index j=layout.offsets[b];j<layout.offsets[b+1];++j){
    rows[b*width+j-layout.offsets[b]]=j;reverse.push_back(b*width+j-layout.offsets[b]);
  }
  return {indices(rows,device).reshape({layout.samples(),width}),indices(reverse,device)};
}
inline Tensor project(const Tensor& x,const Tensor& w){return at::matmul(x,w.t());}
}
