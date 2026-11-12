#pragma once
// Architecture §2: fixed unit transport, grouped guards without value caching.
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace tide::weighted {
namespace unit_transport_detail {
template<class Accept>
void bounded_views(const at::Tensor& value,int64_t capacity,const Accept& accept){
  const auto count=value.numel();
  if(!count)return;
  if(count<=capacity){accept(value);return;}
  if(value.is_contiguous()){
    auto flat=value.view({-1});
    for(int64_t start=0;start<count;){
      const auto size=std::min(capacity,count-start);accept(flat.narrow(0,start,size));start+=size;
    }
    return;
  }
  int64_t axis=0;while(value.size(axis)<=1)++axis;
  const auto width=std::max<int64_t>(1,capacity/(count/value.size(axis)));
  for(int64_t start=0;start<value.size(axis);){
    const auto size=std::min(width,value.size(axis)-start);
    bounded_views(value.narrow(axis,start,size),capacity,accept);start+=size;
  }
}
}

inline void validate_unit_scales(const std::vector<at::Tensor>& values,
                                int64_t budget=8*1024*1024,bool fixed=true){
  // Explicit temporary storage: bounded reshape copies + cat + all-ones peer.
  // Backend workspace and retained allocator blocks are not this byte budget.
  // Empty tensors keep equal(empty, ones_like(empty))'s vacuous value semantics.
  if(budget<1)throw std::invalid_argument("unit transport budget must be a positive int64 byte count");
  constexpr auto message="weighted payload transport requires fixed unit scales";
  std::map<std::tuple<int,int,int>,std::vector<at::Tensor>> groups;
  for(const auto& value:values){
    if(fixed&&value.requires_grad())throw std::invalid_argument(message);
    groups[{int(value.device().type()),value.device().index(),int(value.scalar_type())}].push_back(value);
  }
  at::NoGradGuard guard;
  for(const auto& [key,tensors]:groups){
    const auto capacity=budget/(3*tensors.front().element_size());
    if(capacity<1&&std::any_of(tensors.begin(),tensors.end(),[](const auto& t){return t.numel()!=0;}))
      throw std::invalid_argument("unit transport budget cannot hold one element");
    std::vector<at::Tensor> rows;int64_t count=0;
    auto check=[&](){
      std::vector<at::Tensor> flat;flat.reserve(rows.size());
      for(const auto& row:rows)flat.push_back(row.reshape({-1}));
      auto packed=at::cat(flat);
      if(!at::equal(packed,at::ones_like(packed)))throw std::invalid_argument(message);
    };
    auto accept=[&](const at::Tensor& view){
      if(!rows.empty()&&view.numel()>capacity-count){check();rows.clear();count=0;}
      rows.push_back(view);count+=view.numel();
    };
    for(const auto& tensor:tensors)unit_transport_detail::bounded_views(tensor,capacity,accept);
    if(!rows.empty())check();
  }
}
}
