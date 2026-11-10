#pragma once
// Independent output allocations/versions, copied by a foreach submission.
#include "tide/types.h"
#include "tide/packing_work.h"
#include <ATen/core/grad_mode.h>
#include <map>
#include <tuple>

namespace tide::weighted {
inline std::vector<Tensor> detached_copies(const std::vector<Tensor>& values){
  std::vector<Tensor> result(values.size());
  std::map<std::tuple<int,int,int>,std::vector<size_t>> groups;
  at::NoGradGuard guard;
  for(size_t i=0;i<values.size();++i){
    const auto& value=values[i];
    if(value.layout()!=at::kStrided){result[i]=value.detach().clone();continue;}
    groups[{int(value.device().type()),value.device().index(),int(value.scalar_type())}].push_back(i);
  }
  for(const auto& [key,ids]:groups){
    std::vector<Tensor> sources,outputs;
    for(auto i:ids){sources.push_back(values[i].detach());outputs.push_back(at::empty_like(values[i],values[i].options().requires_grad(false)));}
    at::_foreach_copy_(outputs,sources);
    for(size_t i=0;i<ids.size();++i)result[ids[i]]=outputs[i];
    work::packing::projection("independent_output_copy",ids.size());
  }
  return result;
}
}
