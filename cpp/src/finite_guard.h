#pragma once
// Bounded detached validation of already metadata-checked runtime tensors.
// Do not publish a batch before every chunk has been checked.
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace tide {
inline bool all_finite(const std::vector<at::Tensor>& values,int64_t budget=8*1024*1024){
  if(budget<1)throw std::invalid_argument("finite guard requires a positive byte budget");
  at::NoGradGuard guard;
  std::map<std::tuple<int,int,int>,std::vector<at::Tensor>> groups;
  for(const auto& t:values)groups[{int(t.device().type()),t.device().index(),int(t.scalar_type())}].push_back(t);
  auto finite=[](const at::Tensor& t){return at::isfinite(t).all().item<bool>();};
  for(const auto& [key,tensors]:groups){
    std::vector<at::Tensor> rows;int64_t bytes=0;
    for(const auto& t:tensors){const auto count=t.numel()*t.element_size();
      if(!rows.empty()&&bytes+count>budget){if(!finite(at::cat(rows)))return false;rows.clear();bytes=0;}
      if(count>budget){if(!finite(t))return false;}
      else{rows.push_back(t.reshape({-1}));bytes+=count;}
    }
    if(!rows.empty()&&!finite(at::cat(rows)))return false;
  }
  return true;
}
}
