#pragma once
// Detached bool reads for already complete groups. Never replace controls,
// numerical roots, their aliases, or int64 event metadata with host values.
#include <ATen/ATen.h>
#include <map>
#include <tuple>
#include <vector>

namespace tide::region_metadata {
inline std::vector<bool> group_flags(const std::vector<at::Tensor>& values,bool positive){
  bool scalar=true;for(const auto& value:values)scalar=scalar&&value.numel()==1;
  std::vector<at::Tensor> rows;rows.reserve(values.size());
  for(const auto& value:values)
    rows.push_back(positive||scalar?value.detach().reshape({}):at::isfinite(value.detach()).all());
  auto packed=at::stack(rows);
  auto flags=(positive?packed.gt(0):scalar?at::isfinite(packed):packed).to(at::kCPU).contiguous();
  const auto* data=flags.const_data_ptr<bool>();std::vector<bool> result;result.reserve(values.size());
  for(size_t i=0;i<values.size();++i)result.push_back(data[i]);
  return result;
}
inline std::vector<bool> flags(const std::vector<at::Tensor>& values,bool positive){
  using Key=std::tuple<int,int,int>;std::map<Key,std::vector<size_t>> groups;
  std::vector<bool> result(values.size());
  for(size_t i=0;i<values.size();++i){const auto& value=values[i];
    groups[{int(value.device().type()),int(value.device().index()),int(value.scalar_type())}].push_back(i);}
  for(const auto& [key,ids]:groups){
    if(values[ids[0]].device().is_cpu()||ids.size()==1){
      for(auto i:ids)result[i]=positive?values[i].detach().item<double>()>0:at::isfinite(values[i]).all().item<bool>();
    }else{
      std::vector<at::Tensor> rows;for(auto i:ids)rows.push_back(values[i]);
      const auto read=group_flags(rows,positive);for(size_t i=0;i<ids.size();++i)result[ids[i]]=read[i];
    }
  }
  return result;
}
}  // namespace tide::region_metadata
