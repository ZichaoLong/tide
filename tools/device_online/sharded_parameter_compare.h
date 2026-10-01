#pragma once
#include "sharded_parameter_reduce.h"
namespace tide::device_online::test {
// Assertion boundary only: download already reduced canonical owners, without
// summing or altering their candidate gradients on the host.
inline std::map<std::string,Tensor> sharded_parameter_observations(const std::vector<ParameterVjp>& partitions) {
  std::map<std::string,Tensor> out;
  for(const auto& p:partitions) {
    auto values=p.values.cpu(),on=p.connected.cpu();
    for(size_t i=0;i<p.owners.size();++i) {
      const auto& owner=p.owners[i];Tensor value;
      if(on[i].item<bool>()) {
        if(p.offsets[i]<0)throw std::runtime_error("unallocated canonical gradient connected");
        value=values.narrow(0,p.offsets[i],owner.value.numel()).reshape(owner.value.sizes()).clone();
      }
      if(!out.emplace(owner.canonical,value).second)throw std::runtime_error("canonical gradient duplicated across cards");
    }
  }
  return out;
}
} // namespace tide::device_online::test
