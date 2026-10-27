#pragma once
#include "tide/weighted.h"
#include <cmath>
#include <stdexcept>

namespace tide::weighted::detail {
inline void require(bool ok,const char* message){if(!ok)throw std::invalid_argument(message);}
inline const Tensor& weight(const NodeWeights& w,const std::string& name){return w.extra.at(name);}
inline void shape(const NodeWeights& w,const std::string& name,at::IntArrayRef dims){
  const auto& value=weight(w,name);
  require(value.sizes()==dims&&value.device()==w.bias.device()&&value.scalar_type()==w.bias.scalar_type()
          &&at::isfinite(value).all().item<bool>(),"invalid weighted parameter shape/dtype/device/value");
}
inline std::pair<Tensor,Tensor> visible(const State& s,Index time,const KVConfig& c){
  auto age=time-s.slots.at("times");auto mask=age>=0;
  if(c.window)mask=mask&(age<c.window);
  return {mask,age};
}
inline Tensor rotate(const Tensor& keys,const Tensor& displacement,const std::vector<double>& frequencies){
  if(frequencies.empty())return keys;
  const Index n=frequencies.size(),d=keys.size(-1);
  auto phase=displacement.unsqueeze(-1)*at::tensor(frequencies,keys.options());
  auto pairs=keys.slice(-1,d-2*n).reshape({keys.size(0),n,2});
  auto a=pairs.select(-1,0),b=pairs.select(-1,1);
  auto rotated=at::stack({a*phase.cos()-b*phase.sin(),a*phase.sin()+b*phase.cos()},-1).flatten(-2);
  return at::cat({keys.slice(-1,0,d-2*n),rotated},-1);
}
inline Tensor weighted_read(const Tensor& logits,const Tensor& values,const Tensor& weights,const Tensor& total){
  auto positive=weights>0;
  if(!positive.any().item<bool>())return at::zeros({values.size(-1)},values.options());
  auto probabilities=(logits.index({positive})+weights.index({positive}).log()).softmax(0);
  return total.clamp_max(1)*(probabilities.unsqueeze(-1)*values.index({positive})).sum(0);
}
inline Tensor head(const std::map<std::string,Tensor>& w,const std::string& prefix,const Tensor& x){
  return w.at(prefix+".bias")+(at::silu(at::matmul(x,w.at(prefix+".matrix").t())+w.at(prefix+".hidden_bias"))*w.at(prefix+".vector")).sum(-1);
}
}  // namespace tide::weighted::detail
