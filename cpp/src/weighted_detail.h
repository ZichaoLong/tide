#pragma once
#include "weighted_attention_mean.h"
#include "tide/weighted.h"
#include <cmath>
#include <stdexcept>

namespace tide::weighted::detail {
inline void require(bool ok,const char* message){if(!ok)throw std::invalid_argument(message);}
inline const Tensor& weight(const NodeWeights& w,const std::string& name){return w.extra.at(name);}
Tensor scaled_rms(const Tensor& value,const Tensor& scale,double epsilon);
Tensor epsilon_l2(const Tensor& value,double epsilon);
inline Tensor constants(const std::vector<double>& values,const Tensor& like){
  // LibTorch's host tensor factory has no Half vector overload. Quantize the
  // literal double values once, exactly as Python new_tensor(..., float16).
  if(like.scalar_type()==at::kHalf)return at::tensor(values,at::TensorOptions().dtype(at::kDouble)).to(like.options());
  return at::tensor(values,like.options());
}
inline Tensor time_values(const Tensor& difference,const Tensor& like){
  return difference.to(like.scalar_type()==at::kHalf?at::kFloat:like.scalar_type());
}
inline Tensor passive_decay(const Tensor& difference,const std::vector<double>& rates,const Tensor& like){
  auto clock=time_values(difference,like);
  return at::exp(-clock*constants(rates,clock)).to(like.scalar_type());
}
inline Tensor passive_decay(const Tensor& difference,double rate,const Tensor& like){
  return at::exp(-time_values(difference,like)*rate).to(like.scalar_type());
}
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
  auto x=keys.scalar_type()==at::kHalf?keys.to(at::kFloat):keys;
  auto phase=displacement.unsqueeze(-1)*constants(frequencies,x);
  auto pairs=x.slice(-1,d-2*n).reshape({keys.size(0),n,2});
  auto a=pairs.select(-1,0),b=pairs.select(-1,1);
  auto rotated=at::stack({a*phase.cos()-b*phase.sin(),a*phase.sin()+b*phase.cos()},-1).flatten(-2);
  return at::cat({x.slice(-1,0,d-2*n),rotated},-1).to(keys.scalar_type());
}
inline Tensor weighted_read(const Tensor& logits,const Tensor& values,const Tensor& weights,const Tensor& total){
  auto positive=weights>0;
  if(!positive.any().item<bool>())return at::zeros({values.size(-1)},values.options());
  return total.clamp_max(1)*attention_mean(logits.index({positive})+weights.index({positive}).log(),values.index({positive}));
}
inline Tensor head(const std::map<std::string,Tensor>& w,const std::string& prefix,const Tensor& x){
  return w.at(prefix+".bias")+(at::silu(at::matmul(x,w.at(prefix+".matrix").t())+w.at(prefix+".hidden_bias"))*w.at(prefix+".vector")).sum(-1);
}
}  // namespace tide::weighted::detail
