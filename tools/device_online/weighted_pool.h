#pragma once
#include "device_program.h"
namespace tide::device_online {
// KV §3.2/3.3/4.1: sum/max(1,W), or min(1,W)*positive weighted Attention.
// Lengths/presence describe actual ragged rows; they are not inferred from zeros.
struct WeightedPoolTape {
  at::Tensor weights,values,meta,probabilities,coefficients,total,mean,output;
  bool attention=false;
};
struct WeightedPoolGradient {
  at::Tensor weights,values,logits;
  at::Tensor connected; // [R,3]: weight, value, logit; independent of numeric zero.
};
WeightedPoolTape append_weighted_pool(DeviceProgram&,const at::Tensor& weights,
  const at::Tensor& values,const at::Tensor& logits,const at::Tensor& lengths,
  const at::Tensor& present,const at::Tensor& error,int64_t budget=256*1024*1024);
// Undefined logits selects sum; otherwise shape [R,C] selects Attention.
// roots[R,2] independently selects the vector output and the scalar total.
WeightedPoolGradient append_weighted_pool_vjp(DeviceProgram&,const WeightedPoolTape&,
  const at::Tensor& vector_bar,const at::Tensor& total_bar,const at::Tensor& roots,
  const at::Tensor& error,int64_t budget=256*1024*1024);
}
