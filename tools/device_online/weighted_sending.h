#pragma once
#include "device_program.h"
#include "weighted_sending_layout.h"
namespace tide::device_online {
struct WeightedSendingTape {
  at::Tensor controls,details,statistics,branches,lengths;
  weighted_sending::Kind kind;
  weighted_sending::Base base;
  int64_t k;
};
struct WeightedSendingGradient {
  at::Tensor vectors,scalars,connected; // [score,cap,received], [total,tau], five roots.
};
// Complete candidate rows; lengths/IDs are int64, padding is never a candidate.
// Hard receives adjusted scores. SortedFill receives the already computed B.
// k=0 for Attention/SortedFill, k>0 for Proportional/Hard. Inapplicable base=CapBase.
WeightedSendingTape append_weighted_sending(DeviceProgram&,const at::Tensor& scores,
  const at::Tensor& caps,const at::Tensor& received,const at::Tensor& parameters,
  const at::Tensor& ids,const at::Tensor& lengths,const at::Tensor& error,
  weighted_sending::Kind,weighted_sending::Base=weighted_sending::CapBase,int64_t k=0,
  int64_t scratch_budget=64*1024*1024);
WeightedSendingGradient append_weighted_sending_vjp(DeviceProgram&,const WeightedSendingTape&,
  const at::Tensor& cotangent,const at::Tensor& roots,const at::Tensor& error,
  int64_t scratch_budget=64*1024*1024);
}
