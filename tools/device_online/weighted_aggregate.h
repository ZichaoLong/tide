#pragma once
// 0770593 architecture §1/2; docs/resident-weighted-aggregate.md.
#include "weighted_head.h"
#include <optional>
namespace tide::device_online {
// Effective A/B/e parameter banks. Different semantic roles/owners may alias in
// the outer parameter binding; return every role contribution for that reduction.
struct WeightedSourceParameters {at::Tensor a,b,embedding;};
struct WeightedAggregateTape {
  at::Tensor payload,lengths,owners,clean,normal,rstd,total,values,output;
  std::optional<WeightedHeadTape> source;
  float alpha,epsilon;
};
struct WeightedAggregateGradient {
  at::Tensor payload,connected; // [frames,capacity,d+1], bool [frames,capacity]
  std::optional<WeightedSourceParameters> parameters;
  at::Tensor owner_connected;
};
// Each row is a complete fiber. Dynamic lengths may be zero (no invocation),
// whereas a nonempty fiber with zero positive mass remains a real candidate.
// owners selects the source-function bank per physical atom, not a uniqueness
// constraint. Actual payloads are (m,z), never already multiplied m*z.
WeightedAggregateTape append_weighted_aggregate(DeviceProgram&,const at::Tensor& payload,
  const at::Tensor& lengths,const at::Tensor& owners,const WeightedSourceParameters&,
  float alpha,float epsilon,const at::Tensor& error,int64_t scratch_budget=256*1024*1024);
WeightedAggregateGradient append_weighted_aggregate_vjp(DeviceProgram&,const WeightedAggregateTape&,
  const at::Tensor& cotangent,const at::Tensor& roots,const at::Tensor& error,
  int64_t scratch_budget=256*1024*1024);
}
