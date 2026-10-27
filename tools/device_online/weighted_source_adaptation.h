#pragma once
// 0770593 architecture §2.2: effective A/B with private low-rank factors.
#include "weighted_aggregate.h"
namespace tide::device_online {
struct WeightedSourceFactors {at::Tensor a_left,a_right,b_left,b_right;};
struct WeightedSourceAdaptation {
  WeightedSourceParameters effective;
  WeightedSourceFactors saved;
};
struct WeightedSourceAdaptationGradient {
  WeightedSourceParameters base;
  WeightedSourceFactors factors;
  at::Tensor connected;
};
// All banks have the same d/H/rank in this numerical group. Parameter aliases
// across banks/roles remain an outer canonical-owner reduction responsibility.
WeightedSourceAdaptation append_weighted_source_adaptation(DeviceProgram&,
  const WeightedSourceParameters&,const WeightedSourceFactors&,const at::Tensor& error,
  int64_t scratch_budget=256*1024*1024);
WeightedSourceAdaptationGradient append_weighted_source_adaptation_vjp(DeviceProgram&,
  const WeightedSourceAdaptation&,const WeightedSourceParameters& gradients,
  const at::Tensor& roots,const at::Tensor& error,int64_t scratch_budget=256*1024*1024);
}
