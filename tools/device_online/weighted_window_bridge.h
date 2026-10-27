#pragma once
#include "weighted_window_vjp.h"
namespace tide::device_online {
// Add the later window's LEFT state/pending adjoints to the earlier window's
// RIGHT roots. The caller supplies consecutive actual snapshots, seeds direct
// roots first, then runs this bridge before the earlier window reverse.
void append_weighted_window_bridge(DeviceProgram&,const WeightedWindowSnapshot& earlier,
  WeightedWindowRoots&,const WeightedWindowSnapshot& later,const WeightedWindowAdjoint&,
  const at::Tensor& error,int64_t tensor_budget_bytes);
// Saved parameter snapshots have private TensorImpls. Validate their complete
// alias partition, then bind their partials to the training generation owner.
ParameterVjp bind_weighted_window_parameters(const ParameterVjp&,const ParameterRegistry&);
}
