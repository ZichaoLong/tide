#pragma once
// Shared parameters and private state: architecture §2.2, implementation checks §1.3.
#include "parameter_vjp.h"
namespace tide::device_online {
struct WeightedParameterPartial {
  std::string name;
  at::Tensor values,connected; // actual FP32 role gradient and one bool root
};
// Each entry is one actual use/role contribution; repeated aliases accumulate.
// Resolves names using TensorImpl identity, never equality of numerical values.
// Output uses the existing ParameterVjp/DeviceOptimizer canonical-owner contract.
ParameterVjp append_weighted_parameter_vjp(DeviceProgram&,const ParameterRegistry&,
  const std::vector<WeightedParameterPartial>&,const at::Tensor& error,
  int64_t tensor_budget_bytes=256*1024*1024);
}
