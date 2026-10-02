#pragma once
#include "graph_vjp.h"
#include "tide/parameters.h"

namespace tide::device_online {
struct ParameterVjp {
  // Registry order and TensorImpl alias identity are static host metadata.
  // An offset of -1 means this profile has no differentiable use of that owner.
  // Otherwise a slice exists even when its device connection bit is false.
  std::vector<ParameterOwner> owners;
  std::vector<int64_t> offsets;
  at::Tensor values,connected;
};
// Static shape/alias planning; no numerical graph execution.
ParameterVjp parameter_layout(const Graph&,const ParameterRegistry&,int64_t width,at::Device,int64_t tensor_budget_bytes,bool controls=false);
// Sum graph partials into physical parameter owners on device. The registry
// must describe the same single graph/model used to create the forward owner;
// it may explicitly select only the trainable subset. No numerical CPU prepass.
ParameterVjp append_parameter_vjp(CannProgram&,const Graph&,const ParameterRegistry&,
                                 const GraphVjp&,const at::Tensor& error,int64_t tensor_budget_bytes);
ParameterVjp append_parameter_accumulate(CannProgram&,const ParameterVjp&,const ParameterVjp&,
                                        const at::Tensor& error,int64_t tensor_budget_bytes);
// Only for a private, unexported left bank, disjoint from the right bank.
// Each tile reads/writes its own values; connection flags use new storage so
// block zero cannot race other blocks reading the old connection state.
// This is a consuming update, not an idempotent replay of the original inputs.
ParameterVjp append_private_parameter_accumulate(CannProgram&,const ParameterVjp&,const ParameterVjp&,
                                                const at::Tensor& error,int64_t tensor_budget_bytes);
} // namespace tide::device_online
