#pragma once
#include "parameter_vjp.h"
namespace tide::device_online {
struct ParameterPlan {
  std::vector<ParameterOwner> owners;
  std::vector<int64_t> offsets,owner_table,references,tiles;
  int64_t elements;
  bool has_tanh;
};
ParameterPlan plan_parameters(const Graph&,const ParameterRegistry&,int64_t width,int64_t tensor_budget_bytes);
} // namespace tide::device_online
